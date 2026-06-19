from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import time
import threading
import numpy as np
import os
import joblib

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Guard: only one camera thread ever runs
_stream_thread_started = False
_stream_lock = threading.Lock()

# ── Classifier Models (lazy-loaded) ──────────────────────────────────────────
_classifiers = {}
_label_encoders = {}
_models_loaded = False
_models_lock = threading.Lock()

# Joint indices used by the classifier (same as const LM in index.html)
JOINT_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COORD_JOINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Angle triplets and distance pairs (must match build_features.py)
ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28),
    (11, 23, 25), (12, 24, 26),
]
DISTANCE_PAIRS = [
    (11, 12), (23, 24), (11, 13), (12, 14), (13, 15), (14, 16),
    (23, 25), (24, 26), (25, 27), (26, 28), (15, 16), (27, 28),
]

# Label name maps (from Yoga-82 dataset)
LABEL_6_NAMES = {
    0: "Standing", 1: "Sitting", 2: "Balancing",
    3: "Inverted", 4: "Reclining", 5: "Wheel"
}
LABEL_20_NAMES = {}  # Will be populated from label encoder


def _load_models():
    """Load classifier models from static/ directory."""
    global _classifiers, _label_encoders, _models_loaded
    static = os.path.join(os.path.dirname(__file__), "static")

    for level in ["6", "20", "82"]:
        path = os.path.join(static, f"classifier_{level}.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            _classifiers[level] = data["model"]
            _label_encoders[level] = data["label_encoder"]
            print(f"[CLASSIFIER] Loaded classifier_{level}.joblib")
        else:
            print(f"[CLASSIFIER] WARNING: {path} not found")

    _models_loaded = True


def _extract_features_from_landmarks(landmarks):
    """Convert raw landmark list (33 landmarks) to the 59-float feature vector.
    Matches the exact pipeline in build_features.py.
    """
    # Get coordinates for our 15 joints
    joints = {}
    for idx in JOINT_INDICES:
        lm = landmarks[idx]
        joints[idx] = np.array([lm["x"], lm["y"], lm["z"]], dtype=np.float64)

    # Midpoints
    shoulder_mid = (joints[11] + joints[12]) / 2.0
    hip_mid = (joints[23] + joints[24]) / 2.0
    torso_scale = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_scale < 1e-4:
        return None

    # 1. Torso-normalised coordinates (13 joints × 3 = 39)
    coords = []
    for jidx in COORD_JOINT_INDICES:
        xyz = joints[jidx]
        normed = [
            (xyz[0] - shoulder_mid[0]) / torso_scale,
            (xyz[1] - shoulder_mid[1]) / torso_scale,
            xyz[2],
        ]
        coords.extend(normed)

    # 2. Joint angles (7)
    def cosine_angle(a, b, c):
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        return float(np.arccos(cos_a))

    angles = []
    for a_idx, b_idx, c_idx in ANGLE_TRIPLETS:
        angles.append(cosine_angle(joints[a_idx], joints[b_idx], joints[c_idx]))
    # Spine inclination: shoulder_mid → hip_mid → NOSE
    angles.append(cosine_angle(shoulder_mid, hip_mid, joints[0]))

    # 3. Pairwise distances (13)
    distances = []
    for a_idx, b_idx in DISTANCE_PAIRS:
        distances.append(float(np.linalg.norm(joints[a_idx] - joints[b_idx]) / torso_scale))
    # nose → hip_mid
    distances.append(float(np.linalg.norm(joints[0] - hip_mid) / torso_scale))

    feature_vec = coords + angles + distances
    return np.array(feature_vec, dtype=np.float32).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def test_page():
    return render_template('test.html')


@app.route('/classify', methods=['POST'])
def classify():
    """Classify a pose from landmarks.
    Accepts: {"landmarks": [{x, y, z, visibility}, ...]} (33 landmarks)
    Returns: {"label_6": {"class": ..., "confidence": ...}, ...}
    """
    if not _classifiers:
        return jsonify({"error": "No classifiers loaded. Train models first."}), 503

    try:
        data = request.get_json(force=True)
        landmarks = data.get("landmarks")

        if not landmarks or len(landmarks) < 33:
            return jsonify({"error": "Need 33 landmarks"}), 400

        # Extract features
        features = _extract_features_from_landmarks(landmarks)
        if features is None:
            return jsonify({"error": "Degenerate pose (torso too small)"}), 400

        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        result = {}
        for level in ["6", "20", "82"]:
            if level in _classifiers:
                clf = _classifiers[level]
                le = _label_encoders[level]

                pred = clf.predict(features)[0]
                proba = clf.predict_proba(features)[0]
                confidence = float(np.max(proba))
                class_label = int(le.inverse_transform([pred])[0])

                # Use human-readable name for label_6 if available
                if level == "6":
                    class_name = LABEL_6_NAMES.get(class_label, str(class_label))
                else:
                    class_name = str(class_label)

                result[f"label_{level}"] = {
                    "class": class_name,
                    "class_id": class_label,
                    "confidence": round(float(confidence), 4),
                }

        return jsonify(result)
    except Exception as e:
        print(f"[CLASSIFY ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def pose_stream():
    print("[START] Pose stream started")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera failed to open")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Frame read failed -- retrying...")
            time.sleep(0.1)
            continue

        # Mirror view is more natural for self-mirroring avatar
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # Draw skeleton overlay on webcam window
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 80, 0), thickness=2)
            )

            # Pack all 33 landmarks
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility)
                })

            socketio.emit('pose_data', {"landmarks": landmarks})

        cv2.imshow("Pose Detection — Press ESC to quit", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("[STOP] ESC pressed -- stopping camera")
            break

        time.sleep(0.033)  # ~30 FPS cap

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released")


@socketio.on('connect')
def handle_connect(auth=None):
    global _stream_thread_started
    print("[CONN] Client connected")

    # BUG FIX: Only start ONE camera thread regardless of how many clients connect
    with _stream_lock:
        if not _stream_thread_started:
            _stream_thread_started = True
            socketio.start_background_task(pose_stream)
            print("[INFO] Background camera task started")
        else:
            print("[INFO] Camera thread already running -- skipping duplicate start")


@socketio.on('disconnect')
def handle_disconnect():
    print("[DISC] Client disconnected")


if __name__ == '__main__':
    # Load classifiers eagerly at startup (avoid blocking first request)
    print("[STARTUP] Loading classifiers...")
    _load_models()
    _models_loaded = True
    print("[STARTUP] Ready.")
    socketio.run(app, debug=True, use_reloader=False, port=5000, allow_unsafe_werkzeug=True)