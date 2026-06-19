"""
Standalone Webcam Pose Classifier
Runs MediaPipe Pose + LightGBM classification directly on webcam feed.
No Flask, no browser -- just OpenCV window with classification overlay.
"""
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

JOINT_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COORD_JOINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28),
    (11, 23, 25), (12, 24, 26),
]
DISTANCE_PAIRS = [
    (11, 12), (23, 24), (11, 13), (12, 14), (13, 15), (14, 16),
    (23, 25), (24, 26), (25, 27), (26, 28), (15, 16), (27, 28),
]

LABEL_6_NAMES = {
    0: "Standing", 1: "Sitting", 2: "Balancing",
    3: "Inverted", 4: "Reclining", 5: "Wheel"
}

LABEL_20_NAMES = {
    0: "Standing Straight", 1: "Forward Bend", 2: "Warrior/Lunge",
    3: "Balancing One Leg", 4: "Seated Upright", 5: "Seated Twist",
    6: "Seated Split", 7: "Seated Forward Bend", 8: "Seated Pigeon",
    9: "Arm Balance", 10: "Side Balance", 11: "Headstand/Forearm",
    12: "Shoulderstand/Plow", 13: "Supine/Recline", 14: "Prone Backbend",
    15: "Side Plank", 16: "Plank/Push-up", 17: "Kneeling Backbend",
    18: "All Fours", 19: "Boat/V-sit"
}

LABEL_82_NAMES = {
    0: "Akarna Dhanurasana", 1: "Bharadvajasana", 2: "Boat Pose",
    3: "Bound Angle Pose", 4: "Bow Pose", 5: "Bridge Pose",
    6: "Camel Pose", 7: "Cat Cow Pose", 8: "Chair Pose",
    9: "Child Pose", 10: "Cobra Pose", 11: "Cockerel Pose",
    12: "Corpse Pose", 13: "Cow Face Pose", 14: "Crane/Crow Pose",
    15: "Dolphin Plank", 16: "Dolphin Pose", 17: "Downward Dog",
    18: "Eagle Pose", 19: "Eight-Angle Pose", 20: "Puppy Pose",
    21: "Revolved Side Angle", 22: "Revolved Triangle",
    23: "Feathered Peacock", 24: "Firefly Pose", 25: "Fish Pose",
    26: "Chaturanga", 27: "Frog Pose", 28: "Garland Pose",
    29: "Gate Pose", 30: "Half Lord of Fishes", 31: "Half Moon Pose",
    32: "Handstand", 33: "Happy Baby Pose", 34: "Head-to-Knee Bend",
    35: "Heron Pose", 36: "Side Stretch", 37: "Legs Up Wall",
    38: "Locust Pose", 39: "Lord of Dance", 40: "Low Lunge",
    41: "Noose Pose", 42: "Peacock Pose", 43: "Pigeon Pose",
    44: "Plank Pose", 45: "Plow Pose", 46: "Sage Koundinya",
    47: "King Pigeon", 48: "Reclining Big Toe", 49: "Revolved Head-to-Knee",
    50: "Scale Pose", 51: "Scorpion Pose", 52: "Seated Forward Bend",
    53: "Shoulder Press", 54: "Side Leg Lift", 55: "Side Crow",
    56: "Side Plank", 57: "Seated Normal", 58: "Split Pose",
    59: "Staff Pose", 60: "Standing Forward Bend", 61: "Standing Split",
    62: "Standing Big Toe Hold", 63: "Headstand", 64: "Shoulderstand",
    65: "Supta Baddha Konasana", 66: "Supta Virasana", 67: "Tortoise Pose",
    68: "Tree Pose", 69: "Wheel Pose", 70: "Two-Foot Staff",
    71: "Upward Plank", 72: "Virasana", 73: "Warrior III",
    74: "Warrior II", 75: "Warrior I", 76: "Wide-Angle Seated Bend",
    77: "Wide-Legged Forward Bend", 78: "Wild Thing", 79: "Wind Relieving",
    80: "Yogic Sleep", 81: "Reverse Warrior"
}


def cosine_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def extract_features(pose_landmarks):
    """Convert MediaPipe pose landmarks to 59-float feature vector."""
    joints = {}
    for idx in JOINT_INDICES:
        lm = pose_landmarks.landmark[idx]
        joints[idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)

    shoulder_mid = (joints[11] + joints[12]) / 2.0
    hip_mid = (joints[23] + joints[24]) / 2.0
    torso_scale = np.linalg.norm(shoulder_mid - hip_mid)

    if torso_scale < 1e-4:
        return None

    # 1. Normalised coordinates (13 joints x 3 = 39)
    coords = []
    for jidx in COORD_JOINT_INDICES:
        xyz = joints[jidx]
        coords.extend([
            (xyz[0] - shoulder_mid[0]) / torso_scale,
            (xyz[1] - shoulder_mid[1]) / torso_scale,
            xyz[2],
        ])

    # 2. Joint angles (7)
    angles = []
    for a_idx, b_idx, c_idx in ANGLE_TRIPLETS:
        angles.append(cosine_angle(joints[a_idx], joints[b_idx], joints[c_idx]))
    angles.append(cosine_angle(shoulder_mid, hip_mid, joints[0]))

    # 3. Pairwise distances (13)
    distances = []
    for a_idx, b_idx in DISTANCE_PAIRS:
        distances.append(float(np.linalg.norm(joints[a_idx] - joints[b_idx]) / torso_scale))
    distances.append(float(np.linalg.norm(joints[0] - hip_mid) / torso_scale))

    return np.array(coords + angles + distances, dtype=np.float32).reshape(1, -1)


def load_models():
    """Load all 3 classifiers from static/."""
    classifiers = {}
    encoders = {}
    for level in ["6", "20", "82"]:
        path = os.path.join(STATIC_DIR, f"classifier_{level}.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            classifiers[level] = data["model"]
            encoders[level] = data["label_encoder"]
            print(f"  Loaded classifier_{level}.joblib")
        else:
            print(f"  WARNING: {path} not found!")
    return classifiers, encoders


def classify(features, classifiers, encoders):
    """Run all 3 classifiers on a feature vector."""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    # Confidence thresholds per level
    THRESHOLDS = {"6": 0.40, "20": 0.35, "82": 0.30}
    results = {}
    for level in ["6", "20", "82"]:
        if level in classifiers:
            pred = classifiers[level].predict(features)[0]
            proba = classifiers[level].predict_proba(features)[0]
            conf = float(np.max(proba))
            label = int(encoders[level].inverse_transform([pred])[0])
            if conf < THRESHOLDS.get(level, 0.3):
                name = "Unknown"
            elif level == "6":
                name = LABEL_6_NAMES.get(label, str(label))
            elif level == "20":
                name = LABEL_20_NAMES.get(label, str(label))
            else:
                name = LABEL_82_NAMES.get(label, str(label))
            results[level] = (name, conf)
    return results


def main():
    print("=" * 50)
    print("  Yoga Pose Classifier -- Webcam")
    print("=" * 50)

    # Load models
    print("\nLoading classifiers...")
    classifiers, encoders = load_models()
    if not classifiers:
        print("ERROR: No classifiers found! Run train_classifiers.py first.")
        return
    print(f"Loaded {len(classifiers)} classifiers.\n")

    # Init MediaPipe
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    print("Camera opened. Press ESC to quit.\n")

    last_classify_time = 0
    current_results = {}
    fps_counter = 0
    fps_time = time.time()
    fps_val = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_val = fps_counter
            fps_counter = 0
            fps_time = time.time()

        if results.pose_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 80, 0), thickness=2),
            )

            # Classify every 300ms
            now = time.time()
            if now - last_classify_time > 0.3:
                last_classify_time = now
                feats = extract_features(results.pose_landmarks)
                if feats is not None:
                    current_results = classify(feats, classifiers, encoders)

        # ── Draw HUD overlay ──
        h, w = frame.shape[:2]

        # Dark background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # FPS
        cv2.putText(frame, f"FPS: {fps_val}", (20, 35),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Classification results
        y = 65
        if current_results:
            for level, label_text, color in [
                ("6", "Body", (0, 200, 255)),
                ("20", "Group", (255, 200, 0)),
                ("82", "Pose", (0, 255, 200)),
            ]:
                if level in current_results:
                    name, conf = current_results[level]
                    text = f"{label_text}: {name} ({conf*100:.0f}%)"
                    cv2.putText(frame, text, (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Confidence bar
                    bar_x = 20
                    bar_w = int(340 * conf)
                    cv2.rectangle(frame, (bar_x, y + 5), (bar_x + bar_w, y + 12), color, -1)
                    cv2.rectangle(frame, (bar_x, y + 5), (bar_x + 340, y + 12), (80, 80, 80), 1)
                    y += 38
        else:
            cv2.putText(frame, "No pose detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        cv2.imshow("Yoga Pose Classifier", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Done.")


if __name__ == "__main__":
    main()
