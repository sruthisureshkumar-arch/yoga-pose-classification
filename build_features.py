"""
Feature Vector Construction for Yoga-82 Pose Classification
Converts raw MediaPipe landmarks into normalised, pose-invariant features.

Feature vector composition:
  - Torso-normalised coordinates: 13 joints × 3 (x,y,z) = 39 features
    (ears dropped due to frequent occlusion in yoga images)
  - Joint angles (cosine law): 7 angles
  - Pairwise distances (torso-normalised): 13 distances
  Total: 59 features per sample
"""
import os
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
POSE_DIR = os.path.dirname(__file__)  # pose_vison/
LANDMARKS_TRAIN = os.path.join(POSE_DIR, "landmarks_train.csv")
LANDMARKS_TEST = os.path.join(POSE_DIR, "landmarks_test.csv")
FEATURES_TRAIN = os.path.join(POSE_DIR, "features_train.csv")
FEATURES_TEST = os.path.join(POSE_DIR, "features_test.csv")

# Joint indices used (same as extract_landmarks.py)
JOINT_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Joints to KEEP in coordinate features (drop ears: indices 7, 8)
COORD_JOINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COORD_JOINT_NAMES = [
    "NOSE", "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW",
    "L_WRIST", "R_WRIST", "L_HIP", "R_HIP", "L_KNEE", "R_KNEE",
    "L_ANKLE", "R_ANKLE",
]

# Angle triplets: (joint_A, joint_B_vertex, joint_C) → angle at B
ANGLE_TRIPLETS = [
    (11, 13, 15),  # L_SHOULDER → L_ELBOW → L_WRIST (left elbow flex)
    (12, 14, 16),  # R_SHOULDER → R_ELBOW → R_WRIST (right elbow flex)
    (23, 25, 27),  # L_HIP → L_KNEE → L_ANKLE (left knee flex)
    (24, 26, 28),  # R_HIP → R_KNEE → R_ANKLE (right knee flex)
    (11, 23, 25),  # L_SHOULDER → L_HIP → L_KNEE (left hip angle)
    (12, 24, 26),  # R_SHOULDER → R_HIP → R_KNEE (right hip angle)
    # spine inclination: shoulder_mid → hip_mid → NOSE (computed specially)
]
ANGLE_NAMES = [
    "L_elbow_flex", "R_elbow_flex",
    "L_knee_flex", "R_knee_flex",
    "L_hip_angle", "R_hip_angle",
    "spine_incl",
]

# Distance pairs (joint_A, joint_B)
DISTANCE_PAIRS = [
    (11, 12),  # shoulder–shoulder
    (23, 24),  # hip–hip
    (11, 13),  # L shoulder–elbow
    (12, 14),  # R shoulder–elbow
    (13, 15),  # L elbow–wrist
    (14, 16),  # R elbow–wrist
    (23, 25),  # L hip–knee
    (24, 26),  # R hip–knee
    (25, 27),  # L knee–ankle
    (26, 28),  # R knee–ankle
    (15, 16),  # wrist L–R (hand spread)
    (27, 28),  # ankle L–R (foot spread)
    # nose–hip_mid (computed specially)
]
DISTANCE_NAMES = [
    "shldr_shldr", "hip_hip",
    "L_shldr_elbow", "R_shldr_elbow",
    "L_elbow_wrist", "R_elbow_wrist",
    "L_hip_knee", "R_hip_knee",
    "L_knee_ankle", "R_knee_ankle",
    "wrist_spread", "ankle_spread",
    "nose_hip_mid",
]


def get_joint_xyz(row, joint_idx):
    """Extract (x, y, z) for a joint from a landmark CSV row."""
    return np.array([
        row[f"lm{joint_idx}_x"],
        row[f"lm{joint_idx}_y"],
        row[f"lm{joint_idx}_z"],
    ], dtype=np.float64)


def cosine_angle(a, b, c):
    """Compute angle at vertex B given points A, B, C using cosine law."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)  # radians


def build_features(df):
    """Build feature matrix from a landmarks DataFrame."""
    # Drop failed detections
    df_valid = df[df["FAILED"] == 0].copy()
    print(f"  Valid samples: {len(df_valid)} / {len(df)} (dropped {len(df)-len(df_valid)} failed)")

    features_list = []
    labels_list = []
    paths_list = []

    for idx, row in df_valid.iterrows():
        # ── Midpoints ──
        lm11 = get_joint_xyz(row, 11)  # L_SHOULDER
        lm12 = get_joint_xyz(row, 12)  # R_SHOULDER
        lm23 = get_joint_xyz(row, 23)  # L_HIP
        lm24 = get_joint_xyz(row, 24)  # R_HIP

        shoulder_mid = (lm11 + lm12) / 2.0
        hip_mid = (lm23 + lm24) / 2.0
        torso_scale = np.linalg.norm(shoulder_mid - hip_mid)

        if torso_scale < 1e-4:
            # Degenerate pose, skip
            continue

        # ── 1. Torso-normalised coordinates (13 joints × 3 = 39) ──
        coords = []
        for jidx in COORD_JOINT_INDICES:
            xyz = get_joint_xyz(row, jidx)
            # Translate: subtract shoulder_mid for x,y; keep z as-is
            normed = np.array([
                (xyz[0] - shoulder_mid[0]) / torso_scale,
                (xyz[1] - shoulder_mid[1]) / torso_scale,
                xyz[2],  # z is already relative depth from MediaPipe
            ])
            coords.extend(normed.tolist())

        # ── 2. Joint angles (7 angles) ──
        angles = []
        # First 6: standard triplets
        for a_idx, b_idx, c_idx in ANGLE_TRIPLETS:
            a = get_joint_xyz(row, a_idx)
            b = get_joint_xyz(row, b_idx)
            c = get_joint_xyz(row, c_idx)
            angles.append(cosine_angle(a, b, c))

        # 7th: spine inclination (shoulder_mid → hip_mid → NOSE)
        nose = get_joint_xyz(row, 0)
        angles.append(cosine_angle(shoulder_mid, hip_mid, nose))

        # ── 3. Pairwise distances (13 distances) ──
        distances = []
        # First 12: standard pairs
        for a_idx, b_idx in DISTANCE_PAIRS:
            a = get_joint_xyz(row, a_idx)
            b = get_joint_xyz(row, b_idx)
            distances.append(np.linalg.norm(a - b) / torso_scale)

        # 13th: nose → hip_mid
        distances.append(np.linalg.norm(nose - hip_mid) / torso_scale)

        # ── Combine ──
        feature_vec = coords + angles + distances
        features_list.append(feature_vec)
        labels_list.append((row["label_6"], row["label_20"], row["label_82"]))
        paths_list.append(row["image_path"])

    return features_list, labels_list, paths_list


def build_feature_names():
    """Build column names for the feature vector."""
    names = []
    # Coordinates
    for jname in COORD_JOINT_NAMES:
        names += [f"{jname}_x", f"{jname}_y", f"{jname}_z"]
    # Angles
    names += ANGLE_NAMES
    # Distances
    names += DISTANCE_NAMES
    return names


def main():
    print("=" * 60)
    print("Feature Vector Construction for Yoga-82")
    print("=" * 60)

    assert os.path.exists(LANDMARKS_TRAIN), f"Missing: {LANDMARKS_TRAIN}\nRun extract_landmarks.py first!"
    assert os.path.exists(LANDMARKS_TEST), f"Missing: {LANDMARKS_TEST}\nRun extract_landmarks.py first!"

    feature_names = build_feature_names()
    print(f"\nFeature vector length: {len(feature_names)}")
    print(f"  Coordinates: {len(COORD_JOINT_INDICES) * 3} (13 joints × 3)")
    print(f"  Angles: {len(ANGLE_NAMES)}")
    print(f"  Distances: {len(DISTANCE_NAMES)}")

    for split_name, lm_path, out_path in [
        ("TRAIN", LANDMARKS_TRAIN, FEATURES_TRAIN),
        ("TEST", LANDMARKS_TEST, FEATURES_TEST),
    ]:
        print(f"\n{'-'*40} {split_name} {'-'*40}")
        df = pd.read_csv(lm_path)
        print(f"  Loaded {len(df)} rows from {os.path.basename(lm_path)}")

        features, labels, paths = build_features(df)
        print(f"  Built features for {len(features)} samples")

        # Create output DataFrame
        out_df = pd.DataFrame(features, columns=feature_names)
        out_df.insert(0, "image_path", paths)
        out_df.insert(1, "label_6", [l[0] for l in labels])
        out_df.insert(2, "label_20", [l[1] for l in labels])
        out_df.insert(3, "label_82", [l[2] for l in labels])

        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {out_df.shape}")

        # Report class distribution
        for level in ["label_6", "label_20", "label_82"]:
            vc = out_df[level].value_counts()
            print(f"\n  {level} distribution (top 10):")
            for cls, cnt in vc.head(10).items():
                print(f"    class {cls}: {cnt} samples")
            print(f"    ... {len(vc)} classes total, min={vc.min()}, max={vc.max()}, "
                  f"imbalance ratio={vc.max()/max(1,vc.min()):.1f}:1")


if __name__ == "__main__":
    main()
