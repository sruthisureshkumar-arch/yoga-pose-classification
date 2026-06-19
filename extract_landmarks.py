"""
MediaPipe Landmark Extraction for Yoga-82 Dataset
Extracts 15 joints from each image, saves to CSV.
Matches the live app config: model_complexity=1, min_detection_confidence=0.5
"""
import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
YOGA82_DIR = os.path.join(os.path.dirname(__file__), "..", "Yoga-82")
IMAGES_DIR = os.path.join(YOGA82_DIR, "images")
TRAIN_TXT = os.path.join(YOGA82_DIR, "yoga_train.txt")
TEST_TXT = os.path.join(YOGA82_DIR, "yoga_test.txt")
OUTPUT_DIR = os.path.dirname(__file__)  # pose_vison/

# The 15 joints used by the digital twin (const LM in index.html)
JOINT_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
JOINT_NAMES = [
    "NOSE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
]


def build_csv_header():
    """Build CSV header: image_path, labels, then 4 cols per joint, then FAILED."""
    header = ["image_path", "label_6", "label_20", "label_82"]
    for idx in JOINT_INDICES:
        header += [f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z", f"lm{idx}_vis"]
    header.append("FAILED")
    return header


def parse_split_file(txt_path):
    """Parse yoga_train.txt or yoga_test.txt → list of (image_path, l6, l20, l82)."""
    entries = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                entries.append((parts[0].strip(), int(parts[1]), int(parts[2]), int(parts[3])))
    return entries


def extract_landmarks_for_split(entries, output_csv, pose_model):
    """Extract landmarks for all images in a split, write to CSV."""
    header = build_csv_header()
    num_cols = len(JOINT_INDICES) * 4  # 15 joints × 4 values

    total = len(entries)
    success = 0
    failed = 0
    missing = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for img_rel, l6, l20, l82 in tqdm(entries, desc=os.path.basename(output_csv)):
            img_path = os.path.join(IMAGES_DIR, img_rel)

            if not os.path.exists(img_path):
                # Image not downloaded — mark as failed
                row = [img_rel, l6, l20, l82] + [0.0] * num_cols + [1]
                writer.writerow(row)
                missing += 1
                failed += 1
                continue

            # Read and process image
            image = cv2.imread(img_path)
            if image is None:
                row = [img_rel, l6, l20, l82] + [0.0] * num_cols + [1]
                writer.writerow(row)
                failed += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb)

            if results.pose_landmarks is None:
                row = [img_rel, l6, l20, l82] + [0.0] * num_cols + [1]
                writer.writerow(row)
                failed += 1
                continue

            # Extract the 15 specific joints
            landmarks_data = []
            for idx in JOINT_INDICES:
                lm = results.pose_landmarks.landmark[idx]
                landmarks_data.extend([
                    round(lm.x, 6),
                    round(lm.y, 6),
                    round(lm.z, 6),
                    round(lm.visibility, 6),
                ])

            row = [img_rel, l6, l20, l82] + landmarks_data + [0]
            writer.writerow(row)
            success += 1

    return total, success, failed, missing


def main():
    print("=" * 60)
    print("MediaPipe Landmark Extraction for Yoga-82")
    print("=" * 60)

    # Check dependencies
    assert os.path.exists(TRAIN_TXT), f"Missing: {TRAIN_TXT}"
    assert os.path.exists(TEST_TXT), f"Missing: {TEST_TXT}"
    assert os.path.exists(IMAGES_DIR), f"Missing: {IMAGES_DIR}"

    # Initialize MediaPipe Pose — same config as main.py
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    # Parse splits
    train_entries = parse_split_file(TRAIN_TXT)
    test_entries = parse_split_file(TEST_TXT)
    print(f"Train entries: {len(train_entries)}")
    print(f"Test entries:  {len(test_entries)}")

    # Extract train
    print(f"\n{'-'*40} TRAIN {'-'*40}")
    train_csv = os.path.join(OUTPUT_DIR, "landmarks_train.csv")
    t_total, t_suc, t_fail, t_miss = extract_landmarks_for_split(
        train_entries, train_csv, pose
    )
    print(f"  Total: {t_total} | Success: {t_suc} | Failed: {t_fail} | Missing images: {t_miss}")
    print(f"  Detection rate: {t_suc/max(1,t_total-t_miss)*100:.1f}% (of available images)")

    # Extract test
    print(f"\n{'-'*40} TEST {'-'*40}")
    test_csv = os.path.join(OUTPUT_DIR, "landmarks_test.csv")
    te_total, te_suc, te_fail, te_miss = extract_landmarks_for_split(
        test_entries, test_csv, pose
    )
    print(f"  Total: {te_total} | Success: {te_suc} | Failed: {te_fail} | Missing images: {te_miss}")
    print(f"  Detection rate: {te_suc/max(1,te_total-te_miss)*100:.1f}% (of available images)")

    pose.close()

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 1 — DATASET AUDIT REPORT")
    print(f"{'='*60}")
    print(f"  Train split: {t_total} samples")
    print(f"  Test split:  {te_total} samples")
    print(f"  Total images available: {(t_total-t_miss)+(te_total-te_miss)}")
    print(f"  MediaPipe detection rate (train): {t_suc/max(1,t_total-t_miss)*100:.1f}%")
    print(f"  MediaPipe detection rate (test):  {te_suc/max(1,te_total-te_miss)*100:.1f}%")
    print(f"  Saved: {train_csv}")
    print(f"  Saved: {test_csv}")


if __name__ == "__main__":
    main()
