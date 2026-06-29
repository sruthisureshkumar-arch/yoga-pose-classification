# Yoga Pose Classification
*by Sruthi Suresh Kumar*

A real-time yoga pose classification system built on the Yoga-82 dataset, using MediaPipe for landmark extraction and LightGBM for multi-level classification. Runs live in the browser via a Flask + Socket.IO backend.

> **Phase 1 of a two-phase ML implementation.**
> This repo establishes the classification foundation. Phase 2 extends it into a full real-time pose correction coaching system with step-by-step verification and a local LLM — running entirely offline:
> → [**yoga-pose-correction-ai**](https://github.com/ShadyVoxx/yoga-pose-correction-ai)

---

## Results

| Model    | Accuracy | Macro F1 | Latency |
| -------- | -------- | -------- | ------- |
| 6-class  | **93.6%** | 93.2%   | 1.7ms   |
| 20-class | **90.2%** | 89.4%   | 2.7ms   |
| 82-class | **82.5%** | 79.0%   | 5.4ms   |

All three classifiers run within a 30 FPS real-time budget.

---

## What it does

- Extracts 15 key body joints per frame using MediaPipe and computes a **59-float feature vector** (torso-normalised coordinates, joint angles, limb distances)
- Trains three independent LightGBM classifiers across a 3-level label hierarchy: 6 broad body position categories, 20 pose groups, and 82 individual poses
- Streams webcam landmarks to the browser in real time via Socket.IO
- Renders a Three.js 3D skeleton avatar that mirrors your pose live
- Exposes a `/classify` REST endpoint returning predictions at all three label levels with confidence scores

---

## Setup

```bash
git clone https://github.com/sruthisureshkumar-arch/yoga-pose-classification.git
cd yoga-pose-classification
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:5000`.

---

## Project structure

```
main.py                  # Flask + Socket.IO server, webcam stream, /classify endpoint
build_features.py        # Builds 59-float feature vectors from landmark CSVs
extract_landmarks.py     # MediaPipe landmark extraction from images
train_classifiers.py     # LightGBM training across all 3 label levels
generate_plots.py        # Confusion matrices and performance plots
classify_webcam.py       # Standalone webcam classifier (no web UI)
features_train/test.csv  # Extracted feature datasets
training_results.json    # Saved model metrics
static/                  # Trained .joblib models + 3D avatar
templates/               # Frontend HTML
```

---

## Extension — Phase 2

This repo is the classification foundation. It has been extended into a full pose correction coaching system with step-by-step verification, posture descriptor matching, and local LLM feedback — running entirely offline with no cloud API dependency.

**Phase 2 highlights:**
- Step-level classification across 74 classes (4–7 steps per pose, 13 Common Yoga Protocol poses)
- 89.1% validation accuracy on 164,670 training samples (4-layer MLP: 512-256-128-64)
- IMU-ready 117-float feature vector (99 normalised landmarks + 12 joint angles + 6 sensor slots)
- Body-part specific corrective feedback from posture descriptor analysis
- Local LLM coaching via Ollama — triggered after 3s of sustained posture violation

→ [**yoga-pose-correction-ai**](https://github.com/ShadyVoxx/yoga-pose-correction-ai)
