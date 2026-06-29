# Yoga Pose Classification

A real-time yoga pose classification system built on the Yoga-82 dataset, using MediaPipe for landmark extraction and LightGBM for multi-level classification. Runs live in the browser via a Flask + Socket.IO backend.

## What it does

- Extracts 15 key body joints per frame using MediaPipe and computes a 59-float feature vector (torso-normalised coordinates, joint angles, limb distances)
- Trains three independent LightGBM classifiers across a 3-level label hierarchy: 6 broad body position categories, 20 pose groups, and 82 individual poses
- Streams webcam landmarks to the browser in real time via Socket.IO
- Renders a Three.js 3D skeleton avatar that mirrors your pose live
- Exposes a `/classify` REST endpoint returning predictions at all three label levels with confidence scores

## Results

| Model | Accuracy | Macro F1 | Latency |
|-------|----------|----------|---------|
| 6-class | 93.6% | 93.2% | 1.7ms |
| 20-class | 90.2% | 89.4% | 2.7ms |
| 82-class | 82.5% | 79.0% | 5.4ms |

All three run within a 30 FPS real-time budget.

## Setup

```bash
git clone https://github.com/sruthisureshkumar-arch/yoga-pose-classification.git
cd yoga-pose-classification
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:5000`.

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

## Extension

This repo is the classification foundation. It has been extended into a full pose correction coaching system with step-by-step verification and local LLM feedback — running entirely offline with no cloud API dependency:

**[ml-posecorrection-ai](https://github.com/sruthisureshkumar-arch/ml-posecorrection-ai)** — real-time pose correction coach using a local TensorFlow.js model and Ollama LLM
