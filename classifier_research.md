# Classifier Research — Yoga-82 Pose Classification

## Context
- **Input**: Flat tabular feature vector, 59 floats per sample
- **Tasks**: 3 classification levels — 6, 20, and 82 classes
- **Class imbalance**: Expected 5:1+ at fine-grained level
- **Inference constraint**: Must run at 30 FPS in real-time (< 1ms per sample)
- **Data type**: Single static frames (no temporal sequences)

## Candidate Evaluation

| Rank | Model | Fit to Tabular | Expected Accuracy | Interpretability | Latency | Handles Imbalance | Verdict |
|------|-------|---------------|-------------------|-----------------|---------|-------------------|---------|
| **1** | **LightGBM / XGBoost** | ★★★★★ | ★★★★★ | ★★★★ (feature importance) | < 0.1ms | Native (`is_unbalanced`) | **Top choice** — gradient boosted trees are consistently best on structured tabular data; sub-ms inference; handles class imbalance natively |
| **2** | **Random Forest** | ★★★★★ | ★★★★ | ★★★★★ | < 0.2ms | `class_weight='balanced'` | **Strong fallback** — more robust to hyperparameters; gives feature importance; slightly lower accuracy than GBTs |
| **3** | **SVM (RBF)** | ★★★★ | ★★★★ | ★★ | 0.1–0.5ms | `class_weight='balanced'` | Good for small-medium datasets; RBF kernel works well on normalised joint angles; doesn't scale as well to 82 classes |
| **4** | **MLP (2–3 layers)** | ★★★★ | ★★★★ | ★★ | < 0.1ms | Class weights or focal loss | Worth trying for label_82 where we need maximum discriminative capacity; needs more data |
| **5** | **KNN** | ★★★ | ★★★ | ★★★★ | 1–5ms | Not native | Low training cost but inference scales with dataset size; may exceed latency budget at 20k+ samples |
| — | 1D-CNN / LSTM | ★ | N/A | ★★ | 0.5–2ms | — | **Skip** — no temporal sequences; single-frame data makes this architecturally inappropriate |

## Per-Task Recommendation

| Hierarchy | Primary Model | Rationale |
|-----------|--------------|-----------|
| **label_6** (6 classes) | LightGBM | Easy task; any model works; GBT will be near-perfect |
| **label_20** (20 classes) | LightGBM | Moderate task; GBT's depth + boosting handles the grouping well |
| **label_82** (82 classes) | LightGBM → consider MLP ensemble | Hardest; GBT is the safest bet; MLP worth comparing if GBT < 60% macro F1 |

## Hierarchical vs Flat Classification

**Decision: Train flat classifiers (independent per level).**

Rationale:
- Cascade architectures (predict L6 → condition L20 → condition L82) introduce error propagation: a wrong L6 prediction makes L82 impossible to recover
- The label hierarchy in Yoga-82 is strict (L82 uniquely determines L20 and L6), so flat classifiers already implicitly learn hierarchical structure
- Flat training is simpler, faster, and easier to debug
- If needed later, post-hoc hierarchical consistency can be enforced at inference time

## Decision
**→ Train LightGBM classifiers for all three hierarchy levels.**
