"""
Generate visualizations for Yoga-82 Pose Classification Model Explanation.
Saves all plots to static/plots/ for embedding in the walkthrough.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

POSE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(POSE_DIR, "static", "plots")
STATIC_DIR = os.path.join(POSE_DIR, "static")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.size': 11,
})

LABEL_6_NAMES = {
    0: "Standing", 1: "Sitting", 2: "Balancing",
    3: "Inverted", 4: "Reclining", 5: "Wheel"
}

LABEL_82_SHORT = {
    0: "Akarna Dhan.", 1: "Bharadvaj.", 2: "Boat", 3: "Bound Angle",
    4: "Bow", 5: "Bridge", 6: "Camel", 7: "Cat Cow", 8: "Chair",
    9: "Child", 10: "Cobra", 11: "Cockerel", 12: "Corpse", 13: "Cow Face",
    14: "Crane/Crow", 15: "Dolphin Plank", 16: "Dolphin", 17: "Down Dog",
    18: "Eagle", 19: "Eight-Angle", 20: "Puppy", 21: "Rev. Side Angle",
    22: "Rev. Triangle", 23: "Feath. Peacock", 24: "Firefly", 25: "Fish",
    26: "Chaturanga", 27: "Frog", 28: "Garland", 29: "Gate",
    30: "Half Fishes", 31: "Half Moon", 32: "Handstand", 33: "Happy Baby",
    34: "Head-Knee", 35: "Heron", 36: "Side Stretch", 37: "Legs Up Wall",
    38: "Locust", 39: "Lord Dance", 40: "Low Lunge", 41: "Noose",
    42: "Peacock", 43: "Pigeon", 44: "Plank", 45: "Plow",
    46: "Sage Kound.", 47: "King Pigeon", 48: "Recl. Big Toe",
    49: "Rev. Head-Knee", 50: "Scale", 51: "Scorpion", 52: "Seated Fwd",
    53: "Shoulder Pr.", 54: "Side Leg Lift", 55: "Side Crow",
    56: "Side Plank", 57: "Seated", 58: "Split", 59: "Staff",
    60: "Stand. Fwd Bend", 61: "Stand. Split", 62: "Big Toe Hold",
    63: "Headstand", 64: "Shoulderstand", 65: "Supta B.K.",
    66: "Supta Vira.", 67: "Tortoise", 68: "Tree", 69: "Wheel",
    70: "Two-Foot Staff", 71: "Upward Plank", 72: "Virasana",
    73: "Warrior III", 74: "Warrior II", 75: "Warrior I",
    76: "Wide Seated", 77: "Wide Fwd Bend", 78: "Wild Thing",
    79: "Wind Reliev.", 80: "Yogic Sleep", 81: "Rev. Warrior"
}


def plot_pipeline_overview():
    """Pipeline overview as a flow chart."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')

    steps = [
        (1, "28,450 URLs\nImage Download", "#58a6ff", "19,337 images\n(68% success)"),
        (3.5, "MediaPipe Pose\nExtraction", "#3fb950", "15 joints x 4\n(x, y, z, vis)"),
        (6, "Feature\nEngineering", "#d29922", "59 floats\nnormalized"),
        (8.5, "LightGBM\n5-fold CV", "#f78166", "3 classifiers\ntrained"),
        (11, "Webcam\nInference", "#bc8cff", "~5ms latency\nreal-time"),
    ]

    for i, (x, label, color, detail) in enumerate(steps):
        box = mpatches.FancyBboxPatch((x - 0.9, 0.5), 1.8, 2.0,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color + '33', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 1.8, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color=color)
        ax.text(x, 0.9, detail, ha='center', va='center', fontsize=8,
                color='#8b949e')
        if i < len(steps) - 1:
            next_x = steps[i + 1][0]
            ax.annotate('', xy=(next_x - 1.0, 1.5), xytext=(x + 1.0, 1.5),
                        arrowprops=dict(arrowstyle='->', color='#8b949e', lw=2))

    fig.savefig(os.path.join(PLOTS_DIR, "pipeline_overview.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved pipeline_overview.png")


def plot_class_distributions():
    """Class distribution bar charts for all 3 levels."""
    train_df = pd.read_csv(os.path.join(POSE_DIR, "features_train.csv"))
    test_df = pd.read_csv(os.path.join(POSE_DIR, "features_test.csv"))

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    for ax, level, title, name_map in [
        (axes[0], "label_6", "Label 6 -- Body Position (6 classes)", LABEL_6_NAMES),
        (axes[1], "label_20", "Label 20 -- Pose Group (20 classes)", None),
        (axes[2], "label_82", "Label 82 -- Specific Pose (82 classes)", LABEL_82_SHORT),
    ]:
        train_counts = train_df[level].value_counts().sort_index()
        test_counts = test_df[level].value_counts().sort_index()

        all_classes = sorted(set(train_counts.index) | set(test_counts.index))
        train_vals = [train_counts.get(c, 0) for c in all_classes]
        test_vals = [test_counts.get(c, 0) for c in all_classes]

        x = np.arange(len(all_classes))
        w = 0.4

        ax.bar(x - w/2, train_vals, w, label='Train', color='#58a6ff', alpha=0.8)
        ax.bar(x + w/2, test_vals, w, label='Test', color='#3fb950', alpha=0.8)

        if name_map and len(all_classes) <= 20:
            labels = [name_map.get(c, str(c)) for c in all_classes]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        elif len(all_classes) > 20:
            ax.set_xticks(x[::5])
            ax.set_xticklabels([str(c) for c in all_classes[::5]], fontsize=8)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in all_classes], fontsize=8)

        ax.set_title(title, fontsize=12, fontweight='bold', color='#f0f6fc')
        ax.set_ylabel('Samples')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Annotate imbalance
        ratio = max(train_vals) / max(1, min(train_vals))
        ax.text(0.98, 0.95, f"Imbalance: {ratio:.1f}:1",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color='#f78166',
                bbox=dict(boxstyle='round', facecolor='#f7816622', edgecolor='#f78166'))

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "class_distributions.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved class_distributions.png")


def plot_feature_vector():
    """Feature vector composition diagram."""
    fig, ax = plt.subplots(figsize=(12, 5))

    components = [
        ("Torso-Normalised\nCoordinates", 39, "#58a6ff",
         "13 joints x 3 (x,y,z)\nRelative to shoulder midpoint\nScaled by torso length"),
        ("Joint Angles\n(Cosine Law)", 7, "#3fb950",
         "L/R Elbow, L/R Knee\nL/R Hip, Spine incl.\n7 angles in radians"),
        ("Pairwise\nDistances", 13, "#d29922",
         "Shoulder-shoulder, hip-hip\nLimb lengths, spreads\nNose-hip distance"),
    ]

    x_pos = 0
    for name, count, color, detail in components:
        rect = mpatches.FancyBboxPatch((x_pos, 0.5), count * 0.18, 3.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color + '33', edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        cx = x_pos + count * 0.09
        ax.text(cx, 3.2, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color)
        ax.text(cx, 2.2, f"{count} features", ha='center', va='center',
                fontsize=14, fontweight='bold', color='#f0f6fc')
        ax.text(cx, 1.2, detail, ha='center', va='center', fontsize=8, color='#8b949e')
        x_pos += count * 0.18 + 0.3

    ax.set_xlim(-0.3, x_pos + 0.3)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title("59-Float Feature Vector Composition", fontsize=14,
                 fontweight='bold', color='#f0f6fc', pad=15)

    # Total annotation
    ax.text(x_pos / 2, 4.2, "Total: 39 + 7 + 13 = 59 features per pose",
            ha='center', fontsize=11, color='#bc8cff', fontweight='bold')

    fig.savefig(os.path.join(PLOTS_DIR, "feature_vector.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_vector.png")


def plot_results_summary():
    """Bar chart of accuracy and F1 scores across levels."""
    results = json.load(open(os.path.join(POSE_DIR, "training_results.json")))

    levels = ["label_6", "label_20", "label_82"]
    labels = ["6 Classes\n(Body Position)", "20 Classes\n(Pose Group)", "82 Classes\n(Specific Pose)"]

    acc = [results[l]["accuracy"] * 100 for l in levels]
    macro_f1 = [results[l]["macro_f1"] * 100 for l in levels]
    weighted_f1 = [results[l]["weighted_f1"] * 100 for l in levels]
    latency = [results[l]["latency_ms"] for l in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy & F1
    x = np.arange(3)
    w = 0.25
    ax1.bar(x - w, acc, w, label='Accuracy', color='#58a6ff', alpha=0.9)
    ax1.bar(x, macro_f1, w, label='Macro F1', color='#3fb950', alpha=0.9)
    ax1.bar(x + w, weighted_f1, w, label='Weighted F1', color='#d29922', alpha=0.9)

    for i in range(3):
        ax1.text(i - w, acc[i] + 1, f"{acc[i]:.1f}%", ha='center', fontsize=8, color='#58a6ff')
        ax1.text(i, macro_f1[i] + 1, f"{macro_f1[i]:.1f}%", ha='center', fontsize=8, color='#3fb950')
        ax1.text(i + w, weighted_f1[i] + 1, f"{weighted_f1[i]:.1f}%", ha='center', fontsize=8, color='#d29922')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Score (%)')
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.set_title('Classification Performance', fontsize=12, fontweight='bold', color='#f0f6fc')
    ax1.grid(axis='y', alpha=0.3)

    # Latency
    colors = ['#58a6ff', '#3fb950', '#d29922']
    bars = ax2.bar(x, latency, 0.5, color=colors, alpha=0.9)
    for i, v in enumerate(latency):
        ax2.text(i, v + 0.2, f"{v:.1f}ms", ha='center', fontsize=10, color=colors[i])

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Inference Latency (per sample)', fontsize=12, fontweight='bold', color='#f0f6fc')
    ax2.axhline(y=33, color='#f78166', linestyle='--', alpha=0.5, label='30 FPS budget (33ms)')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "results_summary.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results_summary.png")


def plot_confusion_matrices():
    """Confusion matrices for label_6 and label_20."""
    test_df = pd.read_csv(os.path.join(POSE_DIR, "features_test.csv"))
    META_COLS = ["image_path", "label_6", "label_20", "label_82"]
    feature_cols = [c for c in test_df.columns if c not in META_COLS]
    X_test = test_df[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, level, title, name_map in [
        (axes[0], "6", "Label 6 -- Body Position", LABEL_6_NAMES),
        (axes[1], "20", "Label 20 -- Pose Group", None),
    ]:
        data = joblib.load(os.path.join(STATIC_DIR, f"classifier_{level}.joblib"))
        clf, le = data["model"], data["label_encoder"]

        y_true = le.transform(test_df[f"label_{level}"].values)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_true, y_pred)
        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)

        n = len(cm)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        if name_map:
            labels = [name_map.get(int(le.inverse_transform([i])[0]), str(i)) for i in range(n)]
        else:
            labels = [str(int(le.inverse_transform([i])[0])) for i in range(n)]

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                if val > 0.01:
                    color = 'white' if val > 0.5 else '#c9d1d9'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=6 if n > 10 else 8, color=color)

        ax.set_title(title, fontsize=11, fontweight='bold', color='#f0f6fc')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.colorbar(im, ax=axes, shrink=0.6, label='Proportion')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved confusion_matrices.png")


def plot_feature_importance():
    """Top-20 feature importance from LightGBM for label_82."""
    data = joblib.load(os.path.join(STATIC_DIR, "classifier_82.joblib"))
    clf = data["model"]

    pipeline = joblib.load(os.path.join(STATIC_DIR, "feature_pipeline.joblib"))
    feature_names = pipeline["feature_columns"]

    importance = clf.feature_importances_
    indices = np.argsort(importance)[-20:]  # Top 20

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#58a6ff' if 'angle' in feature_names[i].lower() or 'incl' in feature_names[i].lower()
              else '#3fb950' if any(d in feature_names[i].lower() for d in ['shldr', 'hip', 'elbow', 'knee', 'ankle', 'wrist', 'nose', 'spread'])
              else '#d29922' for i in indices]

    ax.barh(range(20), importance[indices], color=colors, alpha=0.9)
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance (split count)')
    ax.set_title('Top 20 Features for 82-Class Pose Classification',
                 fontsize=12, fontweight='bold', color='#f0f6fc')

    # Legend
    patches = [
        mpatches.Patch(color='#d29922', label='Coordinates'),
        mpatches.Patch(color='#58a6ff', label='Angles'),
        mpatches.Patch(color='#3fb950', label='Distances'),
    ]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_importance.png")


def plot_per_class_f1():
    """Per-class F1 for label_82 (sorted)."""
    test_df = pd.read_csv(os.path.join(POSE_DIR, "features_test.csv"))
    META_COLS = ["image_path", "label_6", "label_20", "label_82"]
    feature_cols = [c for c in test_df.columns if c not in META_COLS]
    X_test = test_df[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test)

    data = joblib.load(os.path.join(STATIC_DIR, "classifier_82.joblib"))
    clf, le = data["model"], data["label_encoder"]

    y_true = le.transform(test_df["label_82"].values)
    y_pred = clf.predict(X_test)

    classes = np.unique(y_true)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Sort by F1
    sorted_idx = np.argsort(f1s)
    f1_sorted = f1s[sorted_idx]
    class_labels = [LABEL_82_SHORT.get(int(le.inverse_transform([classes[i]])[0]),
                                         str(i)) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = ['#f78166' if f < 0.5 else '#d29922' if f < 0.8 else '#3fb950' for f in f1_sorted]
    ax.barh(range(len(f1_sorted)), f1_sorted, color=colors, alpha=0.9)
    ax.set_yticks(range(len(f1_sorted)))
    ax.set_yticklabels(class_labels, fontsize=6)
    ax.set_xlabel('F1 Score')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color='#f78166', linestyle='--', alpha=0.5)
    ax.axvline(x=0.8, color='#d29922', linestyle='--', alpha=0.5)
    ax.set_title('Per-Class F1 Score -- 82 Yoga Poses (sorted)',
                 fontsize=12, fontweight='bold', color='#f0f6fc')

    patches = [
        mpatches.Patch(color='#f78166', label='F1 < 0.5 (poor)'),
        mpatches.Patch(color='#d29922', label='0.5 <= F1 < 0.8'),
        mpatches.Patch(color='#3fb950', label='F1 >= 0.8 (good)'),
    ]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "per_class_f1.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved per_class_f1.png")


def main():
    print("Generating visualizations...")
    plot_pipeline_overview()
    plot_class_distributions()
    plot_feature_vector()
    plot_results_summary()
    plot_confusion_matrices()
    plot_feature_importance()
    plot_per_class_f1()
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
