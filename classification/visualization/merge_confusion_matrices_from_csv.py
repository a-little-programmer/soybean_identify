# -*- coding: utf-8 -*-
import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# ==============================================================================
# 配置区域：想画哪个模型，就保留哪一行；不想画就注释掉。
# CSV 可以是 confusion_matrix_*_counts.csv，也可以是 predictions.csv。
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../evaluate"))

CSV_ITEMS = [
    # ("RegNet", os.path.join(EVALUATE_DIR, "evaluate_per_class", "confusion_matrix_regnet_counts.csv")),
    # ("ResNet50", os.path.join(EVALUATE_DIR, "evaluate_per_class", "confusion_matrix_resnet50_counts.csv")),
    # ("ViT", os.path.join(EVALUATE_DIR, "evaluate_per_class", "confusion_matrix_vit_counts.csv")),
    # ("Swin", os.path.join(EVALUATE_DIR, "evaluate_per_class", "confusion_matrix_swin_counts.csv")),
    ("Swin Diff", os.path.join(EVALUATE_DIR, "evaluate_swin_diff", "confusion_matrix_swin_diff_counts.csv")),
    ("Swin DCA", os.path.join(EVALUATE_DIR, "evaluate_swin_dca", "confusion_matrix_swin_dca_only_counts.csv")),
    ("Swin Diff DCA", os.path.join(EVALUATE_DIR, "evaluate_swin_diff_dca", "confusion_matrix_swin_diff_dca_counts.csv")),
]

PANEL_LABELS = {
    "RegNet": "A",
    "ResNet50": "B",
    "ViT": "C",
    "Swin": "D",
    "Swin Diff": "E",
    "Swin DCA": "F",
    "Swin Diff DCA": "G",
}

OUT_DIR = os.path.join(BASE_DIR, "merged_confusion_matrix")
OUT_NAME = "merged_confusion_matrices"
COLS = 2
DPI = 600
NUMBER_FONTSIZE = 13
CLASS_NAMES = [
    "b73", "hd16", "jd17", "jng20839", "lk314",
    "nn42", "nn43", "nn47", "nn49", "nn55",
    "nn60", "sd29", "sd30", "sn23", "sn29",
    "sz2", "xd18", "xzd1", "zd51", "zd53",
    "zd57", "zd59", "zd61", "zh301", "zld105",
]

TRUE_IDX_CANDIDATES = [
    "true_idx",
    "label",
    "labels",
    "y_true",
    "target",
    "true_label_idx",
]
PRED_IDX_CANDIDATES = [
    "pred_idx",
    "pred",
    "preds",
    "y_pred",
    "prediction",
    "pred_label_idx",
]
TRUE_NAME_CANDIDATES = [
    "true_class",
    "true_label",
    "target_name",
    "class",
]
PRED_NAME_CANDIDATES = [
    "pred_class",
    "pred_label",
    "prediction_name",
]


def find_column(columns, candidates):
    return next((col for col in candidates if col in columns), None)


def infer_model_name(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    for prefix in ["predictions_", "prediction_", "confusion_", "matrix_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    columns = list(df.columns)

    true_idx_col = find_column(columns, TRUE_IDX_CANDIDATES)
    pred_idx_col = find_column(columns, PRED_IDX_CANDIDATES)
    true_name_col = find_column(columns, TRUE_NAME_CANDIDATES)
    pred_name_col = find_column(columns, PRED_NAME_CANDIDATES)

    if true_idx_col is not None and pred_idx_col is not None:
        y_true = df[true_idx_col].astype(int).to_numpy()
        y_pred = df[pred_idx_col].astype(int).to_numpy()

        if true_name_col is not None:
            class_names = (
                df[[true_idx_col, true_name_col]]
                .drop_duplicates()
                .sort_values(true_idx_col)[true_name_col]
                .astype(str)
                .tolist()
            )
        else:
            num_classes = int(max(y_true.max(), y_pred.max())) + 1
            class_names = [str(i) for i in range(num_classes)]

        return y_true, y_pred, class_names

    if true_name_col is not None and pred_name_col is not None:
        true_names = df[true_name_col].astype(str)
        pred_names = df[pred_name_col].astype(str)
        class_names = sorted(set(true_names.tolist()) | set(pred_names.tolist()))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        y_true = true_names.map(class_to_idx).astype(int).to_numpy()
        y_pred = pred_names.map(class_to_idx).astype(int).to_numpy()
        return y_true, y_pred, class_names

    raise ValueError(
        f"无法识别 {csv_path} 的真实/预测列。\n"
        f"当前列名: {columns}\n"
        f"支持数字列: true_idx/pred_idx, label/pred, y_true/y_pred 等。\n"
        f"支持类别名列: true_class/pred_class, true_label/pred_label 等。"
    )


def load_confusion_matrix(csv_path):
    matrix = pd.read_csv(csv_path, header=None).to_numpy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"不是方阵混淆矩阵: {csv_path}, shape={matrix.shape}")
    matrix = matrix.astype(float)
    if np.all(np.isclose(matrix, np.round(matrix))):
        matrix = matrix.astype(int)

    if len(CLASS_NAMES) == matrix.shape[0]:
        class_names = CLASS_NAMES
    else:
        class_names = [str(i) for i in range(matrix.shape[0])]

    return matrix, class_names


def load_matrix_or_predictions(csv_path):
    try:
        return load_confusion_matrix(csv_path)
    except Exception:
        y_true, y_pred, class_names = load_predictions(csv_path)
        labels = list(range(len(class_names)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        return matrix, class_names


def plot_one_matrix(ax, cm, class_names, model_name, panel_label, number_fontsize):
    is_integer_matrix = np.issubdtype(cm.dtype, np.integer)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.text(
        -0.12,
        1.08,
        panel_label,
        transform=ax.transAxes,
        fontsize=46,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=34, fontweight="bold", pad=22)
    ax.set_xlabel("Predicted Label", fontsize=28)
    ax.set_ylabel("True Label", fontsize=28)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=20)
    ax.set_yticklabels(class_names, fontsize=20)

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            value_text = str(int(value)) if is_integer_matrix else f"{value:.2f}"
            ax.text(
                j,
                i,
                value_text,
                ha="center",
                va="center",
                fontsize=number_fontsize,
                fontweight="bold",
                color="white" if value > threshold else "black",
            )

    return im

def main():
    items = [(name, path) for name, path in CSV_ITEMS if not str(name).strip().startswith("#")]
    if not items:
        raise ValueError("CSV_ITEMS 为空。请在配置区域至少保留一个模型 CSV。")

    for model_name, csv_path in items:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{model_name} 的 CSV 不存在: {csv_path}")

    cols = max(1, COLS)
    rows = ceil(len(items) / cols)

    os.makedirs(OUT_DIR, exist_ok=True)

    fig_width = max(24 * cols, 18)
    fig_height = max(22 * rows, 18)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (ax, (model_name, csv_path)) in enumerate(zip(axes.ravel(), items)):
        cm, class_names = load_matrix_or_predictions(csv_path)
        ax.axis("on")
        im = plot_one_matrix(
            ax,
            cm,
            class_names,
            model_name,
            panel_label=PANEL_LABELS.get(model_name, chr(ord("A") + idx)),
            number_fontsize=NUMBER_FONTSIZE,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=22)

    fig.tight_layout()

    png_path = os.path.join(OUT_DIR, f"{OUT_NAME}.png")
    pdf_path = os.path.join(OUT_DIR, f"{OUT_NAME}.pdf")
    svg_path = os.path.join(OUT_DIR, f"{OUT_NAME}.svg")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    print(f"saved: {svg_path}")


if __name__ == "__main__":
    main()
