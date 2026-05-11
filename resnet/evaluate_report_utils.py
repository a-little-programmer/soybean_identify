# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


REPORT_SEPARATOR = "=" * 100
TABLE_SEPARATOR = "-" * 100


def format_eval_report(
    model_name,
    data_dir,
    weight_path,
    class_index_path,
    num_classes,
    num_samples,
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    weighted_precision,
    weighted_recall,
    weighted_f1,
    avg_loss,
    rows,
):
    lines = [
        REPORT_SEPARATOR,
        "大豆品种分类评估报告",
        REPORT_SEPARATOR,
        f"模型: {model_name}",
        f"测试集路径: {data_dir}",
        f"权重路径: {weight_path}",
        f"类别映射路径: {class_index_path if class_index_path else '-'}",
        f"类别数: {num_classes}",
        f"样本数: {num_samples}",
        "",
        "[总体指标]",
        f"Accuracy (%): {accuracy * 100:.2f}",
        f"Macro Precision (%): {macro_precision * 100:.2f}",
        f"Macro Recall (%): {macro_recall * 100:.2f}",
        f"Macro-F1 (%): {macro_f1 * 100:.2f}",
        f"Weighted Precision (%): {weighted_precision * 100:.2f}",
        f"Weighted Recall (%): {weighted_recall * 100:.2f}",
        f"Weighted-F1 (%): {weighted_f1 * 100:.2f}",
        f"Avg Loss: {avg_loss:.4f}",
        "",
        "[各类别指标]",
        TABLE_SEPARATOR,
        f"{'Class':<18} | {'Support':>7} | {'Precision (%)':>13} | {'Recall (%)':>10} | {'F1 (%)':>8} | {'Avg Loss':>8}",
        TABLE_SEPARATOR,
    ]

    for row in rows:
        lines.append(
            f"{row['name']:<18} | "
            f"{int(row['support']):>7} | "
            f"{row['p'] * 100:>13.2f} | "
            f"{row['r'] * 100:>10.2f} | "
            f"{row['f1'] * 100:>8.2f} | "
            f"{row['loss']:>8.4f}"
        )

    lines.extend([
        TABLE_SEPARATOR,
        f"{'Weighted Average':<18} | "
        f"{num_samples:>7} | "
        f"{weighted_precision * 100:>13.2f} | "
        f"{weighted_recall * 100:>10.2f} | "
        f"{weighted_f1 * 100:>8.2f} | "
        f"{avg_loss:>8.4f}",
        f"{'Macro Average':<18} | "
        f"{'-':>7} | "
        f"{macro_precision * 100:>13.2f} | "
        f"{macro_recall * 100:>10.2f} | "
        f"{macro_f1 * 100:>8.2f} | "
        f"{'-':>8}",
        REPORT_SEPARATOR,
    ])
    return lines


def write_report(report_lines, output_dir, append=False):
    os.makedirs(output_dir, exist_ok=True)
    mode = "a" if append else "w"
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, mode, encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    return report_path


def save_confusion_matrix_counts(all_labels, all_preds, class_names, output_dir, file_name="confusion_matrix_counts.png"):
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    fig_size = max(10, len(class_names) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (Counts)",
    )
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=6,
            )

    fig.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_multiple_confusion_matrix_counts(items, output_dir, file_name="confusion_matrix_counts.png"):
    if not items:
        return None

    os.makedirs(output_dir, exist_ok=True)
    num_items = len(items)
    cols = 2 if num_items > 1 else 1
    rows = int(np.ceil(num_items / cols))
    class_count = max(len(item["class_names"]) for item in items)
    subplot_size = max(8, class_count * 0.38)

    fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, item in zip(axes.ravel(), items):
        class_names = item["class_names"]
        labels = list(range(len(class_names)))
        cm = confusion_matrix(item["labels"], item["preds"], labels=labels)

        ax.axis("on")
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True Label",
            xlabel="Predicted Label",
            title=f"{item['model']} Confusion Matrix (Counts)",
        )
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor", fontsize=5)
        plt.setp(ax.get_yticklabels(), fontsize=5)

        threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                ax.text(
                    j,
                    i,
                    format(value, "d"),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=3,
                )

    fig.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
