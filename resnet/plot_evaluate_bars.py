# -*- coding: utf-8 -*-
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 配置区域
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认扫描这些 evaluate 脚本对应的输出目录。
# 如果新增 evaluate_xxx.py，并且输出目录同名，可直接把目录名追加到这里。
EVALUATE_DIR_NAMES = [
    "evaluate_per_class",
    "evaluate_swin_diff",
    "evaluate_swin_dca",
    "evaluate_swin_diff_dca",
]

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluate_bar_charts")

# 总体指标柱状图使用 summary.json 中的这些字段。
SUMMARY_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
]

# 各类别柱状图使用 per_class_metrics.csv 中的这些字段。
PER_CLASS_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_percent(value):
    value = float(value)
    if value <= 1.0:
        return value * 100.0
    return value


def safe_file_name(name):
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def load_summary_files():
    summaries = []

    for eval_dir_name in EVALUATE_DIR_NAMES:
        eval_dir = os.path.join(BASE_DIR, eval_dir_name)
        if not os.path.isdir(eval_dir):
            continue

        for file_name in sorted(os.listdir(eval_dir)):
            if not file_name.endswith("summary.json"):
                continue

            path = os.path.join(eval_dir, file_name)
            with open(path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            model_name = summary.get("model")
            if not model_name:
                model_name = file_name.replace("_summary.json", "").replace("summary.json", eval_dir_name)

            summaries.append({
                "eval_dir": eval_dir_name,
                "model": str(model_name),
                "path": path,
                "summary": summary,
            })

    return summaries


def load_per_class_files():
    metric_files = []

    for eval_dir_name in EVALUATE_DIR_NAMES:
        eval_dir = os.path.join(BASE_DIR, eval_dir_name)
        if not os.path.isdir(eval_dir):
            continue

        for file_name in sorted(os.listdir(eval_dir)):
            if not file_name.endswith("per_class_metrics.csv"):
                continue

            path = os.path.join(eval_dir, file_name)
            model_name = file_name.replace("_per_class_metrics.csv", "")
            if model_name == "per_class_metrics.csv":
                model_name = eval_dir_name.replace("evaluate_", "")

            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)

            if rows:
                metric_files.append({
                    "eval_dir": eval_dir_name,
                    "model": model_name,
                    "path": path,
                    "rows": rows,
                })

    return metric_files


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_summary_grouped_bars(summaries):
    available_metrics = []
    for key, label in SUMMARY_METRICS:
        if any(key in item["summary"] for item in summaries):
            available_metrics.append((key, label))

    if not available_metrics:
        print("未找到可绘制的 summary 指标，跳过总体指标柱状图。")
        return

    labels = [item["model"] for item in summaries]
    x = np.arange(len(labels))
    width = min(0.16, 0.8 / max(len(available_metrics), 1))
    offsets = (np.arange(len(available_metrics)) - (len(available_metrics) - 1) / 2) * width

    fig_width = max(10, len(labels) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    max_score = 0.0
    for i, (metric_key, metric_label) in enumerate(available_metrics):
        values = [
            to_percent(item["summary"].get(metric_key, 0.0))
            for item in summaries
        ]
        max_score = max(max_score, max(values) if values else 0.0)
        rects = ax.bar(
            x + offsets[i],
            values,
            width,
            label=metric_label,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        autolabel(ax, rects)

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, min(100, max(10, max_score + 8)))
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Evaluate Summary Metrics", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "summary_metrics_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"已保存总体指标柱状图: {out_path}")


def plot_per_class_bars(metric_file):
    class_names = [row["class"] for row in metric_file["rows"]]
    x = np.arange(len(class_names))

    fig_width = max(12, len(class_names) * 0.34)
    fig, axes = plt.subplots(
        len(PER_CLASS_METRICS),
        1,
        figsize=(fig_width, 10),
        dpi=300,
        sharex=True,
    )

    if len(PER_CLASS_METRICS) == 1:
        axes = [axes]

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, (metric_key, metric_label), color in zip(axes, PER_CLASS_METRICS, colors):
        values = [to_percent(row.get(metric_key, 0.0)) for row in metric_file["rows"]]
        ax.bar(
            x,
            values,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_ylabel(f"{metric_label} (%)", fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)

    axes[-1].set_xlabel("Class", fontweight="bold")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)

    title = f"{metric_file['model']} Per-Class Metrics"
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_name = f"{safe_file_name(metric_file['eval_dir'])}_{safe_file_name(metric_file['model'])}_per_class_bar.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"已保存各类别指标柱状图: {out_path}")


def main():
    ensure_output_dir()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })

    summaries = load_summary_files()
    per_class_files = load_per_class_files()

    if not summaries and not per_class_files:
        print("未找到 evaluate 输出文件。请先运行 evaluate 脚本生成 summary.json 或 per_class_metrics.csv。")
        print(f"默认扫描目录: {', '.join(EVALUATE_DIR_NAMES)}")
        return

    if summaries:
        plot_summary_grouped_bars(summaries)

    for metric_file in per_class_files:
        plot_per_class_bars(metric_file)

    print(f"柱状图输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
