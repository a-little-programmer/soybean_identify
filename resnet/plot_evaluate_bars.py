# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 配置区域
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认扫描 resnet/ 下所有 evaluate*.py 对应的同名输出目录。
# 每个目录只读取 report.txt，不再依赖 summary.json 或 per_class_metrics.csv。
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluate_bar_charts")

SUMMARY_METRICS = [
    ("Accuracy (%)", "Accuracy"),
    ("Macro Precision (%)", "Macro Precision"),
    ("Macro Recall (%)", "Macro Recall"),
    ("Macro-F1 (%)", "Macro-F1"),
    ("Weighted Precision (%)", "Weighted Precision"),
    ("Weighted Recall (%)", "Weighted Recall"),
    ("Weighted-F1 (%)", "Weighted-F1"),
]

PER_CLASS_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]


def safe_file_name(name):
    chars = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def get_evaluate_output_dirs():
    dirs = []
    for file_name in sorted(os.listdir(BASE_DIR)):
        if file_name == "evaluate_report_utils.py":
            continue
        if not (file_name.startswith("evaluate") and file_name.endswith(".py")):
            continue
        eval_dir = os.path.join(BASE_DIR, os.path.splitext(file_name)[0])
        report_path = os.path.join(eval_dir, "report.txt")
        if os.path.isfile(report_path):
            dirs.append((os.path.basename(eval_dir), report_path))
    return dirs


def split_report_sections(text):
    sections = []
    current = []

    for line in text.splitlines():
        if line.strip() == "大豆品种分类评估报告" and current:
            sections.append(current)
            current = []
        current.append(line)

    if current:
        sections.append(current)

    return sections


def parse_summary_line(line):
    if ":" not in line:
        return None, None
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    try:
        return key, float(value)
    except ValueError:
        return None, None


def parse_report(report_dir_name, report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    parsed = []
    for section in split_report_sections(text):
        model_name = None
        summary = {}
        rows = []
        in_class_table = False

        for line in section:
            stripped = line.strip()
            if stripped.startswith("模型:"):
                model_name = stripped.split(":", 1)[1].strip()
                continue

            key, value = parse_summary_line(stripped)
            if key in {metric_key for metric_key, _ in SUMMARY_METRICS}:
                summary[key] = value
                continue

            if stripped == "[各类别指标]":
                in_class_table = True
                continue

            if not in_class_table or "|" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 6:
                continue
            if parts[0] in ("Class", "Weighted Average", "Macro Average"):
                continue

            try:
                rows.append({
                    "class": parts[0],
                    "support": int(parts[1]),
                    "precision": float(parts[2]),
                    "recall": float(parts[3]),
                    "f1": float(parts[4]),
                    "avg_loss": float(parts[5]),
                })
            except ValueError:
                continue

        if model_name and summary:
            parsed.append({
                "eval_dir": report_dir_name,
                "model": model_name,
                "summary": summary,
                "rows": rows,
            })

    return parsed


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


def plot_summary_grouped_bars(records):
    available_metrics = []
    for report_key, label in SUMMARY_METRICS:
        if any(report_key in record["summary"] for record in records):
            available_metrics.append((report_key, label))

    if not available_metrics:
        print("未在 report.txt 中解析到总体指标，跳过总体指标柱状图。")
        return

    labels = [record["model"] for record in records]
    x = np.arange(len(labels))
    width = min(0.12, 0.8 / max(len(available_metrics), 1))
    offsets = (np.arange(len(available_metrics)) - (len(available_metrics) - 1) / 2) * width

    fig_width = max(12, len(labels) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#FF9DA6"]
    max_score = 0.0
    for i, (report_key, label) in enumerate(available_metrics):
        values = [record["summary"].get(report_key, 0.0) for record in records]
        max_score = max(max_score, max(values) if values else 0.0)
        rects = ax.bar(
            x + offsets[i],
            values,
            width,
            label=label,
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
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Evaluate Summary Metrics", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "summary_metrics_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"已保存总体指标柱状图: {out_path}")


def plot_per_class_bars(record):
    if not record["rows"]:
        return

    class_names = [row["class"] for row in record["rows"]]
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
        values = [row[metric_key] for row in record["rows"]]
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

    fig.suptitle(f"{record['model']} Per-Class Metrics", fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_name = f"{safe_file_name(record['eval_dir'])}_{safe_file_name(record['model'])}_per_class_bar.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"已保存各类别指标柱状图: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    })

    records = []
    for eval_dir_name, report_path in get_evaluate_output_dirs():
        records.extend(parse_report(eval_dir_name, report_path))

    if not records:
        print("未找到可解析的 report.txt。请先运行 resnet/evaluate*.py 生成统一格式报告。")
        return

    plot_summary_grouped_bars(records)
    for record in records:
        plot_per_class_bars(record)

    print(f"柱状图输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
