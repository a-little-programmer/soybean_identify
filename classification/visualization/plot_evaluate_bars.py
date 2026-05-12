# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 配置区域
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../evaluate"))

# 默认扫描 classification/evaluate/ 下所有 evaluate*.py 对应的同名输出目录。
# 每个目录只读取 report.txt，不再依赖 summary.json 或 per_class_metrics.csv。
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluate_bar_charts")
OUTPUT_IMAGE_NAME = "summary_metrics_bar.png"

# 只改这里就能控制总览图画哪些指标。
# 可选项：Accuracy、Macro Precision、Macro Recall、Macro-F1、
#        Weighted Precision、Weighted Recall、Weighted-F1
METRICS_TO_PLOT = ["Accuracy", "Macro-F1", "Weighted-F1"]

# 论文图常用浅色系，顺序对应 METRICS_TO_PLOT。
color_A = "#87CEEB"  # 浅天蓝
color_B = "#FA8072"  # 鲑鱼红
color_C = "#90EE90"  # 淡雅绿
BAR_COLORS = [color_A, color_B, color_C]

# None 表示自动计算。默认开启放大差异，Y 轴不会从 0 开始。
Y_AXIS_MIN = None
Y_AXIS_MAX = 100
ZOOM_SMALL_GAINS = True
ZOOM_PADDING = 1.0
MIN_Y_RANGE = 3.0

REPORT_METRIC_KEYS = {
    "Accuracy": "Accuracy (%)",
    "Macro Precision": "Macro Precision (%)",
    "Macro Recall": "Macro Recall (%)",
    "Macro-F1": "Macro-F1 (%)",
    "Weighted Precision": "Weighted Precision (%)",
    "Weighted Recall": "Weighted Recall (%)",
    "Weighted-F1": "Weighted-F1 (%)",
}


def get_evaluate_output_dirs():
    dirs = []
    for file_name in sorted(os.listdir(EVALUATE_DIR)):
        if file_name == "evaluate_report_utils.py":
            continue
        if not (file_name.startswith("evaluate") and file_name.endswith(".py")):
            continue
        eval_dir = os.path.join(EVALUATE_DIR, os.path.splitext(file_name)[0])
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

        for line in section:
            stripped = line.strip()
            if stripped.startswith("模型:"):
                model_name = stripped.split(":", 1)[1].strip()
                continue

            key, value = parse_summary_line(stripped)
            if key in REPORT_METRIC_KEYS.values():
                summary[key] = value

        if model_name and summary:
            parsed.append({
                "eval_dir": report_dir_name,
                "model": model_name,
                "summary": summary,
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
    for metric_name in METRICS_TO_PLOT:
        if metric_name not in REPORT_METRIC_KEYS:
            print(f"跳过未知指标: {metric_name}")
            continue
        report_key = REPORT_METRIC_KEYS[metric_name]
        if any(report_key in record["summary"] for record in records):
            available_metrics.append((report_key, metric_name))

    if not available_metrics:
        print("未在 report.txt 中解析到总体指标，跳过总体指标柱状图。")
        return

    labels = [record["model"] for record in records]
    x = np.arange(len(labels))
    width = min(0.22, 0.8 / max(len(available_metrics), 1))
    offsets = (np.arange(len(available_metrics)) - (len(available_metrics) - 1) / 2) * width

    fig_width = max(9, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    colors = BAR_COLORS
    max_score = 0.0
    all_values = []
    for i, (report_key, label) in enumerate(available_metrics):
        values = [record["summary"].get(report_key, 0.0) for record in records]
        all_values.extend(values)
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
    if ZOOM_SMALL_GAINS and all_values:
        min_score = min(all_values)
        value_range = max(max_score - min_score, MIN_Y_RANGE)
        y_min_auto = max(0, min_score - ZOOM_PADDING)
        y_max_auto = min(100, y_min_auto + value_range + ZOOM_PADDING * 2)
        if y_max_auto <= max_score:
            y_max_auto = min(100, max_score + ZOOM_PADDING)
            y_min_auto = max(0, y_max_auto - value_range - ZOOM_PADDING * 2)
    else:
        y_min_auto = 0
        y_max_auto = min(100, max(10, max_score + 8))

    y_min = Y_AXIS_MIN if Y_AXIS_MIN is not None else y_min_auto
    y_max = Y_AXIS_MAX if Y_AXIS_MAX is not None else y_max_auto
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Evaluate Summary Metrics", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE_NAME)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"已保存总体指标柱状图: {out_path}")


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
        print("未找到可解析的 report.txt。请先运行 classification/evaluate/evaluate*.py 生成统一格式报告。")
        return

    plot_summary_grouped_bars(records)

    print(f"总览柱状图输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
