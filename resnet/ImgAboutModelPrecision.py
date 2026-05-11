import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 1. 四个模型名称 (X轴的4个大组)
models = ['RegNet', 'ResNet-50', 'ViT', 'Swin Transformer']

# 2. 三种数据/策略名称 (图例名称)
labels = ['Direct Resize', 'Reflect', 'Padding']

# 3. 核心数据：请将这里替换为您真实的准确率！
# 注意：每个列表里必须有 4 个数字，对应上面的 4 个模型
scores_A = [80.21, 84.78, 83.01, 85.98]  # 第一根柱子 (浅蓝)
scores_B = [81.32, 84.92, 84.17, 86.87]  # 第二根柱子 (鲑鱼红)
scores_C = [84.38, 85.43, 85.24, 87.34]  # 第三根柱子 (淡雅绿)

# 4. 学术风配色 (完美复刻原图，并补充了第三种相配的淡绿色)
color_A = '#87CEEB'  # 浅天蓝
color_B = '#FA8072'  # 鲑鱼红
color_C = '#90EE90'  # 淡雅绿 (LightGreen)
# ============================================

# 解决字体报错：使用系统默认自带的无衬线字体，绝对不报错且美观
plt.rcParams.update({'font.family': 'sans-serif'})

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

x = np.arange(len(models))  # X轴的基础位置
width = 0.22  # 【关键调整：为了放下3根柱子，宽度设为0.22，显得更加纤细修长】

# 绘制 3 组柱状图 (edgecolor='black' 完美还原图片里的黑色描边)
# 位置计算：中间柱子在 x，左边柱子在 x - width，右边柱子在 x + width
rects1 = ax.bar(x - width, scores_A, width, label=labels[0], color=color_A, edgecolor='black', zorder=3,linewidth=1,dpi=150)
rects2 = ax.bar(x,         scores_B, width, label=labels[1], color=color_B, edgecolor='black', zorder=3,linewidth=1,dpi=150)
rects3 = ax.bar(x + width, scores_C, width, label=labels[2], color=color_C, edgecolor='black', zorder=3,linewidth=1,dpi=150)

# ================= 添加柱子顶部的数值标签 =================
# 缩小字号为 8，并自带 % 符号，彻底解决 3 根柱子数字互相遮挡的问题
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 向上垂直偏移3个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# ================= 坐标轴与图表美化 =================
# Y轴强制从 80 开始，放大对比效果
ax.set_ylim(75, 90)

# 设置XY轴标题和刻度
ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)

# 添加图例 (放置在左上角)
ax.legend(loc='upper left', fontsize=10, edgecolor='lightgray', framealpha=1)

# 添加网格线 (复刻原图的十字交叉虚线网格)
ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

# 保留外部黑框，具有学术严谨的封闭感
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.0)

# 紧凑布局，防止四周边缘被裁剪
plt.tight_layout()

# 保存为高清图片
plt.savefig('comparison_3bars_perfect.png', dpi=300)

# 显示图表
plt.show()