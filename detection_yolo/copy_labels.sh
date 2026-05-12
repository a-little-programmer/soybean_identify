#!/bin/bash

# --- 脚本功能：从多个硬编码的来源目录复制所有文件到单个目标目录 ---

# ===============================================
# === !!! 请在这里设置您的目录路径 !!! ===
# ===============================================

# 1. 目标目录 (所有文件将复制到这里)
# 请替换成您的实际路径
TARGET_DIR="/nfs/spy/soybean_detect/data/raw_data/labels" 

# 2. 来源目录列表 (文件将从这里复制出来)
# 格式为 ("路径1" "路径2" "路径3" ...)
# 请替换成您的实际路径
SOURCE_DIRS=(
    "/nfs/spy/soybean_detect/data/soybean_name/23_xzd1/labels"
    "/nfs/spy/soybean_detect/data/soybean_name/24_zh301/labels"
    "/nfs/spy/soybean_detect/data/soybean_name/25_nn55/labels"
    # 如果需要更多目录，请在这里添加
)

# ===============================================
# === !!! 以下是执行逻辑，无需修改 !!! ===
# ===============================================

echo "--- 批量文件复制工具 (静态路径) ---"

# 检查目标目录是否存在，如果不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    echo "目标目录 '$TARGET_DIR' 不存在，正在创建..."
    mkdir -p "$TARGET_DIR"
    if [ $? -ne 0 ]; then
        echo "错误：无法创建目标目录。请检查权限。"
        exit 1
    fi
fi

# 检查是否有有效的来源目录
VALID_SOURCE_DIRS=()
for DIR in "${SOURCE_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        VALID_SOURCE_DIRS+=("$DIR")
    else
        echo "警告：来源目录 '$DIR' 不存在或不是一个目录，将跳过。"
    fi
done

if [ ${#VALID_SOURCE_DIRS[@]} -eq 0 ]; then
    echo ""
    echo "未指定任何有效的来源目录。脚本退出。"
    exit 1
fi

# 执行复制操作
echo ""
echo "--- 准备执行复制 ---"
echo "来源目录列表: ${VALID_SOURCE_DIRS[*]}"
echo "目标目录: $TARGET_DIR"
echo "---"

# 使用 find 命令执行复制操作
# -type f 确保只复制文件
# -exec cp -v {} "$TARGET_DIR" \; 执行复制并显示文件名
find "${VALID_SOURCE_DIRS[@]}" -type f -exec cp -v {} "$TARGET_DIR" \;

# 检查 $? 变量，判断上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 文件复制操作完成！所有文件已复制到 '$TARGET_DIR'。"
else
    echo ""
    echo "❌ 复制操作可能存在错误。请检查上方的警告或错误信息。"
fi