import json
import os
import glob
import random
import shutil
import subprocess
import sys

from common import *
from sklearn.model_selection import train_test_split
# 引入切图模块
try:
    from slice_images_and_labels import run_slicing
except ImportError:
    print("❌ 错误：未找到 slice_images_and_labels.py")
    sys.exit(1)

# ===============================================================
# 路径配置 (安全修正版)
# ===============================================================

# 1. 输入源：原始大图和对应的标签
# 假设结构是 data/raw_data/images 和 data/raw_data/labels
# 我们通过 base_src_dir 推导上一级目录，确保不依赖 base_src_dir 具体指向哪
# 假设 base_src_dir 是 .../data/raw_data，那么 os.path.dirname(base_src_dir) 就是 .../data
data_root = os.path.dirname(base_src_dir)  #

raw_data_root = os.path.join(data_root, 'raw_data')
raw_images_dir = os.path.join(raw_data_root, 'images') # 或者 'backgrounds'，请根据实际情况修改
raw_labels_dir = os.path.join(raw_data_root, 'labels')

# 2. 中间临时目录 (sliced_data)
# 🔴 [关键修改] 强制指定为同级的 sliced_data 目录，不直接使用 base_src_dir
sliced_data_root = base_slice_dir

# 🛡️ [安全锁] 防止误删 raw_data
if os.path.abspath(sliced_data_root) == os.path.abspath(raw_data_root):
    print("❌ 危险配置：切片目录与原始数据目录重合！")
    print(f"sliced_data_root: {sliced_data_root}")
    print(f"raw_data_root:    {raw_data_root}")
    print("请修改代码中的 sliced_data_root 路径，防止原始数据被删除。")
    sys.exit(1)

# 3. 最终训练数据目录 (target_data)
image_dir = os.path.join(base_dst_dir, 'images')
image_train_dir = os.path.join(image_dir, 'train')
image_val_dir = os.path.join(image_dir, 'val')

label_dir = os.path.join(base_dst_dir, 'labels')
label_train_dir = os.path.join(label_dir, 'train')
label_val_dir = os.path.join(label_dir, 'val')

generated_label_src_dir = os.path.join(sliced_data_root, generated_label_src_dir_name)
label_src_dir = os.path.join(sliced_data_root, label_src_dir_name) # 切片后的 json 目录

python_executable = sys.executable
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def clean_training_dirs():
    dirs_to_clear = [image_dir, image_train_dir, image_val_dir, label_dir, label_train_dir, label_val_dir, generated_label_src_dir]
    for d in dirs_to_clear:
        if os.path.exists(d):
            for f in glob.glob(os.path.join(d, '*')):
                if os.path.isfile(f):
                    os.remove(f)

def fix_json_slashes_inplace(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace('\\\\', '/')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        print(f"❌ JSON 修复出错: {e}")

def copy_files(label_files, image_dst_dir, label_dst_dir):
    # 拿到 sliced_data_root/images 目录
    slice_images_dir = os.path.join(sliced_data_root, 'images')
    
    print(f"\n--- 🚀 开始复制到目标目录 ({os.path.basename(image_dst_dir)}/{os.path.basename(label_dst_dir)}) ---")
    print(f"目标 TXT 来源目录 (generated_label_src_dir): {generated_label_src_dir}")
    
    successful_copies = 0

    for label_file in label_files:
        
        # 1. 加载 JSON 数据
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"❌ 错误：无法加载 JSON 文件 {label_file}: {e}")
            continue

        # ------------------- 图片文件路径构造与检查 -------------------
        image_path_from_json = label_data.get('imagePath', '')
        if not image_path_from_json:
            print(f"⚠️ 跳过：JSON 文件 {os.path.basename(label_file)} 缺少 'imagePath' 字段。")
            continue
            
        image_basename = os.path.basename(image_path_from_json)
        # 切片图片的完整源路径 (sliced_data_root/images/XXX.jpg)
        image_file_src = os.path.join(slice_images_dir, image_basename)
        
        if not os.path.exists(image_file_src):
            print(f"❌ 跳过 {image_basename}：找不到切片图片。期望路径：{image_file_src}")
            continue
        
        # 目标图片路径检查
        target_image_path = os.path.join(image_dst_dir, image_basename)
        if os.path.exists(target_image_path):
            # print(f"⚠️ 跳过 {image_basename}：目标图片已存在。")
            continue 
        
        # ------------------- TXT 文件路径构造与检查 (最关键部分) -------------------
        
        # 提取文件名，并将其扩展名改为 .txt
        json_basename = os.path.basename(label_file)
        txt_basename = json_basename.replace('.json', '.txt')
        
        # 健壮的路径拼接: 目标目录 + 文件名
        generated_label_file_src = os.path.join(generated_label_src_dir, txt_basename)

        if os.path.exists(generated_label_file_src):
            
            # 4. 执行复制操作
            # 假设 convert_to_required_size 负责将 image_file_src 复制到 image_dst_dir
            convert_to_required_size(image_file_src, image_dst_dir)
            
            # 复制 TXT 标签
            shutil.copy(generated_label_file_src, label_dst_dir)
            successful_copies += 1
            # print(f"✅ 成功复制 {image_basename} 及其标签到 {os.path.basename(label_dst_dir)}")
        else:
            print(f"❌ 跳过 {image_basename}：**找不到 YOLO TXT 文件**。")
            print(f"   期望 TXT 路径: {generated_label_file_src}")
            
    print(f"--- 💡 {os.path.basename(label_dst_dir)} 复制完成，成功 {successful_copies} 组文件。 ---")
    return successful_copies

if __name__ == '__main__':
    os.chdir(base_code_dir)

    # ------------------------------------------------------------------
    # 第一步：物理切片 (Raw -> Sliced)
    # ------------------------------------------------------------------
    print(f"--- 步骤 1: 自动切图 (Input: {raw_images_dir}) ---")
    
    if os.path.exists(sliced_data_root):
        try:
            shutil.rmtree(sliced_data_root)
        except Exception as e:
            print(f"⚠️ 警告：无法删除旧切片目录: {e}")
    
    if os.path.exists(raw_images_dir):
        run_slicing(
            input_img_dir=raw_images_dir,
            input_label_dir=raw_labels_dir,
            output_base_dir=sliced_data_root,
            slice_size=2048,
            overlap=0.2
        )
    else:
        print(f"❌ 错误：原始图片目录不存在 {raw_images_dir}")
        print("请检查 common.py 中的配置或 raw_data 文件夹结构。")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 第二步：准备训练数据 (Sliced -> Target)
    # ------------------------------------------------------------------
    print("--- 步骤 2: 清理目标目录 ---")
    clean_training_dirs()
    
    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)
    os.makedirs(generated_label_src_dir, exist_ok=True)

    sliced_label_files = glob.glob(os.path.join(label_src_dir, '*.json'))
    if not sliced_label_files:
        print("❌ 错误：切片后没有找到 JSON 文件。请先运行 auto_label.py 生成原始标签。")
        sys.exit(1)

    for lf in sliced_label_files:
        fix_json_slashes_inplace(lf)

    # print("--- 步骤 3: 转换标签格式 (JSON -> YOLO TXT) ---")
    # args = [python_executable, convert_script, label_src_dir, generated_label_src_dir, '--class_list'] + label_names
    # try:
    #     # 🌟 关键修改：添加 check=True
    #     subprocess.run(args, check=True) 
    #     print("✅ 转换脚本运行成功。")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ 严重错误: 标签转换脚本 {convert_script} 运行失败，未能生成 TXT 文件。")
    #     print("请检查转换脚本的内部代码，特别是类别列表是否匹配！")
    #     # 打印子进程的标准输出和错误输出来帮助诊断
    #     if e.stdout: print(f"--- Subprocess Output ---\n{e.stdout.decode()}")
    #     if e.stderr: print(f"--- Subprocess Error ---\n{e.stderr.decode()}")
    #     sys.exit(1)
    print("--- 步骤 3: 划分并复制到 Target Data ---") 
    sliced_label_files = glob.glob(os.path.join(label_src_dir, '*.json'))
    if not sliced_label_files:
        print("❌ 错误：切片后没有找到 JSON 文件。请先运行 auto_label.py 生成原始标签。")
        sys.exit(1)

    for lf in sliced_label_files:
        fix_json_slashes_inplace(lf) # 确保 JSON 格式正确

    print("--- 步骤 4: 转换标签格式 (JSON -> YOLO TXT) ---")
    args = [python_executable, convert_script, label_src_dir, generated_label_src_dir, '--class_list'] + label_names
    # 🌟 使用 check=True 确保转换成功
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 严重错误: 标签转换脚本 {convert_script} 运行失败。")
        sys.exit(1)


    # 1. 解析所有 JSON 文件并为每个文件创建类别标签 (Stratification Key)
    file_list = [] # 存储所有 JSON 文件的路径
    file_labels = [] # 存储每个文件的主导/稀有类别标签，用于分层

    print(f"🔄 正在解析 {len(sliced_label_files)} 个切片 JSON 文件以获取分层标签...")

    for label_file in sliced_label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            present_classes = set()
            for shape in label_data['shapes']:
                present_classes.add(shape['label'])

            if present_classes:
                # 🌟 分层键 (Stratification Key) 策略：
                # 对于包含多个类别的文件，选取一个能代表该文件的稀有类别作为标签。
                # 简单起见，我们使用一个组合字符串作为标签（例如：'clsA_clsB'），
                # 但更稳健的做法是选取文件中最稀有的类别标签。
                
                # 简单策略：将所有类别排序并组合成一个字符串作为分层标签
                stratify_key = "_".join(sorted(list(present_classes)))

                file_list.append(label_file)
                file_labels.append(stratify_key)
                
        except Exception as e:
            print(f"❌ 解析 JSON 文件 {os.path.basename(label_file)} 失败: {e}")
            continue

    # 2. 执行分层抽样 (使用 Sklearn)
    # 我们使用 80% 训练集, 20% 验证集
    if len(file_list) > 1:
        # random_state 用于确保每次划分结果一致
        train_files, val_files = train_test_split(
            file_list,
            test_size=0.2, 
            random_state=RANDOM_SEED,
            stratify=file_labels # 🌟 核心：使用类别组合作为分层键
        )
    else:
        # 如果文件太少，无法分层，则全部作为训练集
        train_files = file_list
        val_files = []

    print(f"✅ 分层抽样完成。总文件数: {len(file_list)}")
    print(f"   - 分配到训练集: {len(train_files)} 个")
    print(f"   - 分配到验证集: {len(val_files)} 个")

    # 3. 执行复制操作
    copy_files(train_files, image_train_dir, label_train_dir)
    copy_files(val_files, image_val_dir, label_val_dir)

    print(f"✅ 训练集: {len(glob.glob(os.path.join(image_train_dir, '*')))} 张")
    print(f"✅ 验证集: {len(glob.glob(os.path.join(image_val_dir, '*')))} 张")

    # ------------------------------------------------------------------
    # 第三步：清理临时切片
    # ------------------------------------------------------------------
    print(f"--- 步骤 5: 删除临时切片数据 ---")
    if os.path.exists(sliced_data_root):
        try:
            os.chdir(base_code_dir)
            shutil.rmtree(sliced_data_root)
            print(f"✅ 临时文件清理完毕: {sliced_data_root}")
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")
