import os
from ultralytics import YOLO
from common import *
from datetime import datetime
# --- 配置 ---
LOG_FILE = "train_log.txt"
MODEL_PATH = "../shared/yolo11n.pt"  # 或者使用你修正后的 "../shared/1.yaml"
DATA_CONFIG = '/nfs/spy/soybean_detect/code/config/data.yaml'

def main():
    # 1. 加载模型
    # 如果你已经修正了自定义的 1.yaml（包含正确层级结构），建议改用下面这两行：
    # model = YOLO("../shared/1.yaml")
    # model.load(MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # 2. 开始训练 (在你原始参数基础上进行了优化)
    results = model.train(
        data = DATA_CONFIG,
        task = 'detect', 
        mode = 'train', 
        imgsz = model_image_size, 
        max_det = yolo_max_det,
        device = [0, 1],
        
        # -------------------- 优化器和损失权重 --------------------
        optimizer = 'AdamW', 
        weight_decay = 0.0001, 
        cls = 4.0,           # 🌟 调整：从 2.0 提高到 4.0，大幅增强分类惩罚，解决误判
        box = 10.0,          # 保持高定位精度
        
        # -------------------- 训练控制 --------------------
        batch = 128, 
        epochs = 200, 
        patience = 100,
        freeze = 10,         # 冻结前10层，保护预训练特征
        
        # -------------------- 激进数据增强 (应对多背景/多角度) --------------------
        mosaic = 1.0,
        close_mosaic = 10,
        mixup = 0.1,
        scale = 0.5,
        degrees = 180, 
        fliplr = 0.25,
        flipud = 0.25,
        hsv_h = 0.015,
        hsv_s = 0.8,         # 颜色饱和度增强
        hsv_v = 0.5,         # 亮度增强
        shear = 30.0,        # 错切变换，模拟侧视角度
        perspective = 0.0001 # 透视变换，模拟3D空间感
    )

    # 3. 提取最终指标
    if results is not None:
        metrics = results.results_dict
        p = metrics.get('metrics/precision(B)', 0)
        r = metrics.get('metrics/recall(B)', 0)
        m50 = metrics.get('metrics/mAP50(B)', 0)
        m95 = metrics.get('metrics/mAP50-95(B)', 0)
    else:
        # 如果是多卡训练返回 None，我们尝试从保存的 csv 中读取（或者设为 0）
        print("⚠️ 警告: 训练结果对象为空（多卡 DDP 模式下常见），跳过直接提取。")
        p, r, m50, m95 = 0, 0, 0, 0

    # 4. 详细报告部分（model.val 通常在多卡下也会正常工作）
    try:
        print("正在生成各类别详细评估报告...")
        val_results = model.val()
        
        # ... 写入文件的逻辑保持不变 ...
        # (建议在 open(LOG_FILE, "a") 前加一个 if val_results is not None 判断)
    except Exception as e:
        print(f"❌ 自动保存日志失败: {e}")

    # 5. 格式化并写入日志文件
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"训练完成时间: {now}\n")
        f.write(f"总体性能: P={p:.4f}, R={r:.4f}, mAP50={m50:.4f}, mAP50-95={m95:.4f}\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'类别名称 (Class)':<20} | {'精度 (P)':<10} | {'召回 (R)':<10} | {'mAP50':<10}\n")
        
        # 遍历每个类别记录详细数据
        for i, name in enumerate(val_results.names.values()):
            class_p = val_results.box.p[i]
            class_r = val_results.box.r[i]
            class_m50 = val_results.box.map50[i]
            f.write(f"{name:<20} | {class_p:<10.3f} | {class_r:<10.3f} | {class_m50:<10.3f}\n")
            
    print(f"🌟 实验报告已追加保存至: {os.path.abspath(LOG_FILE)}")

if __name__ == "__main__":
    main()