import torch
import torch.nn as nn
from torchvision import models
import time
#生成推理时间、和参数量的评估脚本
# 定义类别数
NUM_CLASSES = 25
# 模拟单张图片的输入 (Batch Size = 1, 3通道, 224x224像素)
DUMMY_INPUT = torch.randn(1, 3, 224, 224)

# ================= 1. 模型加载函数 =================
def get_resnet():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_regnet():
    model = models.regnet_y_3_2gf()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_swin():
    model = models.swin_v2_b()
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    return model

def get_vit():
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model

# ================= 2. 评估核心逻辑 =================
def measure_model(model_name, model):
    # 强制使用 CPU 进行测试 (因为树莓派和手机绝大多数时候用 CPU 或 NPU 推理，而非独立显卡)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    inputs = DUMMY_INPUT.to(device)

    # 1. 计算参数量 (Parameters)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型物理大小 (通常参数是 float32，占 4 Bytes)
    model_size_mb = total_params * 4 / (1024 ** 2)

    # 2. 计算推理时间 (Inference Time)
    # GPU/CPU 测速前需要先"热身"几轮，让硬件进入高频状态
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

        # 正式测速 (跑 100 次取平均值)
        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            _ = model(inputs)
        end_time = time.time()

    avg_time_ms = ((end_time - start_time) / iterations) * 1000

    # 3. 打印结果
    print(f"| {model_name:<10} | {total_params / 1e6:>6.2f} M | {model_size_mb:>6.2f} MB | {avg_time_ms:>7.2f} ms |")

# ================= 3. 执行评估 =================
if __name__ == "__main__":
    print("正在评估各模型的参数量与 CPU 推理耗时 (模拟边缘设备测速环境)...")
    print("-" * 55)
    print(f"| {'模型名称':<10} | {'参数量(M)':<8} | {'物理大小':<9} | {'单张耗时':<10} |")
    print("-" * 55)
    
    measure_model("ResNet50", get_resnet())
    measure_model("RegNetY", get_regnet())
    measure_model("Swin-V2-B", get_swin())
    measure_model("ViT-B-16", get_vit())
    
    print("-" * 55)
    print("💡 注: '单张耗时' 为 CPU 上的纯推理时间。实际部署至树莓派时，时间会等比例变长。")