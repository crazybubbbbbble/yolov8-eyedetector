import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import time
import cv2
import sys
import random

def benchmark_suite():
    # ================= 🔬 实验配置 (Experimental Setup) =================
    
    # 1. 自动搜索模型：去 runs/detect 下找所有的 best.pt
    models_dir = Path('runs/detect')
    
    # 2. 【关键修改】测试图片文件夹路径
    # 注意：为了防止 \t 转义错误，请全部使用斜杠 /，Python 在 Windows 下也能识别！
    test_images_dir = Path('datasets/eye-mouth-dataset/test/712_jpg.rf.fe9753ab502c3d13feb5ddda6e9437c1.jpg')
    
    # 或者如果你的数据集是另一个，请修改这里：
    # test_images_dir = Path('datasets/eye_dataset/test/images')

    # 3. 实验参数
    WARMUP_RUNS = 20        # 预热次数
    TEST_RUNS = 200         # 正式测试次数
    DEVICE = 0              # 使用 GPU
    
    # ===================================================================

    print("="*60)
    print("🚀 科研级 YOLO 基准测试脚本 V2 (Auto-Image)")
    print("="*60)

    # --- 0. 环境检查 ---
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 GPU！")
        sys.exit()
    
    # --- 1. 自动寻找一张测试图片 ---
    # 不需要你手动填文件名了，代码自动去文件夹里抓一张
    test_img_path = None
    if test_images_dir.exists():
        # 搜索 jpg, png, jpeg
        supported_ext = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        found_images = []
        for ext in supported_ext:
            found_images.extend(list(test_images_dir.glob(ext)))
        
        if found_images:
            # 随机选一张，或者选第一张
            test_img_path = found_images[0]
            print(f"✅ 成功自动获取测试图片：\n   {test_img_path.name}")
        else:
            print(f"⚠️ 警告：目录 {test_images_dir} 存在，但没找到图片。")
    else:
        print(f"⚠️ 警告：找不到目录 {test_images_dir}")

    # 准备图片数据 (预读取到内存)
    if test_img_path and test_img_path.exists():
        img_input = cv2.imread(str(test_img_path))
        if img_input is None:
             print("❌ 图片读取失败，使用纯黑图像代替。")
             img_input = np.zeros((640, 640, 3), dtype=np.uint8)
    else:
        print("⚠️ 未找到有效图片，将使用 640x640 纯黑图像进行基准测试（仅影响演示，不影响测速准确性）。")
        img_input = np.zeros((640, 640, 3), dtype=np.uint8)

    # --- 2. 自动寻找模型 ---
    model_paths = list(models_dir.rglob('weights/best.pt'))
    if not model_paths:
        print("❌ 未找到任何模型！请检查 runs/detect 目录。")
        sys.exit()

    print(f"🔍 共发现 {len(model_paths)} 个模型待测。")
    print("-" * 60)

    results = []

    # --- 3. 开始循环测试 ---
    for p in model_paths:
        task_name = p.parent.parent.name
        print(f"🚀 正在评测: {task_name:<25} ...", end="", flush=True)

        try:
            model = YOLO(p)
            
            # 获取参数量
            try:
                model_info = model.info(verbose=False)
                if isinstance(model_info, tuple):
                    params_m = model_info[1] / 1e6 
                else:
                    params_m = 0
            except:
                params_m = 0

            # 预热
            for _ in range(WARMUP_RUNS):
                model.predict(img_input, device=DEVICE, verbose=False, half=False)

            # 正式测速
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            latencies = []

            for _ in range(TEST_RUNS):
                starter.record()
                res = model.predict(img_input, device=DEVICE, verbose=False)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))

            # 统计
            latencies = np.array(latencies)
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            fps = 1000 / avg_latency

            print(f" ✅ FPS: {fps:.1f}")

            results.append({
                'Model': task_name,
                'Params(M)': round(params_m, 2),
                'Latency(ms)': round(avg_latency, 2),
                'Jitter(ms)': round(std_latency, 2), # 抖动/标准差
                'FPS': round(fps, 1)
            })

        except Exception as e:
            print(f" ❌ 失败: {e}")

    # --- 4. 输出结果 ---
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='FPS', ascending=False)
        print("\n" + "="*20 + " 🏆 最终实验结果 " + "="*20)
        print(df.to_string(index=False))
        df.to_csv('benchmark_scientific_results.csv', index=False)
        print(f"\n📄 结果已保存至: benchmark_scientific_results.csv")

if __name__ == '__main__':
    benchmark_suite()