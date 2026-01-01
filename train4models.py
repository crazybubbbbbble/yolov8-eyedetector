import os
import sys
from ultralytics import YOLO

if __name__ == '__main__':
    # ================= ⚙️ 核心配置区域 (修改这里) =================
    
    # 【一键切换】在这里指定你要训练的版本：'v5', 'v8', 'v11'
    TARGET_VERSION = 'v5'  
    
    # 定义要训练的大小后缀 (n, s, m, l)
    model_types = ['n', 's', 'm', 'l']

    # 针对不同大小模型的显存保护配置 (Batch Size)
    # 如果 v11l 报显存溢出 (OOM)，请将 'l' 改为 2
    batch_config = {
        'n': 64,  
        's': 32,  
        'm': 12,   
        'l':6    
    }
    
    # 基础训练参数
    epoch_num = 50
    img_size = 640
    workers = 2
    
    # 数据集路径配置
    dataset_root = os.path.join('datasets', 'eye-mouth-dataset')
    yaml_path = os.path.join(dataset_root, 'data.yaml')
    test_source = os.path.join(dataset_root, 'test', 'images') 
    test_out_root = os.path.join('out', 'test')

    # ============================================================

    # --- 1. 自动解析版本与路径逻辑 ---
    # 根据 TARGET_VERSION 自动决定文件夹和文件前缀
    if TARGET_VERSION == 'v5':
        model_dir = os.path.join('model', 'yolov5')
        filename_prefix = 'yolov5'  # 文件名如 yolov5n.pt
    elif TARGET_VERSION == 'v8':
        model_dir = os.path.join('model', 'yolov8')
        filename_prefix = 'yolov8'  # 文件名如 yolov8n.pt
    elif TARGET_VERSION == 'v11':
        model_dir = os.path.join('model', 'yolov11')
        filename_prefix = 'yolo11'  # 注意：v11官方通常命名为 yolo11n.pt (没有v)
    else:
        print(f"❌ 错误：不支持的版本 '{TARGET_VERSION}'，请检查拼写 (v5/v8/v11)")
        sys.exit()

    # --- 2. 环境检查 ---
    if not os.path.exists(test_source):
        print(f"❌ 错误：找不到测试集路径：{test_source}")
        sys.exit()
    
    if not os.path.exists(model_dir):
        print(f"⚠️ 警告：模型目录 '{model_dir}' 不存在。")
        print("建议先运行 download_models.py 下载权重，否则将尝试在线下载。")

    print(f"\n🚀 启动训练任务 | 目标版本: {TARGET_VERSION.upper()} | 模式: 序列训练 {model_types}")
    print(f"📂 本地模型库: {model_dir}")

    # ================= 3. 循环训练流水线 =================
    for suffix in model_types:
        # 自动组装文件名：例如 yolo11n.pt
        model_filename = f'{filename_prefix}{suffix}.pt'
        model_path = os.path.join(model_dir, model_filename)
        
        # 自动组装任务名：例如 v11_n_eye_mouth (加上版本号防止混淆)
        task_name = f'{TARGET_VERSION}_{suffix}_eye_mouth'
        
        current_batch = batch_config[suffix]

        print(f"\n{'='*50}")
        print(f"▶️  正在处理: {model_filename}")
        print(f"📦  Batch Size: {current_batch}")
        print(f"📝  任务ID: {task_name}")
        print(f"{'='*50}")

        # last.pt 路径 (用于断点续训)
        last_ckpt_path = os.path.join('runs', 'detect', task_name, 'weights', 'last.pt')

        # --- 阶段一：训练 (Train) ---
        try:
            if os.path.exists(last_ckpt_path):
                print(f"✅ 检测到存档，恢复训练...")
                model = YOLO(last_ckpt_path)
                model.train(resume=True)
            else:
                # 检查本地是否有预训练权重
                if not os.path.exists(model_path):
                    print(f"⚠️ 本地未找到 {model_path}，将自动下载...")
                    load_target = model_filename # 传文件名，让 ultralytics 自己下
                else:
                    print(f"✅ 加载本地权重: {model_path}")
                    load_target = model_path

                model = YOLO(load_target)
                model.train(
                    data=yaml_path,
                    epochs=epoch_num,
                    imgsz=img_size,
                    batch=current_batch,
                    device=0,
                    workers=workers,
                    name=task_name,
                    exist_ok=True,
                    project='runs/detect' # 显式指定保存根目录
                )
        except Exception as e:
            print(f"❌ {task_name} 训练失败，错误信息：\n{e}")
            continue 

        # --- 阶段二：测试 (Predict) ---
        best_weight_path = os.path.join('runs', 'detect', task_name, 'weights', 'best.pt')
        
        if os.path.exists(best_weight_path):
            print(f"🔎 正在对 {task_name} 进行测试推理...")
            best_model = YOLO(best_weight_path)
            
            # 预测结果保存路径
            current_test_out = os.path.join(test_out_root, task_name)
            
            best_model.predict(
                source=test_source,
                save=True,
                project=test_out_root, # 保存到 out/test/
                name=task_name,        # 子文件夹名
                exist_ok=True,
                conf=0.25,
                device=0
            )
            print(f"✅ 测试完成，结果已保存至: {current_test_out}")
        else:
            print(f"⚠️ 未找到最佳权重 {best_weight_path}，跳过测试。")

    print("\n" + "="*50)
    print(f"🎉 {TARGET_VERSION} 版本全系列训练任务结束！")