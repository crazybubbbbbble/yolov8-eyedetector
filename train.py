import os
from ultralytics import YOLO

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 1. 定义任务名字
    task_name = 'yolov8n_eye_mouth'
    
    # 2. 定义 last.pt 的路径
    last_ckpt_path = os.path.join('runs', 'detect', task_name, 'weights', 'last.pt')
    
    # 3. 基础参数配置
    yaml_path = os.path.join('datasets', 'eye-mouth-dataset', 'data.yaml')
    base_model = 'yolov8l.pt'
    epoch_num = 50
    img_size = 640
    batch_size = 8
    
    # 4. 【关键修改】测试集配置
    # 必须精确指向 test 下面的 images 文件夹，而不是 test 根目录
    test_source = os.path.join('datasets', 'eye-mouth-dataset', 'test', 'images') 
    
    test_out_root = os.path.join('out', 'test')
    # ===========================================

    # --- 第一阶段：训练 (Training) ---
    
    if os.path.exists(last_ckpt_path):
        print(f"✅ 检测到中断的存档文件：{last_ckpt_path}")
        print("🚀 正在恢复训练 (Resume Training)...")
        model = YOLO(last_ckpt_path)
        results = model.train(resume=True)
    else:
        print(f"❌ 未找到存档文件：{last_ckpt_path}")
        print(f"🆕 将使用 {base_model} 开始全新的训练...")
        model = YOLO(base_model)
        results = model.train(
            data=yaml_path,
            epochs=epoch_num,
            imgsz=img_size,
            batch=batch_size,
            device=0,
            workers=2,
            name=task_name
        )

    print("🎉 训练流程结束！")

    # --- 第二阶段：自动测试 (Auto Testing) ---
    
    print("\n" + "="*30)
    print("🔎 准备开始对测试集进行推理...")
    
    best_weight_path = os.path.join('runs', 'detect', task_name, 'weights', 'best.pt')
    
    # 这里加一个防御性检查，防止文件夹为空或者路径写错
    if not os.path.exists(test_source):
        print(f"❌ 严重错误：找不到测试集图片路径：{test_source}")
        print("请检查 datasets/eye-mouth-dataset/test 下面是否有 images 文件夹！")
    
    elif os.path.exists(best_weight_path):
        print(f"🏆 加载最佳模型权重：{best_weight_path}")
        best_model = YOLO(best_weight_path)
        
        print(f"▶️ 正在处理测试集：{test_source}")
        
        best_model.predict(
            source=test_source,     # 现在这里指向了 .../test/images
            save=True,
            project=test_out_root,
            name=task_name,
            exist_ok=True,
            conf=0.25,
            device=0
        )
        
        final_save_path = os.path.join(test_out_root, task_name)
        print(f"✅ 测试完成！请查看结果文件夹：{final_save_path}")
        
    else:
        print(f"⚠️ 警告：未找到最佳模型文件 {best_weight_path}")