👁️ Real-time Eye Detection & Pupil Tracking System
基于多版本 YOLO 的眼部检测与实时瞳孔定位系统
Core Insight: 经实证研究对比 12 个不同架构的模型，YOLOv8n 被证实为当前算力条件下兼顾高精度 (mAP@0.5 > 94.7%) 与实时性 (FPS > 150) 的最佳选择。

📖 项目介绍 (Introduction)
本项目旨在构建一个高鲁棒性、低延迟的实时眼部追踪系统，适用于人机交互、疲劳监测及视线追踪等场景。项目包含实证研究与系统实现两大部分：

大规模模型评估：系统性地训练并评估了 YOLOv5, YOLOv8, YOLOv11 三个版本的 Nano, Small, Medium, Large 共 12 个模型。

科研级基准测试：设计了严格的 Benchmark 脚本，从参数量、GFLOPs、推理延迟 (Latency)、抖动 (Jitter) 及 FPS 多个维度进行对比。

级联检测系统：基于优选的 YOLOv8n 模型，结合 OpenCV 图像处理算法（ROI 裁剪 + 自适应二值化 + 轮廓重心法），实现了像素级的实时瞳孔中心定位。

✨ 主要特性 (Key Features)
🏆 高性能：在 RTX 4060 Laptop 上实现 150+ FPS 的检测速度。

🎯 高精度：YOLOv8n 模型在测试集上 mAP@0.5 达到 94.79%。

🛡️ 强鲁棒性：能够有效应对佩戴眼镜、侧脸、闭眼、暗光等复杂场景（见下图）。

📊 可视化分析：提供 Nature 风格的科研绘图脚本，自动生成速度-精度权衡图 (Speed-Accuracy Trade-off)。

📂 数据集来源 (Dataset)
本项目使用的高质量眼部标注数据来源于 Roboflow Universe。

数据源名称: Eye-mouth Dataset

发布者: aya-almahasneh-hceez

下载链接: https://universe.roboflow.com/aya-almahasneh-hceez/eye-mouth-nlay2

数据规模: 包含训练集、验证集和测试集，已进行清洗和预处理。

🛠️ 部署与安装 (Installation)
建议使用 Conda 创建独立的虚拟环境。

1. 克隆项目
Bash

git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
2. 创建环境
Bash

conda create -n yolov8_eye python=3.9
conda activate yolov8_eye
3. 安装依赖
本项目依赖 ultralytics 和 opencv-python 等库。

Bash

pip install ultralytics opencv-python pandas matplotlib seaborn
(如果是 GPU 环境，请确保安装了对应 CUDA 版本的 PyTorch)

🚀 启动程序 (Usage)
1. 启动实时瞳孔追踪系统 (Demo)
这是本项目的核心功能展示，调用摄像头进行实时检测与瞳孔定位。

Bash

python test_webcam_pupil.py
操作指南:

按 q 退出程序。

按 w 增加二值化阈值（适应亮环境）。

按 s 降低二值化阈值（适应暗环境）。

2. 运行科研基准测试 (Benchmark)
对所有训练好的模型进行批量测速，生成 final_model_comparison.csv。

Bash

python benchmark_scientific.py
3. 生成论文图表 (Plotting)
读取基准测试数据，绘制 Nature 风格的对比图。

Bash

python plot_nature_paper.py
4. 训练模型 (Training)
如果你需要重新训练模型，可以使用以下命令：

Bash

# 训练 YOLOv8n
yolo detect train data=datasets/eye-mouth-dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 name=v8_n_eye_mouth
