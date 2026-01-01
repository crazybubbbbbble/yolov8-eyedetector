import cv2
import time
import os
import numpy as np
import torch
from ultralytics import YOLO

def run_pupil_tracking():
    # ================= 🔧 配置区域 =================
    # 任务名 (使用你效果最好的模型，比如 v8n)
    task_name = 'v8_n_eye_mouth' # 或者是 'yolov8n_eye'，请根据实际文件夹修改
    
    # 模型路径
    model_path = os.path.join('runs', 'detect', task_name, 'weights', 'best.pt')
    
    # 瞳孔定位参数
    # 阈值：越小越黑。瞳孔是眼睛里最黑的地方。
    # 如果环境很亮，把这个值调低 (e.g., 30)；如果环境暗，调高 (e.g., 60)
    PUPIL_THRESH = 40 
    
    conf_threshold = 0.45
    device = 0 if torch.cuda.is_available() else 'cpu'
    # ==============================================

    # 检查模型
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型：{model_path}\n请修改代码里的 task_name！")
        return
    
    print(f"📥 加载模型: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(0)
    # 提高一点分辨率，让瞳孔更清晰
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("✅ 系统启动！按 'q' 退出，按 'w'/'s' 调整阈值。")
    print(f"当前二值化阈值: {PUPIL_THRESH}")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO 推理
        results = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)
        
        # 2. 拿到预测框
        boxes = results[0].boxes
        
        # 在原图上画 YOLO 的框 (也可以自己画，这里用 ultralytics 自带的方便点)
        # 但为了画红点，我们尽量在 copy 上画，或者最后再画框
        annotated_frame = frame.copy()

        for box in boxes:
            # 获取坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 绘制 YOLO 框 (绿色)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Eye", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ================= 👁️ OpenCV 瞳孔定位核心算法 =================
            
            # A. 裁剪出眼睛区域 (ROI)
            # 注意边界检查，防止报错
            eye_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            if eye_roi.size > 0:
                # B. 转灰度
                gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                
                # C. 高斯模糊 (去噪)
                blurred_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                
                # D. 二值化 (Inverse: 黑的变白，白的变黑，方便找轮廓)
                _, binary_roi = cv2.threshold(blurred_roi, PUPIL_THRESH, 255, cv2.THRESH_BINARY_INV)
                
                # E. 查找轮廓
                contours, _ = cv2.findContours(binary_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # F. 找最大的轮廓 (假设最大的黑色块是瞳孔)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                
                if len(contours) > 0:
                    pupil_contour = contours[0]
                    
                    # 计算重心 (Moments)
                    M = cv2.moments(pupil_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # G. 坐标映射 (ROI 坐标 -> 全局坐标)
                        global_cx = x1 + cx
                        global_cy = y1 + cy
                        
                        # H. 画红点 (瞳孔中心) 🔴
                        cv2.circle(annotated_frame, (global_cx, global_cy), 4, (0, 0, 255), -1)
                        
                        # 画十字准星辅助
                        cv2.line(annotated_frame, (global_cx - 5, global_cy), (global_cx + 5, global_cy), (0, 0, 255), 1)
                        cv2.line(annotated_frame, (global_cx, global_cy - 5), (global_cx, global_cy + 5), (0, 0, 255), 1)

            # =============================================================

        # 计算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 显示信息
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Threshold: {PUPIL_THRESH} (Press W/S to adjust)", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('YOLOv8 + OpenCV Pupil Tracking', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'): # 增加阈值
            PUPIL_THRESH = min(255, PUPIL_THRESH + 5)
        elif key == ord('s'): # 减少阈值
            PUPIL_THRESH = max(1, PUPIL_THRESH - 5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_pupil_tracking()