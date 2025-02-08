import cv2
import numpy as np
from ultralytics import YOLO
import time

# COCO 預設 17 個關節的名稱，可視需求調整/增加
body_parts_list = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def resize_frame(frame, width=None, height=None, inter=cv2.INTER_AREA):
    """
    根據指定大小做等比縮放。
    若只指定 width，height=None，則自動按比例計算 height；
    若只指定 height，width=None，則自動按比例計算 width。
    """
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=inter)

def process_video(
    video_path,
    ball_model_path='model/tennisball_OD_v1.pt',
    pose_model_path='model/yolov8n-pose.pt',
    # 輸出主畫面大小 (推論 & 顯示都用這尺寸)
    OUTPUT_WIDTH=1280,
    OUTPUT_HEIGHT=720,
    # 每幾幀推論一次
    skip_frames=3,
    # YOLO 批次大小
    yolo_batch_size=8
):
    # -----------------------------
    # Step 1. 初始化模型 (GPU or CPU)
    # -----------------------------
    step1_start = time.time()
    device_str = 'cuda'  # 若無GPU，就改為 'cpu'
    ball_model = YOLO(ball_model_path).to(device_str)
    pose_model = YOLO(pose_model_path).to(device_str)
    step1_end = time.time()
    print(f"[Step 1] 模型初始化完成 (使用 {device_str})，耗時: {step1_end - step1_start:.2f} 秒")

    # -----------------------------
    # Step 2. 讀取影片 & 前置處理
    # -----------------------------
    step2_start = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法讀取影片: {video_path}")
        return

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames_for_output = []
    frames_for_infer = []
    infer_indices = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 只做一次縮放 (1280×720)
        resized_frame = resize_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        frames_for_output.append(resized_frame)

        if frame_idx % skip_frames == 0:
            frames_for_infer.append(resized_frame)
            infer_indices.append(frame_idx)

    cap.release()
    total_frames = len(frames_for_output)
    print(f"總共讀取 {total_frames} 幀，實際需要推論 {len(infer_indices)} 幀。")
    step2_end = time.time()
    print(f"[Step 2] 影片前置處理完成，耗時: {step2_end - step2_start:.2f} 秒")

    if total_frames == 0:
        return

    # -----------------------------
    # Step 3. YOLO 批次推論
    # -----------------------------
    step3_start = time.time()
    print("開始對選定幀做整批推論(球+姿態) ...")
    pose_results_batch = pose_model.predict(
        frames_for_infer,
        verbose=False,
        device=device_str,
        batch=yolo_batch_size
    )
    ball_results_batch = ball_model.predict(
        frames_for_infer,
        verbose=False,
        device=device_str,
        batch=yolo_batch_size
    )
    step3_end = time.time()
    print(f"推論完成, 耗時: {step3_end - step3_start:.2f} 秒")

    # -----------------------------
    # Step 4. 整理推論結果 (含插值)
    # -----------------------------
    step4_start = time.time()

    ball_positions = [None] * total_frames
    ball_confidences = [None] * total_frames
    keypoints_per_frame = [None] * total_frames

    # 將推論結果對應到正確幀索引
    for i, fidx in enumerate(infer_indices):
        pose_result = pose_results_batch[i]
        ball_result = ball_results_batch[i]

        # Pose
        if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
            kpts = pose_result.keypoints.xy[0]  # shape (17,2)
            kpts_xy = [(int(x), int(y)) for x, y in kpts]
        else:
            kpts_xy = None

        # Ball (只取分數最高的 box)
        boxes = ball_result.boxes
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            ball_pos = (cx, cy)
            ball_conf = float(box.conf[0])
        else:
            ball_pos = None
            ball_conf = None

        idx_in_list = fidx - 1  # fidx從1開始, list從0開始
        ball_positions[idx_in_list] = ball_pos
        ball_confidences[idx_in_list] = ball_conf
        keypoints_per_frame[idx_in_list] = kpts_xy

    # 插值：沒推論到的幀 -> 沿用前一次結果
    last_ball = None
    last_conf = None
    last_kpts = None
    for i in range(total_frames):
        if ball_positions[i] is None:
            ball_positions[i] = last_ball
            ball_confidences[i] = last_conf
        else:
            last_ball = ball_positions[i]
            last_conf = ball_confidences[i]

        if keypoints_per_frame[i] is None:
            keypoints_per_frame[i] = last_kpts
        else:
            last_kpts = keypoints_per_frame[i]

    step4_end = time.time()
    print(f"[Step 4] 推論結果整理完成，耗時: {step4_end - step4_start:.2f} 秒")

    # -----------------------------
    # Step 5. 繪製 & 輸出新影片
    # -----------------------------
    step5_start = time.time()
    output_path = video_path.replace('.mp4', '_styledPanel.mp4')

    info_panel_width = 400
    output_width = OUTPUT_WIDTH + info_panel_width  # 1680
    output_height = OUTPUT_HEIGHT                  # 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (output_width, output_height))

    # 軌跡 (示範只追蹤 right_wrist = 10)
    TRACKED_KEYPOINTS = [10]
    keypoint_trails = {kp: [] for kp in TRACKED_KEYPOINTS}
    ball_trail = []

    print("開始產生輸出影片...")

    for i in range(total_frames):
        frame = frames_for_output[i].copy()

        ball_pos = ball_positions[i]
        ball_conf = ball_confidences[i]
        kpts = keypoints_per_frame[i]

        # -- 更新球軌跡
        ball_trail.append(ball_pos)

        # -- 更新姿態軌跡 (只示範追蹤 keypoint=10)
        if kpts is not None:
            for kp_idx in TRACKED_KEYPOINTS:
                if kp_idx < len(kpts):
                    keypoint_trails[kp_idx].append(kpts[kp_idx])
                else:
                    keypoint_trails[kp_idx].append(None)
        else:
            for kp_idx in TRACKED_KEYPOINTS:
                keypoint_trails[kp_idx].append(None)

        # -- 繪製球軌跡
        valid_ball_positions = [p for p in ball_trail if p is not None]
        for b in range(1, len(valid_ball_positions)):
            p1 = valid_ball_positions[b - 1]
            p2 = valid_ball_positions[b]
            if p1 and p2:
                progress = b / len(valid_ball_positions)
                color = (0, int(255*(1 - progress)), int(255*progress))
                cv2.line(frame, p1, p2, color, 4)  # 球軌跡稍微細一點

        # -- 繪製姿態軌跡 (keypoints=10)
        for kp_idx, trail in keypoint_trails.items():
            valid_trail = [p for p in trail if p is not None]
            for t in range(1, len(valid_trail)):
                p1 = valid_trail[t-1]
                p2 = valid_trail[t]
                progress = t / len(valid_trail)
                color = (int(255*(1 - progress)), int(255*progress), 0)
                cv2.line(frame, p1, p2, color, 4)

        # -- (選擇性) 繪製全部 keypoints
        if kpts is not None:
            for idx, (xx, yy) in enumerate(kpts):
                color = (0, 0, 255) if idx in TRACKED_KEYPOINTS else (0, 255, 0)
                cv2.circle(frame, (xx, yy), 5, color, -1)
                cv2.putText(frame, str(idx), (xx+5, yy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # =======================
        #      製作 info_panel
        # =======================
        info_panel = np.ones((output_height, info_panel_width, 3), dtype=np.uint8) * 40

        # (1) 最上方綠底區塊
        header_height = 50
        cv2.rectangle(info_panel, (0, 0), (info_panel_width, header_height), (0, 150, 0), -1)
        cv2.putText(info_panel, "Tennis Ball Detection", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # 球偵測資訊起始 y 座標
        y_text = header_height + 30

        # (1-1) Ball Status
        if ball_pos is not None:
            cv2.putText(info_panel, "Ball Status: Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(info_panel, "Ball Status: Not Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_text += 30

        # (1-2) 顯示座標
        if ball_pos is not None:
            cx, cy = ball_pos
            cv2.putText(info_panel, f"Position: ({cx}, {cy})", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            y_text += 30
            # (1-3) 顯示 confidence
            if ball_conf is not None:
                cv2.putText(info_panel, f"Confidence: {ball_conf:.2f}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                y_text += 30

        # (2) 藍色區塊 (Pose Estimation Title)
        pose_header_height = 40
        pose_header_top = y_text
        pose_header_bottom = pose_header_top + pose_header_height

        cv2.rectangle(info_panel,
                      (0, pose_header_top),
                      (info_panel_width, pose_header_bottom),
                      (255, 100, 0),  # BGR=(255,100,0) 約藍色
                      -1)
        cv2.putText(info_panel, "Pose Estimation",
                    (10, pose_header_top + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # 結束後再把 y_text 往下移一些，避免文字與藍色區重疊
        y_text = pose_header_bottom + 30

        # (3) 列印所有 keypoints 座標
        if kpts is not None:
            for idx, part_name in enumerate(body_parts_list):
                if idx < len(kpts):
                    xx, yy = kpts[idx]
                    text_line = f"{part_name:<15}: ({xx}, {yy})"
                else:
                    text_line = f"{part_name:<15}: ( -, - )"
                cv2.putText(info_panel, text_line, (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                y_text += 22
                if y_text >= output_height - 10:
                    # 超過面板底部就中斷(避免文字被截斷)
                    break
        else:
            cv2.putText(info_panel, "No keypoints found",
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 合併左右畫面
        combined_frame = np.hstack((frame, info_panel))
        out.write(combined_frame)

    out.release()
    step5_end = time.time()
    print(f"輸出影片: {output_path}")
    print(f"[Step 5] 完成繪製與輸出，耗時: {step5_end - step5_start:.2f} 秒")

if __name__ == "__main__":
    total_start = time.time()

    process_video(
        video_path="pro_1_1_45.mp4",
        OUTPUT_WIDTH=1280,
        OUTPUT_HEIGHT=720,
        skip_frames=2,      # 越大越省算力
        yolo_batch_size=10
    )

    total_end = time.time()
    print(f"===== 程式總耗時: {total_end - total_start:.2f} 秒 =====")
