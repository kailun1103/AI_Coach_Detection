import numpy as np
import cv2
import json
import time
from ultralytics import YOLO

# -----------------------------
# 1) 定義 resize_frame
# -----------------------------
def resize_frame(frame, width=None, height=None, inter=cv2.INTER_AREA):
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


# -----------------------------
# 2) 你的 Keypoint Names
# -----------------------------
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# -----------------------------
# 3) 原先的 JSON 產生程式
# -----------------------------
def process_video(pose_model, ball_model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_json = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        body_results = pose_model(frame, verbose=False)
        ball_results = ball_model(frame, verbose=False)

        frame_data = {
            "frame": frame_number,
            "tennis_ball": {"x": None, "y": None}
        }
        
        # Initialize all keypoints as None
        for keypoint in keypoint_names:
            frame_data[keypoint] = {"x": None, "y": None}

        # Get all body keypoints
        for result in body_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                if len(keypoints) == len(keypoint_names):
                    for idx, keypoint in enumerate(keypoint_names):
                        x, y = keypoints[idx][:2]
                        coords = {
                            "x": int(x) if x != 0.0 else None,
                            "y": int(y) if y != 0.0 else None
                        }
                        frame_data[keypoint].update(coords)

        # Get tennis ball coordinates
        for result in ball_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if float(box.conf[0]) > 0.2:
                    frame_data["tennis_ball"].update({
                        "x": (x1 + x2) // 2,
                        "y": (y1 + y2) // 2
                    })
                    break

        frame_json.append(frame_data)
        frame_number += 1

    cap.release()

    # Handle last frame - copy previous frame's keypoints if missing
    if frame_json and len(frame_json) > 1:
        last_frame = frame_json[-1]
        prev_frame = frame_json[-2]
        for keypoint in keypoint_names:
            if last_frame[keypoint]["x"] is None:
                last_frame[keypoint] = prev_frame[keypoint]

    return frame_json


def analyze_trajectory(pose_model, ball_model, video_path):
    trajectory = process_video(pose_model, ball_model, video_path)
    output_path = video_path.replace('.mp4', '(2D_trajectory).json')
    
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    return output_path


# -----------------------------
# 4) 新增的 overlay 函式 (要用 resize_frame)
# -----------------------------
def overlay_with_skip(
    video_path,
    ball_model,
    pose_model,
    output_video_path=None,
    skip_frames=3,
    yolo_batch_size=8,
    OUTPUT_WIDTH=1280,
    OUTPUT_HEIGHT=720
):
    """
    同時實現姿態與網球軌跡疊加的功能
    """
    if output_video_path is None:
        output_video_path = video_path.replace('.mp4', '_overlay.mp4')

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

        # 使用 resize_frame
        resized_frame = resize_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        frames_for_output.append(resized_frame)

        # skip frames
        if frame_idx % skip_frames == 0:
            frames_for_infer.append(resized_frame)
            infer_indices.append(frame_idx)

    cap.release()
    total_frames = len(frames_for_output)
    if total_frames == 0:
        print("影片長度為 0，無法處理。")
        return

    # 批次推論
    pose_results_batch = pose_model.predict(frames_for_infer, verbose=False, device=pose_model.device, batch=yolo_batch_size)
    ball_results_batch = ball_model.predict(frames_for_infer, verbose=False, device=ball_model.device, batch=yolo_batch_size)

    # 預留容器
    ball_positions = [None] * total_frames
    ball_confidences = [None] * total_frames
    keypoints_per_frame = [None] * total_frames

    # 填入推論結果
    for i, fidx in enumerate(infer_indices):
        pose_result = pose_results_batch[i]
        ball_result = ball_results_batch[i]

        # Pose
        if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
            kpts = pose_result.keypoints.xy[0]  # shape (17,2)
            kpts_xy = [(int(x), int(y)) for x, y in kpts]
        else:
            kpts_xy = None

        # Ball
        boxes = ball_result.boxes
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ball_pos = (cx, cy)
            ball_conf = float(box.conf[0])
        else:
            ball_pos = None
            ball_conf = None

        idx_in_list = fidx - 1
        ball_positions[idx_in_list] = ball_pos
        ball_confidences[idx_in_list] = ball_conf
        keypoints_per_frame[idx_in_list] = kpts_xy

    # 用前一幀結果補齊
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

    # 合併畫面 + 軌跡
    info_panel_width = 400
    output_width = OUTPUT_WIDTH + info_panel_width
    output_height = OUTPUT_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (output_width, output_height))

    TRACKED_KEYPOINTS = [10]  # 追蹤第10關節
    keypoint_trails = {kp: [] for kp in TRACKED_KEYPOINTS}
    ball_trail = []

    for i in range(total_frames):
        frame = frames_for_output[i].copy()

        ball_pos = ball_positions[i]
        ball_conf = ball_confidences[i]
        kpts = keypoints_per_frame[i]

        # 累計球軌跡
        ball_trail.append(ball_pos)
        # 累計關節軌跡
        if kpts is not None:
            for kp_idx in TRACKED_KEYPOINTS:
                if kp_idx < len(kpts):
                    keypoint_trails[kp_idx].append(kpts[kp_idx])
                else:
                    keypoint_trails[kp_idx].append(None)
        else:
            for kp_idx in TRACKED_KEYPOINTS:
                keypoint_trails[kp_idx].append(None)

        # 畫球軌跡
        valid_ball_positions = [p for p in ball_trail if p is not None]
        for b in range(1, len(valid_ball_positions)):
            p1 = valid_ball_positions[b - 1]
            p2 = valid_ball_positions[b]
            if p1 and p2:
                progress = b / len(valid_ball_positions)
                color = (0, int(255*(1 - progress)), int(255*progress))
                cv2.line(frame, p1, p2, color, 4)

        # 畫關節軌跡
        for kp_idx, trail in keypoint_trails.items():
            valid_trail = [p for p in trail if p is not None]
            for t in range(1, len(valid_trail)):
                p1 = valid_trail[t-1]
                p2 = valid_trail[t]
                progress = t / len(valid_trail)
                color = (int(255*(1 - progress)), int(255*progress), 0)
                cv2.line(frame, p1, p2, color, 4)

        # 畫當前 keypoints
        if kpts is not None:
            for idx, (xx, yy) in enumerate(kpts):
                color = (0, 0, 255) if idx in TRACKED_KEYPOINTS else (0, 255, 0)
                cv2.circle(frame, (xx, yy), 5, color, -1)
                cv2.putText(frame, str(idx), (xx+5, yy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 右側 info panel
        info_panel = np.ones((output_height, info_panel_width, 3), dtype=np.uint8) * 40

        header_height = 50
        cv2.rectangle(info_panel, (0, 0), (info_panel_width, header_height), (0, 150, 0), -1)
        cv2.putText(info_panel, "Tennis Ball Detection", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        y_text = header_height + 30
        if ball_pos is not None:
            cv2.putText(info_panel, "Ball Status: Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(info_panel, "Ball Status: Not Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_text += 30

        if ball_pos is not None:
            cx, cy = ball_pos
            cv2.putText(info_panel, f"Position: ({cx}, {cy})", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            y_text += 30
            if ball_conf is not None:
                cv2.putText(info_panel, f"Confidence: {ball_conf:.2f}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                y_text += 30

        pose_header_height = 40
        pose_header_top = y_text
        pose_header_bottom = pose_header_top + pose_header_height
        cv2.rectangle(info_panel, (0, pose_header_top),
                      (info_panel_width, pose_header_bottom),
                      (255, 100, 0), -1)
        cv2.putText(info_panel, "Pose Estimation", (10, pose_header_top + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_text = pose_header_bottom + 30

        # 列印所有 keypoints
        if kpts is not None:
            for idx, part_name in enumerate(keypoint_names):
                if idx < len(kpts):
                    xx, yy = kpts[idx]
                    text_line = f"{part_name:<15}: ({xx}, {yy})"
                else:
                    text_line = f"{part_name:<15}: ( -, - )"
                cv2.putText(info_panel, text_line, (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                y_text += 22
                if y_text >= output_height - 10:
                    break
        else:
            cv2.putText(info_panel, "No keypoints found", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        combined_frame = np.hstack((frame, info_panel))
        out.write(combined_frame)

    out.release()
    print(f"Overlay video saved to: {output_video_path}")


def process_tennis_video(pose_model, ball_model, video_path):
    output_json_path = analyze_trajectory(pose_model, ball_model, video_path)
    overlay_with_skip(
        video_path=video_path,
        ball_model=ball_model,
        pose_model=pose_model,
        output_video_path=None,   # None 表示自動生成 "_overlay.mp4"
        skip_frames=3,
        yolo_batch_size=8,
        OUTPUT_WIDTH=1280,
        OUTPUT_HEIGHT=720
    )
    return output_json_path

if __name__ == "__main__":
    total_start_time = time.time()

    # Time model loading
    model_load_start = time.time()
    pose_model = YOLO('model/yolov8n-pose.pt')
    ball_model = YOLO('model/tennisball_OD_v1.pt')
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.8f}s")

    # 指定影片路徑
    video_path = 'pro_1_1_45.mp4'

    analysis_start = time.time()
    output_json_path = process_tennis_video(pose_model, ball_model, video_path)
    analysis_time = time.time() - analysis_start
    print(f"Trajectory analysis time (JSON): {analysis_time:.8f}s")
    print(f"JSON file saved at: {output_json_path}")

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")