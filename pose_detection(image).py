import cv2
from ultralytics import YOLO

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 初始化 YOLOv8 模型
racket_model = YOLO("tennis_racket.pt")  # 網球拍子節點
pose_model = YOLO("yolov8n-pose.pt")  # 身體節點

# 讀取圖片
image_path = "123.jpg"  # 請替換為您的圖片路徑
frame = cv2.imread(image_path)

if frame is None:
    print(f"無法讀取圖片：{image_path}")
    exit()

# 調整圖片大小
frame = resize_image(frame, width=800)  # 將圖片寬度調整為800像素，高度等比例縮放

# 進行網球拍子節點偵測
racket_results = racket_model(frame)

# 進行身體節點偵測
pose_results = pose_model(frame)

# 繪製網球拍子節點結果
annotated_frame = racket_results[0].plot()

# 在同一幀上繪製身體節點結果
annotated_frame = pose_results[0].plot(img=annotated_frame)

# 顯示結果
cv2.imshow("Combined YOLO Detection", annotated_frame)
cv2.waitKey(0)  # 等待直到按下任意鍵

# 保存結果
cv2.imwrite("output_combined_resized.jpg", annotated_frame)

cv2.destroyAllWindows()