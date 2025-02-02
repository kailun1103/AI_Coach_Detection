import os
from PIL import Image

def crop_images_with_ratio(folder_path, ratio=0.7):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG')
    
    for filename in os.listdir(folder_path):
        if filename.endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            
            # 計算新尺寸
            width, height = img.size
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # 計算裁剪位置
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            # 裁剪並儲存
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(file_path, quality=95)
            print(f'已處理: {filename}')

# 執行
folder_path = 'binocular_correction/indoor_0109/forehand/45'
ratio = 0.6  # 設定想要的比例，例如0.8代表原尺寸的80%
crop_images_with_ratio(folder_path, ratio)