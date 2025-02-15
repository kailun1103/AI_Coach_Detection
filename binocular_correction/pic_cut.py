import os
from PIL import Image

def crop_images_with_ratio(folder_path, ratio=0.7):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG')
    ratio = float(ratio)  # 確保 ratio 是浮點數
    
    for filename in os.listdir(folder_path):
        if filename.endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            
            width, height = img.size
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(file_path, quality=95)
            print(f'已處理: {filename} ({width}x{height} -> {new_width}x{new_height})')

folder_path = 'binocular_correction/outdoor_1126/backhand'
ratio = 0.7
crop_images_with_ratio(folder_path, ratio)