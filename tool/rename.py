import os
import re
import shutil

def rename_45_files(i, base_dir="./"):
    """重命名 45度 文件夾中的檔案"""
    # Pattern to match files like junior/i/45度/i-X.mp4
    pattern = re.compile(fr'(junior/{i}/45度/)(\d+)-(\d+)(\.mp4)')
    
    # Find all matching files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            old_path = os.path.join(root, file)
            
            # Normalize path separators
            normalized_path = old_path.replace('\\', '/')
            
            # Check if the file matches our pattern
            match = pattern.search(normalized_path)
            if match:
                # Extract components
                prefix = match.group(1)  # junior/i/45度/
                num1 = match.group(2)    # first number
                num2 = match.group(3)    # second number
                extension = match.group(4)  # .mp4
                
                # Create new path
                new_name = f"{i}_{num2}/{i}_{num2}__45{extension}"
                new_path = f"junior/{i}/{new_name}"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                # Rename the file
                print(f"Renaming: {normalized_path} → {new_path}")
                shutil.move(old_path, new_path)
    
    print(f"45度 file renaming completed for junior/{i}!")

def rename_side_files(i, base_dir="./"):
    """重命名 側面 文件夾中的檔案"""
    # Pattern to match files like junior/i/側面/i-X.mp4
    pattern = re.compile(fr'(junior/{i}/側面/)(\d+)-(\d+)(\.mp4)')
    
    # Find all matching files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            old_path = os.path.join(root, file)
            
            # Normalize path separators
            normalized_path = old_path.replace('\\', '/')
            
            # Check if the file matches our pattern
            match = pattern.search(normalized_path)
            if match:
                # Extract components
                prefix = match.group(1)  # junior/i/側面/
                num1 = match.group(2)    # first number
                num2 = match.group(3)    # second number
                extension = match.group(4)  # .mp4
                
                # Create new path
                new_name = f"{i}_{num2}/{i}_{num2}__side{extension}"
                new_path = f"junior/{i}/{new_name}"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                # Rename the file
                print(f"Renaming: {normalized_path} → {new_path}")
                shutil.move(old_path, new_path)
    
    print(f"側面 file renaming completed for junior/{i}!")


if __name__ == "__main__":
    # 设置要跳过的数字
    skip_numbers = [14]
    
    # 处理 12 到 23 之间的数字，但跳过 skip_numbers 中的数字
    for i in range(12, 17):
        if i in skip_numbers:
            print(f"跳过 junior/{i}")
            continue
            
        rename_45_files(i)
        rename_side_files(i)
        print(f"All file renaming completed for junior/{i}!")