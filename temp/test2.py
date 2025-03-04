import json
import argparse
import os
from tqdm import tqdm

def mirror_keypoints(data, width=1920):
    """
    將關鍵點數據水平鏡像反轉（僅計算座標，不交換左右標籤）
    
    參數:
    data: 包含骨架關鍵點的字典列表
    width: 影片寬度，用於計算鏡像位置，默認為1920像素
    
    返回:
    鏡像後的關鍵點數據
    """
    mirrored_data = []
    
    for frame_data in data:
        mirrored_frame = frame_data.copy()
        
        # 處理每個節點
        for key in frame_data:
            # 跳過非關鍵點的數據
            if key in ["frame", "tennis_ball_hit", "tennis_ball_angle"]:
                continue
                
            # 反轉 x 座標 (如果不是 null)
            if key in frame_data and frame_data[key].get("x") is not None:
                mirrored_frame[key]["x"] = width - frame_data[key]["x"]
        
        mirrored_data.append(mirrored_frame)
    
    return mirrored_data

def process_keypoints_file(input_file, output_file=None, width=1920):
    """
    處理包含關鍵點的 JSON 文件並生成鏡像版本
    
    參數:
    input_file: 輸入 JSON 文件路徑
    output_file: 輸出 JSON 文件路徑，如果未提供則自動生成
    width: 影片寬度，默認為1920像素
    """
    # 如果未提供輸出文件路徑，自動生成一個
    if output_file is None:
        file_name = os.path.basename(input_file)
        directory = os.path.dirname(input_file)
        name_without_ext, ext = os.path.splitext(file_name)
        output_file = os.path.join(directory, f"{name_without_ext}_mirrored{ext}")
    
    print(f"讀取關鍵點數據：{input_file}")
    
    # 讀取 JSON 數據
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"找到 {len(data)} 幀的關鍵點數據")
    print("進行鏡像反轉...")
    
    # 鏡像關鍵點
    mirrored_data = mirror_keypoints(data, width)
    
    # 寫入結果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mirrored_data, f, indent=2)
    
    print(f"鏡像數據已保存到：{output_file}")

def process_directory(directory, output_directory=None, width=1920):
    """
    處理目錄中的所有 JSON 文件
    
    參數:
    directory: 包含 JSON 文件的目錄
    output_directory: 輸出目錄，如果未提供則使用 input_directory/mirrored
    width: 影片寬度，默認為1920像素
    """
    # 如果未提供輸出目錄，創建一個默認目錄
    if output_directory is None:
        output_directory = os.path.join(directory, "mirrored")
    
    # 確保輸出目錄存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 獲取目錄中的所有 JSON 文件
    json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    
    print(f"找到 {len(json_files)} 個 JSON 文件")
    
    # 處理每個文件
    for i, json_file in enumerate(json_files):
        input_path = os.path.join(directory, json_file)
        output_path = os.path.join(output_directory, json_file)
        
        print(f"\n處理文件 {i+1}/{len(json_files)}: {json_file}")
        process_keypoints_file(input_path, output_path, width)
    
    print("\n所有文件處理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='將骨架關鍵點數據水平鏡像反轉（保持標籤不變）')
    parser.add_argument('input', help='輸入 JSON 文件或目錄')
    parser.add_argument('-o', '--output', help='輸出 JSON 文件或目錄')
    parser.add_argument('-w', '--width', type=int, default=1920, help='影片寬度（默認：1920像素）')
    parser.add_argument('-d', '--directory', action='store_true', help='處理整個目錄')
    
    args = parser.parse_args()
    
    if args.directory:
        process_directory(args.input, args.output, args.width)
    else:
        process_keypoints_file(args.input, args.output, args.width)