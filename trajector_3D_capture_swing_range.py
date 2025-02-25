import json
import time
import os

def extract_frames(input_file, start_frame, end_frame):
    """
    從 JSON 檔案中擷取指定幀範圍的資料並儲存到新檔案
    """
    # 計時開始
    start_time = time.time()
    
    # 讀取 JSON 資料
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_frames = len(data)
    print(f"成功讀取 JSON 檔案，共 {total_frames} 個幀")
    
    # 驗證幀範圍
    if start_frame < 0:
        start_frame = 0
        print(f"警告：起始幀小於 0，已調整為 0")
    
    if end_frame >= total_frames:
        end_frame = total_frames - 1
        print(f"警告：結束幀超出範圍，已調整為 {end_frame}")
    
    # 擷取指定範圍的幀
    extracted_data = data[start_frame:end_frame+1]

    output_file = input_file.replace('.json','_only_swing.json')
    
    # 儲存到新檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    # 計時結束
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"已從幀 {start_frame} 到 {end_frame} 擷取資料")
    print(f"總共擷取 {len(extracted_data)} 個幀")
    print(f"儲存到 '{os.path.basename(output_file)}'")
    print(f"處理時間：{execution_time:.4f} 秒")

    return output_file

if __name__ == "__main__":
    # 使用簡單的輸入方式
    input_file = 'testing_0224__1(3D_trajectory_smoothed).json'
    start_frame = 139
    end_frame = 192
    
    # 生成默認的輸出檔案名稱
    

    output_file = extract_frames(input_file, start_frame, end_frame)
