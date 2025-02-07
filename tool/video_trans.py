import cv2
import os
from tqdm import tqdm

def convert_to_h264(input_path, output_path=None, fps=None):
    """
    將影片轉換為 H.264 編碼
    Args:
        input_path: 輸入影片路徑
        output_path: 輸出影片路徑（如果不指定，將在原檔名後加上 _h264）
        fps: 指定輸出影片的 FPS（如果不指定，將使用原影片的 FPS）
    """
    try:
        # 檢查輸入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到輸入文件：{input_path}")

        # 如果沒有指定輸出路徑，自動生成
        if output_path is None:
            filename, ext = os.path.splitext(input_path)
            output_path = f"{filename}_h264{ext}"

        # 打開輸入影片
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"無法打開影片：{input_path}")

        # 獲取影片參數
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果沒有指定 fps，使用原影片的 fps
        if fps is None:
            fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 創建 VideoWriter 物件
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # 使用 H.264 編碼
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise Exception("無法創建輸出影片")

        print(f"開始轉換影片到 H.264 編碼...")
        print(f"輸入影片：{input_path}")
        print(f"輸出影片：{output_path}")
        print(f"解析度：{width}x{height}")
        print(f"FPS：{fps}")

        # 使用 tqdm 顯示進度條
        with tqdm(total=total_frames, desc="轉換進度") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 寫入幀
                out.write(frame)
                pbar.update(1)

        # 釋放資源
        cap.release()
        out.release()

        print("\n轉換完成！")
        print(f"輸出文件已保存至：{output_path}")
        
        # 顯示輸出文件大小
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # 轉換為 MB
        print(f"輸出文件大小：{output_size:.2f} MB")

    except Exception as e:
        print(f"轉換過程中發生錯誤：{str(e)}")
        # 確保資源被釋放
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
    
    finally:
        cv2.destroyAllWindows()

def main():
    """
    主函數，用於處理用戶輸入和調用轉換函數
    """
    input_path = input("請輸入要轉換的影片路徑：").strip()
    
    # 詢問是否要自定義輸出路徑
    custom_output = input("是否要指定輸出路徑？(y/n)：").strip().lower()
    output_path = None
    if custom_output == 'y':
        output_path = input("請輸入輸出路徑：").strip()
    
    # 詢問是否要自定義 FPS
    custom_fps = input("是否要指定輸出 FPS？(y/n)：").strip().lower()
    fps = None
    if custom_fps == 'y':
        try:
            fps = int(input("請輸入 FPS："))
        except ValueError:
            print("無效的 FPS 值，將使用原影片的 FPS")
    
    # 執行轉換
    convert_to_h264(input_path, output_path, fps)

if __name__ == "__main__":
    main()