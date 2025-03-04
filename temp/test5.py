import os
import subprocess
import argparse
from pathlib import Path

def compress_video(input_file, output_file=None, crf=23, preset='medium', audio_bitrate='128k'):
    """
    使用FFmpeg壓縮影片，保留原始幀率。
    
    參數:
        input_file (str): 輸入影片的路徑
        output_file (str, optional): 輸出影片的路徑，默認為加上'_compressed'的原檔名
        crf (int, optional): 壓縮率（18-28之間，數值越小質量越高），默認23
        preset (str, optional): 編碼速度預設值，默認'medium'
        audio_bitrate (str, optional): 音頻比特率，默認'128k'
    
    返回:
        str: 輸出檔案的路徑
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到輸入檔案: {input_file}")
    
    if output_file is None:
        # 從輸入檔案路徑生成輸出檔案路徑
        input_path = Path(input_file)
        output_file = str(input_path.with_stem(f"{input_path.stem}_compressed"))
    
    # 檢查FFmpeg是否安裝
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        raise RuntimeError("FFmpeg未安裝或無法執行。請確保FFmpeg已正確安裝並添加到系統PATH中。")
    
    # 構建FFmpeg命令
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',        # 使用H.264編碼器
        '-crf', str(crf),         # 控制畫質
        '-preset', preset,        # 編碼速度預設
        '-c:a', 'aac',            # 音頻編碼器
        '-b:a', audio_bitrate,    # 音頻比特率
        '-vf', 'scale=-1:1080',   # 確保1080p高度，寬度自適應
        '-movflags', '+faststart', # 讓影片在網絡上可以更快開始播放
        output_file
    ]
    
    # 執行命令
    print(f"正在壓縮影片: {input_file}")
    print(f"使用的命令: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"壓縮完成！輸出檔案: {output_file}")
        
        # 獲取原始和壓縮後的檔案大小
        original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        compressed_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        reduction = (1 - compressed_size / original_size) * 100
        print(f"原始檔案大小: {original_size:.2f} MB")
        print(f"壓縮後檔案大小: {compressed_size:.2f} MB")
        print(f"大小減少: {reduction:.2f}%")
        
        return output_file
    
    except subprocess.CalledProcessError as e:
        print(f"壓縮影片時出錯: {e}")
        print(f"FFmpeg錯誤輸出: {e.stderr.decode('utf-8', errors='replace')}")
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='壓縮影片並保留原始幀率')
    parser.add_argument('input', help='輸入影片的路徑')
    parser.add_argument('-o', '--output', help='輸出影片的路徑')
    parser.add_argument('-c', '--crf', type=int, default=28, help='壓縮率（18-28，數值越小質量越高）')
    parser.add_argument('-p', '--preset', default='medium', 
                       choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 
                               'medium', 'slow', 'slower', 'veryslow'],
                       help='編碼速度預設值')
    parser.add_argument('-a', '--audio-bitrate', default='128k', help='音頻比特率')
    
    args = parser.parse_args()
    
    compress_video(args.input, args.output, args.crf, args.preset, args.audio_bitrate)