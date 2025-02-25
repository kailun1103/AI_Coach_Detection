import pygame
import time
import os
import sys

def play_sound():
    # 初始化pygame混音器
    pygame.mixer.init()
    
    # 設定預設音效文件
    # 你可以替換這個路徑為你自己的音效文件
    sound_file = "sound.mp3"  # 預設音效檔案名稱
    
    # 檢查是否有通過命令列提供音效檔案路徑
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        sound_file = sys.argv[1]
    
    # 檢查檔案是否存在
    if not os.path.exists(sound_file):
        print(f"找不到音效文件: {sound_file}")
        print("請確保音效文件存在，或者通過命令列參數提供正確的路徑")
        print("用法: python script.py [音效文件路徑]")
        time.sleep(3)  # 讓用戶有時間閱讀錯誤信息
        return
    
    try:
        # 載入並播放音效
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
        
        # 等待音效播放完畢
        duration = sound.get_length()
        time.sleep(duration)
        
        print(f"已播放音效: {sound_file}")
        
    except Exception as e:
        print(f"播放音效時發生錯誤: {e}")
        time.sleep(3)  # 讓用戶有時間閱讀錯誤信息

if __name__ == "__main__":
    play_sound()