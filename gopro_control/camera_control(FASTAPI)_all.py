import time
import json
import asyncio
from pathlib import Path
from enum import Enum
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="GoPro Controller API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有方法
    allow_headers=["*"]   # 允許所有 headers
)

# ----------------------------------------
# Global Variables
# ----------------------------------------
current_user_folder: Optional[Path] = None
current_user_name: Optional[str] = None

yolo_pose_model: Optional[YOLO] = None
yolo_tennis_ball_model: Optional[YOLO] = None

# ----------------------------------------
# Enums, Models & Utilities
# ----------------------------------------
class DominantHand(str, Enum):
    right = "right"
    left = "left"

class UserData(BaseModel):
    name: str
    height: float
    dominant_hand: str

def find_next_trajectory_number(base_folder: Path) -> int:
    """
    找到下一個可用的軌跡編號 (trajectory number)
    """
    try:
        if not base_folder.exists():
            return 1
        
        max_number = 0
        for folder in base_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("trajectory__"):
                try:
                    number = int(folder.name.split("__")[-1])
                    max_number = max(max_number, number)
                except (ValueError, IndexError):
                    continue
        return max_number + 1
    except Exception as e:
        print(f"Error in find_next_trajectory_number: {str(e)}")
        return 1

async def post_gopro(session: aiohttp.ClientSession, url: str, data=None) -> dict:
    """
    封裝對 GoPro API 發送 POST 請求的方法
    """
    try:
        if isinstance(data, dict):
            form = aiohttp.FormData()
            for key, value in data.items():
                form.add_field(key, str(value))
            async with session.post(url, data=form) as response:
                return await response.json()
        elif data is not None:
            async with session.post(url, data=data) as response:
                return await response.json()
        else:
            async with session.post(url) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------
# Application Events
# ----------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    伺服器啟動時載入 YOLO 模型
    """
    global yolo_pose_model, yolo_tennis_ball_model
    print("正在載入 YOLO 模型...")
    try:
        yolo_pose_model = YOLO('model/yolov8n-pose.pt')
        yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
        print("YOLO 模型載入完成!")
    except Exception as e:
        print(f"模型載入失敗: {str(e)}")
        raise e

# ----------------------------------------
# Endpoints
# ----------------------------------------
@app.get("/model_status")
async def check_model_status():
    """
    檢查 YOLO 模型是否已成功載入
    """
    return {
        "pose_model_loaded": yolo_pose_model is not None,
        "tennis_ball_model_loaded": yolo_tennis_ball_model is not None
    }

@app.get("/input_data")
async def input_user_data(
    name: str,
    height: float,
    dominant_hand: int  # 0為左手，1為右手
):
    """
    接收使用者資料，建立使用者特定資料夾與 JSON 紀錄
    GET 請求版本：
    - name: 使用者名稱
    - height: 身高(cm)
    - dominant_hand: 0=左手, 1=右手
    """
    start_time = time.time()
    try:
        # 檢查 dominant_hand 輸入
        if dominant_hand not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail="dominant_hand must be 0 (left) or 1 (right)"
            )
            
        # 轉換 dominant_hand 數值為字串
        hand = "left" if dominant_hand == 0 else "right"
        
        global current_user_folder, current_user_name
        current_user_name = name
        
        # 建立使用者特定資料夾
        current_user_folder = Path(f"trajectory/{name}__trajectory")
        current_user_folder.mkdir(parents=True, exist_ok=True)
        
        # 準備要寫入的資料
        data_to_save = {
            "name": name,
            "height": height,
            "hand": hand,
            "timestamp": time.strftime("%Y_%m_%d_%H_%M_%S"),
            "file_path": str(current_user_folder)
        }
        
        # 將使用者資料存入 JSON 檔
        file_path = f"play_records/{name}_{str(height).replace('.0','')}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        
        execution_time = time.time() - start_time
        return {
            "message": "200 success",
            "data": data_to_save,
            "file_path": file_path,
            "execution_time": f"{execution_time:.2f} seconds"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{execution_time:.2f} seconds"
            }
        )

@app.get("/take_photo")
async def take_photo():
    """
    同時對兩台 GoPro 執行拍照
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/take_photo"),
            post_gopro(session, "http://localhost:9436/take_photo")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/start_recording")
async def start_recording():
    """
    同時對兩台 GoPro 開始錄影
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/start_recording"),
            post_gopro(session, "http://localhost:9436/start_recording")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/stop_recording")
async def stop_recording():
    """
    同時對兩台 GoPro 停止錄影
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/stop_recording"),
            post_gopro(session, "http://localhost:9436/stop_recording")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/stop_recording_and_download")
async def stop_recording_and_download():
    """
    同時對兩台 GoPro 停止錄影並下載影片，並依據使用者資料夾與軌跡建立結構
    """
    global current_user_folder, current_user_name
    
    if not current_user_folder or not current_user_name:
        raise HTTPException(
            status_code=400,
            detail="User information not found. Please call /input_data first."
        )
    
    base_folder = Path(current_user_folder)
    if not base_folder.exists():
        base_folder.mkdir(parents=True, exist_ok=True)
    
    # 取得下一個可用的軌跡編號並建立新資料夾
    next_number = find_next_trajectory_number(base_folder)
    trajectory_folder = base_folder / f"trajectory__{next_number}"
    trajectory_folder.mkdir(parents=True, exist_ok=True)
    
    # 要傳給 GoPro API 的資料
    form_data = {
        "user_name": current_user_name,
        "user_folder": str(current_user_folder),
        "trajectory_folder": str(trajectory_folder),
        "next_number": str(next_number)
    }
    
    print(f"Sending data to GoPros: {form_data}")  # Debug
    
    async with aiohttp.ClientSession() as session:
        try:
            results = await asyncio.gather(
                post_gopro(session, "http://localhost:3253/stop_recording_and_download", form_data),
                post_gopro(session, "http://localhost:9436/stop_recording_and_download", form_data)
            )
            
            gopro1_result = results[0]
            gopro2_result = results[1]
            
            print(f"GoPro 1 response: {gopro1_result}")  # Debug
            print(f"GoPro 2 response: {gopro2_result}")  # Debug
            
            # 如果兩者都有成功下載資訊，可進一步取得檔案路徑
            if (isinstance(gopro1_result, dict) and 
                isinstance(gopro2_result, dict) and
                "download_status" in gopro1_result and 
                "download_status" in gopro2_result):
                
                side_video = gopro1_result.get("video_path")
                video_45 = gopro2_result.get("video_path")
                
                if side_video and video_45:
                    print(f"Side video path: {side_video}")
                    print(f"45-degree video path: {video_45}")
            
            return {
                "gopro1": gopro1_result,
                "gopro2": gopro2_result,
                "user_name": current_user_name,
                "user_folder": str(current_user_folder),
                "trajectory_folder": str(trajectory_folder),
                "form_data_sent": form_data
            }
        except Exception as e:
            print(f"Error in stop_recording_and_download: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "user_name": current_user_name,
                    "user_folder": str(current_user_folder),
                    "trajectory_folder": str(trajectory_folder)
                }
            )

# ----------------------------------------
# Main Entry
# ----------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
