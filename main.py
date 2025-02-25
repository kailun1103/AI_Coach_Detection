import time
import json
import asyncio
from pathlib import Path
from enum import Enum
import numpy as np
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

from sound2 import play_sound
from processing_trajectory import processing_trajectory
from trajectory_gpt_overall_feedback import find_and_format_feedback_jsons, conclude

# ------------------------------
# Calibration Matrices
# ------------------------------
P1 = np.array([
    [1856.204034,     0.000000, 1842.334089,     0.000000],
    [   0.000000, 1848.190924, 1072.463818,     0.000000],
    [   0.000000,     0.000000,    1.000000,     0.000000],
])

P2 = np.array([
    [689.640601,   -3.080844, 2543.075304, -1723276.428567],
    [-565.656293, 1800.162272,  830.494335,  689370.702720],
    [  -0.508536,   -0.037628,    0.860218,    604.500373],
])

# ------------------------------
# Global Variables
# ------------------------------
current_user_folder: Optional[Path] = None
current_user_name: Optional[str] = None

yolo_pose_model: Optional[YOLO] = None
yolo_tennis_ball_model: Optional[YOLO] = None

# ------------------------------
# FastAPI App Initialization
# ------------------------------
app = FastAPI(title="GoPro Controller API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------
# Enums & Data Models
# ------------------------------
class DominantHand(str, Enum):
    right = "right"
    left = "left"

class UserData(BaseModel):
    name: str
    height: float
    dominant_hand: str

# ------------------------------
# Utility Functions
# ------------------------------
def find_next_trajectory_number(base_folder: Path) -> int:
    """
    根據 base_folder 中現有的軌跡資料夾，返回下一個可用的編號。
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

async def post_gopro(session: aiohttp.ClientSession, url: str, data: Optional[dict] = None) -> dict:
    """
    封裝對 GoPro API 發送 POST 請求。
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

async def wait_for_file_ready(file_path: str, timeout: int = 120, check_interval: int = 2) -> bool:
    """
    檢查檔案是否完成寫入，依據檔案大小是否穩定來確認。
    """
    path = Path(file_path)
    if not path.exists():
        return False

    start_time = time.time()
    last_size = path.stat().st_size

    while True:
        if time.time() - start_time > timeout:
            print(f"Timeout waiting for {file_path} to complete")
            return path.exists()
        
        await asyncio.sleep(check_interval)

        if not path.exists():
            return False

        current_size = path.stat().st_size
        if current_size == last_size:
            await asyncio.sleep(check_interval)
            if path.exists() and path.stat().st_size == current_size:
                print(f"File {file_path} is ready with size {current_size} bytes")
                return True
        last_size = current_size
        print(f"File {file_path} still being written, current size: {current_size} bytes")

async def process_trajectory_async(P1, P2, pose_model, ball_model, side_video, video_45, knn_dataset: str):
    """
    背景任務：在背景執行 processing_trajectory 避免阻塞主線程。
    """
    try:
        await asyncio.to_thread(
            processing_trajectory,
            P1, P2, pose_model, ball_model,
            side_video, video_45, knn_dataset
        )
        print("軌跡處理完成！")
    except Exception as e:
        print(f"軌跡處理過程中發生錯誤: {str(e)}")

# ------------------------------
# Application Startup Event
# ------------------------------
@app.on_event("startup")
async def startup_event():
    """
    伺服器啟動時載入 YOLO 模型。
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

# ------------------------------
# API Endpoints
# ------------------------------
@app.get("/model_status")
async def check_model_status():
    """
    回傳 YOLO 模型是否已成功載入。
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
    接收使用者資料，建立使用者專屬資料夾與 JSON 記錄。
    """
    start_time = time.time()
    try:
        if dominant_hand not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail="dominant_hand must be 0 (left) or 1 (right)"
            )
        
        hand = "left" if dominant_hand == 0 else "right"
        global current_user_folder, current_user_name
        current_user_name = name

        # 建立使用者資料夾
        current_user_folder = Path(f"trajectory/{name}__trajectory")
        current_user_folder.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "name": name,
            "height": height,
            "hand": hand,
            "timestamp": time.strftime("%Y_%m_%d_%H_%M_%S"),
            "file_path": str(current_user_folder)
        }

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

@app.get("/gpt_response")
async def gpt_response():
    """
    檢查使用者資料，執行 GPT 回饋處理，並回傳最終結論。
    """
    global current_user_folder, current_user_name
    if not current_user_folder or not current_user_name:
        raise HTTPException(
            status_code=400,
            detail="User information not found. Please call /input_data first."
        )
    
    try:
        gpt_single_results = await asyncio.to_thread(
            find_and_format_feedback_jsons,
            current_user_folder
        )
        final_conclusion = await asyncio.to_thread(
            conclude,
            gpt_single_results
        )
        return {
            "status": "success",
            "user_name": current_user_name,
            "conclusion": final_conclusion
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "user_name": current_user_name,
                "user_folder": str(current_user_folder)
            }
        )

@app.get("/take_photo")
async def take_photo():
    """
    同時向兩台 GoPro 發送拍照請求。
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
    同時向兩台 GoPro 發送開始錄影請求。
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
    同時向兩台 GoPro 發送停止錄影請求。
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
async def stop_recording_and_download(background_tasks: BackgroundTasks):
    """
    同時對兩台 GoPro 停止錄影並下載影片，建立軌跡資料夾，檢查檔案是否準備好後，啟動背景任務進行軌跡處理。
    """
    global current_user_folder, current_user_name
    if not current_user_folder or not current_user_name:
        raise HTTPException(
            status_code=400,
            detail="User information not found. Please call /input_data first."
        )
    
    base_folder = Path(current_user_folder)
    base_folder.mkdir(parents=True, exist_ok=True)
    
    next_number = find_next_trajectory_number(base_folder)
    trajectory_folder = base_folder / f"trajectory__{next_number}"
    trajectory_folder.mkdir(parents=True, exist_ok=True)
    
    form_data = {
        "user_name": current_user_name,
        "user_folder": str(current_user_folder),
        "trajectory_folder": str(trajectory_folder),
        "next_number": str(next_number)
    }
    
    print(f"Sending data to GoPros: {form_data}")
    
    async with aiohttp.ClientSession() as session:
        try:
            results = await asyncio.gather(
                post_gopro(session, "http://localhost:3253/stop_recording_and_download", form_data),
                post_gopro(session, "http://localhost:9436/stop_recording_and_download", form_data)
            )
            gopro1_result, gopro2_result = results[0], results[1]
            print(f"GoPro 1 response: {gopro1_result}")
            print(f"GoPro 2 response: {gopro2_result}")
            
            side_video_path = gopro1_result.get("video_path")
            video_45_path = gopro2_result.get("video_path")
            video_files_ready = False
            
            if (isinstance(gopro1_result, dict) and isinstance(gopro2_result, dict) and
                "download_status" in gopro1_result and "download_status" in gopro2_result):
                if (side_video_path and video_45_path and 
                    Path(side_video_path).exists() and Path(video_45_path).exists()):
                    
                    side_video_ready = await wait_for_file_ready(side_video_path)
                    video_45_ready = await wait_for_file_ready(video_45_path)
                    
                    if side_video_ready and video_45_ready:
                        video_files_ready = True
                        print("Both videos confirmed ready")
                        play_sound()
                        background_tasks.add_task(
                            process_trajectory_async,
                            P1, P2,
                            yolo_pose_model,
                            yolo_tennis_ball_model,
                            side_video_path,
                            video_45_path,
                            'knn_dataset.json'
                        )
                    else:
                        if not side_video_ready:
                            print(f"Side video not fully written at: {side_video_path}")
                        if not video_45_ready:
                            print(f"45-degree video not fully written at: {video_45_path}")
                else:
                    if not side_video_path or not Path(side_video_path).exists():
                        print(f"Side video not found at: {side_video_path}")
                    if not video_45_path or not Path(video_45_path).exists():
                        print(f"45-degree video not found at: {video_45_path}")
            
            response_data = {
                "gopro1": gopro1_result,
                "gopro2": gopro2_result,
                "user_name": current_user_name,
                "user_folder": str(current_user_folder),
                "trajectory_folder": str(trajectory_folder),
                "videos_ready": video_files_ready,
                "form_data_sent": form_data
            }
            
            if video_files_ready:
                response_data["side_video"] = side_video_path
                response_data["video_45"] = video_45_path
            
            return response_data
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

# ------------------------------
# Main Entry Point
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)