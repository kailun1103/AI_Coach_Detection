from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from open_gopro import WiredGoPro, Params
from pathlib import Path
from typing import Optional
import uvicorn
import asyncio
import cv2
import time

app = FastAPI(title="GoPro Controller API")

# Global variable to store GoPro instance
gopro_instance: Optional[WiredGoPro] = None

# SERIAL_NUMBER = "C3531350279436"  # GoPro serial number
SERIAL_NUMBER = "C3531324813253"  # GoPro serial number
DOWNLOAD_MEDIA_NAME = 'test2.mp4'  # download_name
DOWNLOAD_PATH = Path(r"C:\Users\d93xj\OneDrive\Desktop\AI_Coach_Detection")  # download_path

async def download_latest_media(custom_filename: str = None):
    """Download the latest media file"""
    start_time = time.time()
    gopro = await get_gopro()
    try:
        last_media = await gopro.http_command.get_last_captured_media()
        if not last_media.ok or not last_media.data:
            execution_time = time.time() - start_time
            raise HTTPException(
                status_code=404, 
                detail={
                    "error": "Last captured media file not found",
                    "execution_time": f"{execution_time:.2f} seconds"
                }
            )

        media_path = str(last_media.data)
        
        # Ensure download directory exists
        DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
        
        # Use custom filename or original filename
        if custom_filename:
            local_file = DOWNLOAD_PATH / custom_filename
        else:
            filename = media_path.split('/')[-1]
            local_file = DOWNLOAD_PATH / filename
        
        response = await gopro.http_command.download_file(
            camera_file=media_path,
            local_file=local_file
        )
        
        execution_time = time.time() - start_time
        if response.ok:
            return {
                "message": f"File successfully downloaded to: {str(local_file)}",
                "execution_time": f"{execution_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Download failed",
                    "execution_time": f"{execution_time:.2f} seconds"
                }
            )
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "execution_time": f"{execution_time:.2f} seconds"
            }
        )
    
def get_video_info(video_path: str) -> tuple[float, int]:
    """Get video duration and total frames"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    return duration, total_frames

async def get_gopro():
    """Get or create GoPro instance"""
    global gopro_instance
    if gopro_instance is None:
        try:
            gopro_instance = WiredGoPro(serial=SERIAL_NUMBER)
            await gopro_instance.open()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to connect to GoPro: {str(e)}")
    return gopro_instance

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up GoPro connection when application shuts down"""
    global gopro_instance
    if gopro_instance:
        await gopro_instance.close()

@app.get("/connect")
async def connect_gopro():
    """Connect to GoPro"""
    start_time = time.time()
    try:
        await get_gopro()
        execution_time = time.time() - start_time
        return {
            "message": "Successfully connected to GoPro",
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

@app.post("/take_photo")
async def take_photo():
    """Take photo - Make sure GoPro is manually switched to photo mode"""
    start_time = time.time()
    gopro = await get_gopro()
    try:
        # Take photo
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
        # Wait for photo capture to complete
        await asyncio.sleep(0.5)
        # Reset shutter
        await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        
        execution_time = time.time() - start_time
        if response.ok:
            return {
                "message": "Photo captured successfully",
                "execution_time": f"{execution_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to capture photo",
                    "execution_time": f"{execution_time:.2f} seconds"
                }
            )
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{execution_time:.2f} seconds"
            }
        )
    
@app.post("/start_recording")
async def start_recording():
    """Start recording - Make sure GoPro is manually switched to video mode"""
    start_time = time.time()
    gopro = await get_gopro()
    try:
        # Start recording directly
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
        execution_time = time.time() - start_time
        if response.ok:
            return {
                "message": "Recording started",
                "execution_time": f"{execution_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to start recording",
                    "execution_time": f"{execution_time:.2f} seconds"
                }
            )
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "execution_time": f"{execution_time:.2f} seconds"
            }
        )

@app.post("/stop_recording")
async def stop_recording():
    """Stop recording - Make sure GoPro is manually switched to video mode"""
    start_time = time.time()
    gopro = await get_gopro()
    try:
        # Stop recording
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        execution_time = time.time() - start_time
        
        if response.ok:
            return {
                "message": "Recording stopped successfully",
                "execution_time": f"{execution_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to stop recording",
                    "execution_time": f"{execution_time:.2f} seconds"
                }
            )
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "execution_time": f"{execution_time:.2f} seconds"
            }
        )

@app.post("/stop_recording_and_download")
async def stop_recording_and_download():
    """Stop recording and download the file"""
    start_time = time.time()
    gopro = await get_gopro()
    try:
        # Stop recording
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        if not response.ok:
            execution_time = time.time() - start_time
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to stop recording",
                    "execution_time": f"{execution_time:.2f} seconds"
                }
            )
        
        # Wait to ensure file is completely saved
        await asyncio.sleep(2)
        
        # Download file
        video_path = DOWNLOAD_PATH / DOWNLOAD_MEDIA_NAME
        download_result = await download_latest_media(DOWNLOAD_MEDIA_NAME)
        
        # Get video information
        duration, total_frames = get_video_info(str(video_path))
        
        execution_time = time.time() - start_time
        
        return {
            "message": "Recording stopped successfully",
            "download": f"File successfully downloaded to: {str(video_path)}",
            "video_length": f"{duration:.2f} seconds ({total_frames} frames)",
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

@app.get("/download_last")
async def download_last_media():
    """Download the last captured media file"""
    return await download_latest_media()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3253)