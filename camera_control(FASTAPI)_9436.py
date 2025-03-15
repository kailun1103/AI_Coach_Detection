import time
import json
import asyncio
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pathlib import Path
from typing import Optional

from open_gopro import WiredGoPro, Params

app = FastAPI(title="GoPro Controller API")

# -----------------------------------------------------
# Global Variables
# -----------------------------------------------------
SERIAL_NUMBER = "C3531350279436"  # GoPro serial number
gopro_instance: Optional[WiredGoPro] = None


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------
async def get_gopro() -> WiredGoPro:
    """
    Get or create the GoPro instance for subsequent operations.
    If the instance is not yet initialized, create and open it.
    """
    global gopro_instance
    if gopro_instance is None:
        try:
            gopro_instance = WiredGoPro(serial=SERIAL_NUMBER)
            await gopro_instance.open()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to connect to GoPro: {str(e)}"
            )
    return gopro_instance

def get_video_info(video_path: str) -> tuple[float, int]:
    """
    Get the duration (in seconds) and total frame count of the given video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    cap.release()
    return duration, total_frames

async def download_latest_media(download_path: Path, custom_filename: str = None) -> dict:
    """
    Download the latest media file from the GoPro to the specified folder.
    A custom filename can be provided; otherwise a timestamped filename is used.
    """
    start_time = time.time()
    gopro = await get_gopro()
    try:
        last_media = await gopro.http_command.get_last_captured_media()
        if not last_media.ok or not last_media.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Last captured media file not found",
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )
        media_path = str(last_media.data)

        # Ensure download path exists
        download_path.mkdir(parents=True, exist_ok=True)

        # Decide on the local file name
        if custom_filename:
            local_file = download_path / custom_filename
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            extension = media_path.split('.')[-1]
            filename = f"recording_{timestamp}.{extension}"
            local_file = download_path / filename

        # Download the file
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
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        )


# -----------------------------------------------------
# Application Lifecycle Events
# -----------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up GoPro connection when the application shuts down.
    """
    global gopro_instance
    if gopro_instance:
        await gopro_instance.close()


# -----------------------------------------------------
# API Endpoints
# -----------------------------------------------------
@app.get("/connect")
async def connect_gopro():
    """
    Connect to the GoPro camera.
    """
    start_time = time.time()
    try:
        await get_gopro()
        return {
            "message": "Successfully connected to GoPro",
            "execution_time": f"{time.time() - start_time:.2f} seconds"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        )

@app.post("/take_photo")
async def take_photo():
    """
    Capture a photo with the GoPro.
    Note: Ensure the GoPro is manually set to photo mode.
    """
    start_time = time.time()
    gopro = await get_gopro()
    try:
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
        await asyncio.sleep(0.5)  # Wait for the photo to be captured
        await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)

        if response.ok:
            return {
                "message": "Photo captured successfully",
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to capture photo",
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        )

@app.post("/start_recording")
async def start_recording():
    """
    Start video recording on the GoPro.
    Note: Ensure the GoPro is manually set to video mode.
    """
    start_time = time.time()
    gopro = await get_gopro()
    try:
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
        if response.ok:
            return {
                "message": "Recording started",
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to start recording",
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        )

@app.post("/stop_recording")
async def stop_recording():
    """
    Stop the current video recording on the GoPro.
    """
    start_time = time.time()
    gopro = await get_gopro()
    try:
        response = await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        if response.ok:
            return {
                "message": "Recording stopped successfully",
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to stop recording",
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        )

@app.post("/download")
async def download(
    user_name: str = Form(..., description="User name for the recording"),
    user_folder: str = Form(..., description="User folder path"),
    trajectory_folder: str = Form(..., description="Trajectory folder path"),
    next_number: str = Form(..., description="Next trajectory number")
):
    """
    Stop the current recording and download the video file to a specific trajectory folder.
    """
    start_time = time.time()
    gopro = await get_gopro()
    try:
        # Stop recording
        # stop_response = await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
        # if not stop_response.ok:
        #     raise HTTPException(
        #         status_code=500,
        #         detail={
        #             "error": "Failed to stop recording",
        #             "execution_time": f"{time.time() - start_time:.2f} seconds"
        #         }
        #     )

        # # Wait for the file to be finalized on the GoPro
        # await asyncio.sleep(2)

        # Convert string paths to Path objects
        trajectory_path = Path(trajectory_folder)
        trajectory_path.mkdir(parents=True, exist_ok=True)

        # Generate filename using user_name and next_number
        filename = f"{user_name}__{next_number}_45.mp4"
        local_file = trajectory_path / filename

        # Get last media info
        last_media = await gopro.http_command.get_last_captured_media()
        if not last_media.ok or not last_media.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Last captured media file not found",
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )

        media_path = str(last_media.data)

        # Download the file
        download_response = await gopro.http_command.download_file(
            camera_file=media_path,
            local_file=local_file
        )
        if not download_response.ok:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to download file",
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )

        # Verify the downloaded file
        if not local_file.exists() or local_file.stat().st_size == 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Video file not found or empty after download",
                    "attempted_path": str(local_file),
                    "execution_time": f"{time.time() - start_time:.2f} seconds"
                }
            )

        # Gather video info
        try:
            duration, total_frames = get_video_info(str(local_file))
            return {
                "message": "Recording stopped and downloaded successfully",
                "trajectory_folder": str(trajectory_path),
                "video_path": str(local_file),
                "download_status": "Complete",
                "video_length": f"{duration:.2f} seconds ({total_frames} frames)",
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        except ValueError as ve:
            # If reading video info fails, still return success but with a warning
            return {
                "message": "Recording stopped and downloaded successfully",
                "trajectory_folder": str(trajectory_path),
                "video_path": str(local_file),
                "download_status": "Complete",
                "video_info_error": str(ve),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.2f} seconds"
            }
        )


# -----------------------------------------------------
# Main Execution
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9436)