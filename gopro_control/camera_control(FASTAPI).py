from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess, re
from datetime import datetime
from pathlib import Path
from rich.console import Console
from open_gopro import Params, WirelessGoPro
from bleak import BleakScanner
from typing import List, Dict, Optional

app = FastAPI(title="GoPro Controller API")
console = Console()
gopro_connection = None
last_scan_result: Optional[List[Dict]] = None
is_scanning = False

# Pydantic models
class ConnectRequest(BaseModel):
    mac_address: str
    wifi_interface: str

# GoPro class
class GoPro:
    def __init__(self, mac_address: str, wifi_interface: str):
        self.mac_address = mac_address
        self.wifi_interface = wifi_interface
        self.connection = None

    async def connect(self):
        try:
            self.connection = await WirelessGoPro(
                identifier=self.mac_address,
                wifi_interface=self.wifi_interface,
                target_mac_addr=self.mac_address
            ).__aenter__()
            return True
        except Exception as e:
            console.print(f"[red]連接錯誤: {str(e)}")
            return False

    async def disconnect(self):
        if self.connection:
            await self.connection.__aexit__(None, None, None)
            self.connection = None

# Utility functions
def calculate_distance(rssi: int) -> str:
    if rssi >= -55:
        return "非常近 (0-2公尺)"
    elif -55 > rssi >= -65:
        return "近 (2-5公尺)"
    elif -65 > rssi >= -80:
        return "中等距離 (5-10公尺)"
    return "遠 (10公尺以上)"

# Scanning functions
async def scan_for_gopros():
    global last_scan_result, is_scanning
    if is_scanning:
        return last_scan_result
    
    is_scanning = True
    try:
        devices = await BleakScanner.discover()
        gopro_devices = [
            {
                "名稱": device.name,
                "MAC地址": device.address,
                "訊號強度": f"{device.rssi} dBm",
                "掃描時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "預估距離": calculate_distance(device.rssi)
            }
            for device in devices
            if device.name and "GoPro" in device.name
        ]
        last_scan_result = gopro_devices
        return gopro_devices
    finally:
        is_scanning = False

# API Endpoints
@app.get("/wifi_interface")
async def wifi_interface():
    try:
        output = subprocess.check_output(['netsh', 'wlan', 'show', 'interfaces'], encoding='big5')
        wifi_names = re.findall(r'Name\s+:\s+(.+?)\r?\n', output)
        if not wifi_names:
            raise HTTPException(status_code=404, detail="找不到任何 WiFi 介面")
        return {"status": "success", "message": f"總共有 {len(wifi_names)} 個 WiFi: {' 、 '.join(wifi_names)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"發生錯誤: {str(e)}")

@app.get("/scan_gopro")
async def get_gopro_devices(background_tasks: BackgroundTasks):
    if is_scanning and last_scan_result:
        return {"status": "scanning", "message": "正在掃描中，返回上次掃描結果", "devices": last_scan_result}
    devices = await scan_for_gopros()
    return {"status": "success", "devices": devices if devices else []}

@app.post("/gopro_connect")
async def connect_camera(request: ConnectRequest):
    global gopro_connection
    if gopro_connection:
        await gopro_connection.disconnect()
    
    gopro_connection = GoPro(request.mac_address, request.wifi_interface)
    if await gopro_connection.connect():
        return {"message": f"成功連接到 MAC address 為 {request.mac_address} 的 GoPro"}
    raise HTTPException(status_code=500, detail="無法連接到 GoPro")

@app.post("/start_recording")
async def start_recording():
    if not gopro_connection or not gopro_connection.connection:
        raise HTTPException(status_code=400, detail="GoPro 未連接")
    response = await gopro_connection.connection.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
    if response.ok:
        return {"message": "開始錄影"}
    raise HTTPException(status_code=500, detail="開始錄影失敗")

@app.post("/stop_recording")
async def stop_recording():
    if not gopro_connection or not gopro_connection.connection:
        raise HTTPException(status_code=400, detail="GoPro 未連接")
    response = await gopro_connection.connection.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
    if response.ok:
        return {"message": "停止錄影"}
    raise HTTPException(status_code=500, detail="停止錄影失敗")

@app.post("/download/last")
async def download_last_media():
    if not gopro_connection or not gopro_connection.connection:
        raise HTTPException(status_code=400, detail="GoPro 未連接")
    try:
        last_media = await gopro_connection.connection.http_command.get_last_captured_media()
        if not last_media.ok or not last_media.data:
            raise HTTPException(status_code=404, detail="找不到最後拍攝的媒體檔案")
        
        media_path = str(last_media.data)
        download_dir = Path.home() / "GoPro_Downloads"
        download_dir.mkdir(exist_ok=True)
        
        filename = media_path.split('/')[-1]
        local_file = download_dir / filename
        
        download_response = await gopro_connection.connection.http_command.download_file(
            camera_file=media_path,
            local_file=local_file
        )
        
        if download_response.ok:
            return {"message": f"檔案成功下載到: {str(local_file)}"}
        raise HTTPException(status_code=500, detail="下載失敗")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下載時發生錯誤: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7414)