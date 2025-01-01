import asyncio
import argparse
from rich.console import Console
from open_gopro import Params, WiredGoPro
from open_gopro.util import add_cli_args_and_parse
from pathlib import Path

console = Console()

async def download_media(gopro):
    try:
        last_media = await gopro.http_command.get_last_captured_media()
        if not last_media.ok or not last_media.data:
            console.print("[red]找不到最後拍攝的媒體檔案")
            return

        media_path = str(last_media.data)
        download_dir = Path.home() / "GoPro_Downloads"
        download_dir.mkdir(exist_ok=True)
        
        filename = media_path.split('/')[-1]
        local_file = download_dir / filename
        
        console.print(f"[blue]正在下載最後拍攝的檔案: {filename}...")
        response = await gopro.http_command.download_file(
            camera_file=media_path,
            local_file=local_file
        )
        if response.ok:
            console.print(f"[green]檔案成功下載到: {str(local_file)}")
        else:
            console.print("[red]下載失敗")
    except Exception as e:
        console.print(f"[red]下載時發生錯誤: {str(e)}")

async def control_gopro_by_serial(serial_number: str):
    try:
        console.print(f"[blue]正在嘗試連接 GoPro，序列號: {serial_number}")
        async with WiredGoPro(serial=serial_number) as gopro:
            console.print("[green]成功通過 USB 連接到 GoPro")

            while True:
                console.print("\n=== GoPro 控制選單 ===")
                console.print("1. 開始錄影")
                console.print("2. 停止錄影")
                console.print("3. 查看相機狀態")
                console.print("4. 下載最後拍攝的檔案")
                console.print("5. 離開")

                choice = input("\n請選擇 (1-5): ")

                if choice == "1":
                    response = await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)
                    if response.ok:
                        console.print("[green]開始錄影")
                    else:
                        console.print("[red]錄影失敗")

                elif choice == "2":
                    response = await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)
                    if response.ok:
                        console.print("[green]停止錄影")
                    else:
                        console.print("[red]停止錄影失敗")

                elif choice == "3":
                    try:
                        status = await gopro.http_command.get_camera_state()
                        if status.ok:
                            console.print("[green]相機狀態:")
                            console.print(f"電池電量: {status.data.get('battery_level', '未知')}%")
                            console.print(f"錄影狀態: {'錄影中' if status.data.get('encoding', False) else '未錄影'}")
                            console.print(f"剩餘空間: {status.data.get('remaining_space', '未知')} MB")
                        else:
                            console.print("[red]無法獲取相機狀態")
                    except Exception as e:
                        console.print(f"[red]獲取狀態時發生錯誤: {str(e)}")

                elif choice == "4":
                    await download_media(gopro)

                elif choice == "5":
                    console.print("[blue]退出程序")
                    break
                else:
                    console.print("[red]無效的選擇，請重試")

    except Exception as e:
        console.print(f"[red]錯誤: {str(e)}")

async def main(args: argparse.Namespace) -> int:
    serial_number = "C3531350279436"  
    # serial_number = 'C3531324813253'
    await control_gopro_by_serial(serial_number)
    return 0

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GoPro USB 控制程序（通過序列號）")
    return add_cli_args_and_parse(parser)

if __name__ == "__main__":
    asyncio.run(main(parse_arguments()))