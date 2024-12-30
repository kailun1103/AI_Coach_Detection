import asyncio
import argparse
from rich.console import Console
from open_gopro import Params, WirelessGoPro
from open_gopro.util import add_cli_args_and_parse
import wifi # 需要安裝 python-wifi 套件
import subprocess

console = Console()

def get_wifi_interfaces():
    """獲取系統上所有可用的 WiFi 介面"""
    try:
        # 使用 iwconfig 命令獲取無線網卡資訊
        result = subprocess.run(['iwconfig'], capture_output=True, text=True)
        interfaces = []
        for line in result.stdout.split('\n'):
            if 'IEEE 802.11' in line:  # 這表示是一個 WiFi 介面
                interfaces.append(line.split()[0])
        return interfaces
    except:
        return []

async def main(args: argparse.Namespace) -> int:
    try:
        # 檢查是否提供了 MAC address
        if not args.mac_address:
            console.print("[red]錯誤: 請提供 GoPro 的 MAC address")
            return -1

        # 如果沒有指定 WiFi 介面，顯示可用的介面供選擇
        if not args.wifi_interface:
            available_interfaces = get_wifi_interfaces()
            if not available_interfaces:
                console.print("[red]錯誤: 找不到可用的 WiFi 介面")
                return -1
            
            console.print("\n=== 可用的 WiFi 介面 ===")
            for i, interface in enumerate(available_interfaces, 1):
                console.print(f"{i}. {interface}")
            
            while True:
                try:
                    choice = int(input("\n請選擇 WiFi 介面 (輸入數字): ")) - 1
                    if 0 <= choice < len(available_interfaces):
                        args.wifi_interface = available_interfaces[choice]
                        break
                    else:
                        console.print("[red]無效的選擇，請重試")
                except ValueError:
                    console.print("[red]請輸入有效的數字")

        # 使用無線連接，並指定 MAC address 和 WiFi 介面
        async with WirelessGoPro(
            identifier=args.mac_address,
            wifi_interface=args.wifi_interface,
            target_mac_addr=args.mac_address
        ) as gopro:
            console.print(f"[green]成功連接到 MAC address 為 {args.mac_address} 的 GoPro")
            console.print(f"[green]使用的 WiFi 介面: {args.wifi_interface}")

            while True:
                console.print("\n=== GoPro 控制選單 ===")
                console.print("1. 開始錄影")
                console.print("2. 停止錄影")
                console.print("3. 查看相機狀態")
                console.print("4. 離開")

                choice = input("\n請選擇 (1-4): ")

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
                    break
                else:
                    console.print("[red]無效的選擇，請重試")

    except Exception as e:
        console.print(f"[red]錯誤: {str(e)}")
        return -1

    return 0

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GoPro 錄影控制")
    parser.add_argument(
        "--mac-address",
        type=str,
        help="GoPro 相機的 MAC address (格式: XX:XX:XX:XX:XX:XX)",
        required=True
    )
    parser.add_argument(
        "--wifi-interface",
        type=str,
        help="要使用的 WiFi 介面名稱 (例如: wlan0)",
        required=False
    )
    return add_cli_args_and_parse(parser)

if __name__ == "__main__":
    asyncio.run(main(parse_arguments()))


# python camera_control.py --mac-address F0:9F:BD:09:78:C1 --wifi-interface "Wi-Fi"
# python camera_control.py --mac-address D6:45:83:C4:E3:C2 --wifi-interface "Wi-Fi 2"