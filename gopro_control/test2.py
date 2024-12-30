import requests
import time
import argparse
import subprocess
import platform
from rich.console import Console
from rich.table import Table
from typing import Dict, Optional
import json

console = Console()

class SimpleGoProControl:
    def __init__(self, ip: str = "10.5.5.9", interface: Optional[str] = None):
        self.base_url = f"http://{ip}/gp/gpControl"
        self.status_url = f"{self.base_url}/status"
        self.command_url = f"{self.base_url}/command"
        self.interface = interface
        self.is_windows = platform.system() == "Windows"
        
        if self.interface:
            self._setup_interface()
    
    def _setup_interface(self):
        try:
            if self.is_windows:
                # 檢查網路介面狀態
                check_status = subprocess.run(
                    ['netsh', 'interface', 'show', 'interface', f'name="{self.interface}"'],
                    capture_output=True,
                    text=True
                )
                console.print("[yellow]網路介面狀態:")
                console.print(check_status.stdout)

                # 啟用網路介面
                enable_cmd = subprocess.run(
                    ['netsh', 'interface', 'set', 'interface', f'name="{self.interface}"', 'admin=enabled'],
                    capture_output=True,
                    text=True
                )
                
                if enable_cmd.returncode == 0:
                    console.print(f"[green]成功啟用網路介面 {self.interface}")
                else:
                    console.print("[red]啟用網路介面時出現錯誤")
                    console.print(f"[yellow]錯誤輸出: {enable_cmd.stderr}")
                    console.print(f"[yellow]命令輸出: {enable_cmd.stdout}")
            else:
                # Linux/Unix 系統使用 ifconfig
                subprocess.run(['sudo', 'ifconfig', self.interface, 'up'], 
                             check=True,
                             capture_output=True,
                             text=True)
                console.print(f"[green]成功啟用網路介面 {self.interface}")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]設置網路介面時發生錯誤")
            console.print(f"[red]錯誤訊息: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            console.print("[yellow]請確保你有足夠的權限執行此操作")
        except Exception as e:
            console.print(f"[red]設置網路介面時發生未預期錯誤: {str(e)}")
        
    def _send_command(self, command: str, params: Dict = None) -> bool:
        try:
            url = f"{self.command_url}/{command}"
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            console.print(f"[red]發送命令時發生錯誤: {str(e)}")
            return False

    def start_recording(self) -> bool:
        return self._send_command("shutter", {"p": 1})

    def stop_recording(self) -> bool:
        return self._send_command("shutter", {"p": 0})

    def get_status(self) -> Optional[Dict]:
        try:
            response = requests.get(self.status_url, timeout=5)
            console.print("[yellow]API 回應狀態碼:", response.status_code)
            
            if response.status_code == 200:
                data = response.json()
                console.print("[yellow]原始回應數據:")
                console.print(json.dumps(data, indent=2))
                
                try:
                    status = data.get("status", {})
                    console.print("[yellow]Status 數據:")
                    console.print(json.dumps(status, indent=2))
                    
                    battery = status.get("2", {}).get("1", 0) if isinstance(status.get("2"), dict) else 0
                    recording = status.get("8", {}).get("13", 0) if isinstance(status.get("8"), dict) else 0
                    sd_space = status.get("54", {}).get("1", 0) if isinstance(status.get("54"), dict) else 0
                    
                    return {
                        "battery": battery,
                        "recording": recording == 1,
                        "sd_space": sd_space,
                        "connected": True
                    }
                except Exception as e:
                    console.print(f"[red]解析狀態數據時發生錯誤: {str(e)}")
                    return None
            
            console.print(f"[red]無法連接到相機 (HTTP {response.status_code})")
            return None
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]連接錯誤: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            console.print(f"[red]JSON 解析錯誤: {str(e)}")
            return None
        except Exception as e:
            console.print(f"[red]未預期的錯誤: {str(e)}")
            console.print(f"錯誤類型: {type(e)}")
            return None

def display_status(status: Optional[Dict]):
    if not status:
        console.print("[red]無法獲取相機狀態")
        return

    table = Table(title="GoPro 狀態", show_header=False)
    table.add_column("項目", style="cyan")
    table.add_column("狀態", style="green")
    
    table.add_row("電池電量", f"{status['battery']}%")
    table.add_row("錄影狀態", "錄影中" if status['recording'] else "未錄影")
    table.add_row("SD卡剩餘空間", f"{status['sd_space']} MB")
    
    console.print(table)

def get_network_interfaces():
    """獲取系統上的網路介面列表"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['netsh', 'interface', 'show', 'interface'],
                capture_output=True,
                text=True
            )
            console.print("[green]可用的網路介面:")
            console.print(result.stdout)
        else:
            result = subprocess.run(
                ['ifconfig', '-a'],
                capture_output=True,
                text=True
            )
            console.print("[green]可用的網路介面:")
            console.print(result.stdout)
    except Exception as e:
        console.print(f"[red]獲取網路介面列表時發生錯誤: {str(e)}")

def main():
    # 設置命令列參數
    parser = argparse.ArgumentParser(description='GoPro 控制程式')
    parser.add_argument('--ip', default='10.5.5.9', help='GoPro 的 IP 位址 (預設: 10.5.5.9)')
    parser.add_argument('--interface', help='指定要使用的網路介面 (例如: "Wi-Fi" 或 "wlan0")')
    parser.add_argument('--list-interfaces', action='store_true', help='顯示所有可用的網路介面')
    args = parser.parse_args()

    if args.list_interfaces:
        get_network_interfaces()
        return

    console.print(f"[yellow]正在初始化 GoPro 控制...\n")
    console.print(f"[yellow]使用 IP: {args.ip}")
    if args.interface:
        console.print(f"[yellow]使用網路介面: {args.interface}")

    gopro = SimpleGoProControl(ip=args.ip, interface=args.interface)

    # 檢查連接並獲取除錯資訊
    try:
        status = gopro.get_status()
        if not status:
            console.print("[red]無法連接到 GoPro。請確保：")
            console.print("1. GoPro 已開啟")
            console.print("2. 你的電腦已連接到 GoPro 的 WiFi")
            console.print(f"3. GoPro 的 IP 是 {args.ip}")
            console.print("\n要查看 GoPro 的實際 IP，可以：")
            console.print("1. 在 GoPro 的設定中查看")
            console.print("2. 或使用網路工具檢查連接的 IP")
            console.print("\n要查看可用的網路介面，請使用：")
            console.print("python script.py --list-interfaces")
            return

        console.print("[green]成功連接到 GoPro!\n")

    except Exception as e:
        console.print(f"[red]初始化時發生錯誤: {str(e)}")
        return

    while True:
        console.print("\n=== GoPro 控制選單 ===")
        console.print("1. 開始錄影")
        console.print("2. 停止錄影")
        console.print("3. 查看相機狀態")
        console.print("4. 離開")

        choice = input("\n請選擇 (1-4): ")

        if choice == "1":
            console.print("\n[yellow]正在開始錄影...")
            if gopro.start_recording():
                console.print("[green]成功開始錄影")
            else:
                console.print("[red]開始錄影失敗")

        elif choice == "2":
            console.print("\n[yellow]正在停止錄影...")
            if gopro.stop_recording():
                console.print("[green]成功停止錄影")
            else:
                console.print("[red]停止錄影失敗")

        elif choice == "3":
            console.print("\n[yellow]正在獲取相機狀態...")
            status = gopro.get_status()
            display_status(status)

        elif choice == "4":
            console.print("\n[green]感謝使用！再見！")
            break

        else:
            console.print("[red]無效的選擇，請重試")
        
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]程式被使用者中斷")
    except Exception as e:
        console.print(f"\n[red]發生未預期的錯誤: {str(e)}")