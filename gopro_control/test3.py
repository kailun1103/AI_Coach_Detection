import nmap
import socket
import netifaces

def scan_network():
    # 獲取當前設備的IP地址和子網掩碼
    gateways = netifaces.gateways()
    default_gateway = gateways['default'][netifaces.AF_INET][0]
    
    # 創建一個nmap掃描器實例
    nm = nmap.PortScanner()
    
    # 掃描整個子網
    # 假設子網掩碼是/24 (255.255.255.0)
    network = default_gateway.rsplit('.', 1)[0] + '.0/24'
    
    print(f"開始掃描網絡: {network}")
    nm.scan(hosts=network, arguments='-sn')
    
    # 遍歷所有活動主機
    for host in nm.all_hosts():
        try:
            # 嘗試獲取主機名
            hostname = socket.gethostbyaddr(host)[0]
            # 檢查是否為GoPro設備
            if 'gopro' in hostname.lower():
                print(f"找到GoPro設備:")
                print(f"IP地址: {host}")
                print(f"主機名: {hostname}")
        except:
            # 如果無法獲取主機名，只顯示IP
            print(f"活動主機: {host}")

if __name__ == "__main__":
    scan_network()