import multiprocessing
import time
import subprocess

def run_script(script_name):
    subprocess.run(['python', script_name])

if __name__ == '__main__':
    # 建立兩個程序
    p1 = multiprocessing.Process(target=run_script, args=('test4.py',))
    p2 = multiprocessing.Process(target=run_script, args=('test5.py',))
    
    # 等待5秒
    print("等待5秒...")
    time.sleep(5)
    
    # 同時啟動兩個程序
    p1.start()
    p2.start()
    
    # 等待兩個程序完成
    p1.join()
    p2.join()
    
    print("所有程序執行完成")