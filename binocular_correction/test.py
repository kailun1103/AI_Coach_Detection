import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def check_specific_chessboard(image_path, width=10, height=7):
    """
    檢查圖像中是否有特定大小的棋盤格
    
    Parameters:
    image_path (str): 圖像路徑
    width (int): 棋盤格寬度（列數）
    height (int): 棋盤格高度（行數）
    
    Returns:
    tuple: (是否檢測到, 角點, 帶角點的圖像)
    """
    # 讀取圖像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"無法讀取圖像: {image_path}")
        return False, None, None
    
    # 創建彩色圖像用於顯示
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 檢測棋盤格
    pattern_size = (width, height)
    ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
    
    if ret:
        # 如果檢測到，進行角點精確化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        
        # 繪製角點
        img_with_corners = img_color.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret)
        
        print(f"成功檢測到 {width}x{height} 的棋盤格，共 {len(corners)} 個角點")
        return True, corners, img_with_corners
    else:
        # 還可以嘗試反轉寬高
        reversed_pattern = (height, width)
        ret, corners = cv2.findChessboardCorners(img, reversed_pattern, None)
        
        if ret:
            # 進行角點精確化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            
            # 繪製角點
            img_with_corners = img_color.copy()
            cv2.drawChessboardCorners(img_with_corners, reversed_pattern, corners, ret)
            
            print(f"成功檢測到 {height}x{width} 的棋盤格，共 {len(corners)} 個角點")
            return True, corners, img_with_corners
        else:
            print(f"未檢測到 {width}x{height} 或 {height}x{width} 的棋盤格")
            return False, None, img_color

def try_detect_with_params(image_path, width=10, height=7, try_alternative=True):
    """
    使用多種參數嘗試檢測棋盤格
    
    Parameters:
    image_path (str): 圖像路徑
    width (int): 棋盤格寬度
    height (int): 棋盤格高度
    try_alternative (bool): 是否嘗試替代參數
    
    Returns:
    tuple: (最佳檢測結果, 最佳參數設置)
    """
    # 基本檢測
    ret, corners, img = check_specific_chessboard(image_path, width, height)
    if ret:
        return (ret, corners, img), ("基本設置", width, height)
    
    # 如果基本檢測失敗且需要嘗試替代參數
    if try_alternative:
        # 嘗試不同的預處理方法
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 1. 嘗試調整亮度和對比度
        adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        ret, corners = cv2.findChessboardCorners(adjusted, (width, height), None)
        if ret:
            cv2.cornerSubPix(adjusted, corners, (11, 11), (-1, -1), 
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_with_corners = img_color.copy()
            cv2.drawChessboardCorners(img_with_corners, (width, height), corners, ret)
            print(f"使用亮度調整後檢測到 {width}x{height} 的棋盤格")
            return (ret, corners, img_with_corners), ("亮度調整", width, height)
        
        # 2. 嘗試高斯模糊
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        ret, corners = cv2.findChessboardCorners(blurred, (width, height), None)
        if ret:
            cv2.cornerSubPix(blurred, corners, (11, 11), (-1, -1), 
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_with_corners = img_color.copy()
            cv2.drawChessboardCorners(img_with_corners, (width, height), corners, ret)
            print(f"使用高斯模糊後檢測到 {width}x{height} 的棋盤格")
            return (ret, corners, img_with_corners), ("高斯模糊", width, height)
        
        # 3. 嘗試自適應閾值
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        ret, corners = cv2.findChessboardCorners(thresh, (width, height), None)
        if ret:
            cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), 
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_with_corners = img_color.copy()
            cv2.drawChessboardCorners(img_with_corners, (width, height), corners, ret)
            print(f"使用自適應閾值後檢測到 {width}x{height} 的棋盤格")
            return (ret, corners, img_with_corners), ("自適應閾值", width, height)
        
        # 4. 嘗試 CLAHE（對比度受限自適應直方圖均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        ret, corners = cv2.findChessboardCorners(cl1, (width, height), None)
        if ret:
            cv2.cornerSubPix(cl1, corners, (11, 11), (-1, -1), 
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_with_corners = img_color.copy()
            cv2.drawChessboardCorners(img_with_corners, (width, height), corners, ret)
            print(f"使用 CLAHE 後檢測到 {width}x{height} 的棋盤格")
            return (ret, corners, img_with_corners), ("CLAHE", width, height)
        
        # 5. 嘗試倒置圖像
        inverted = cv2.bitwise_not(img)
        ret, corners = cv2.findChessboardCorners(inverted, (width, height), None)
        if ret:
            cv2.cornerSubPix(inverted, corners, (11, 11), (-1, -1), 
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_with_corners = img_color.copy()
            cv2.drawChessboardCorners(img_with_corners, (width, height), corners, ret)
            print(f"使用倒置圖像後檢測到 {width}x{height} 的棋盤格")
            return (ret, corners, img_with_corners), ("倒置圖像", width, height)
        
        # 也嘗試反轉寬高
        reversed_result = try_detect_with_params(image_path, height, width, False)
        if reversed_result[0][0]:
            return reversed_result
    
    return (False, None, img_color), ("無法檢測", width, height)

def batch_detect_chessboard(directory, width=10, height=7, pattern="*.JPG"):
    """批量處理目錄中的所有圖像"""
    from glob import glob
    
    # 獲取目錄中的所有圖像
    image_paths = glob(os.path.join(directory, pattern))
    results = {}
    
    print(f"在 {directory} 中找到 {len(image_paths)} 張圖像")
    
    output_dir = Path("chessboard_detection_results")
    output_dir.mkdir(exist_ok=True)
    
    # 統計信息
    successful_detections = 0
    failed_detections = 0
    detection_methods = {}
    
    for img_path in image_paths:
        print(f"\n處理圖像: {os.path.basename(img_path)}")
        (ret, corners, img), (method, w, h) = try_detect_with_params(img_path, width, height)
        
        if ret:
            successful_detections += 1
            
            # 統計成功的方法
            if method in detection_methods:
                detection_methods[method] += 1
            else:
                detection_methods[method] = 1
            
            # 保存檢測結果
            img_name = Path(img_path).stem
            output_path = output_dir / f"{img_name}_detected_{w}x{h}_{method}.jpg"
            cv2.imwrite(str(output_path), img)
            
            print(f"成功！使用方法: {method}, 大小: {w}x{h}")
        else:
            failed_detections += 1
            print(f"失敗！未能檢測到 {width}x{height} 的棋盤格")
    
    # 輸出統計信息
    print("\n====== 檢測統計 ======")
    print(f"總圖像數: {len(image_paths)}")
    print(f"成功檢測: {successful_detections} ({successful_detections/len(image_paths)*100:.1f}%)")
    print(f"失敗檢測: {failed_detections} ({failed_detections/len(image_paths)*100:.1f}%)")
    
    if detection_methods:
        print("\n成功檢測的方法統計:")
        for method, count in sorted(detection_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} 張 ({count/successful_detections*100:.1f}%)")
    
    return successful_detections, failed_detections, detection_methods

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='檢測特定大小的棋盤格')
    parser.add_argument('path', help='圖像路徑或目錄')
    parser.add_argument('--width', type=int, default=10, help='棋盤格寬度 (默認: 10)')
    parser.add_argument('--height', type=int, default=7, help='棋盤格高度 (默認: 7)')
    parser.add_argument('--batch', action='store_true', help='批處理模式')
    parser.add_argument('--pattern', default="*.JPG", help='文件匹配模式 (默認: *.JPG)')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批處理模式
        batch_detect_chessboard(args.path, args.width, args.height, args.pattern)
    else:
        # 單一圖像模式
        (ret, corners, img), (method, w, h) = try_detect_with_params(args.path, args.width, args.height)
        
        if ret:
            # 保存並顯示結果
            output_dir = Path("chessboard_detection_results")
            output_dir.mkdir(exist_ok=True)
            img_name = Path(args.path).stem
            output_path = output_dir / f"{img_name}_detected_{w}x{h}_{method}.jpg"
            cv2.imwrite(str(output_path), img)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"檢測到 {w}x{h} 棋盤格 - 使用方法: {method}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"{img_name}_result.png")
            plt.show()
            
            print(f"\n檢測成功！結果已保存至 {output_path}")
        else:
            print("\n檢測失敗！未能找到指定大小的棋盤格。")