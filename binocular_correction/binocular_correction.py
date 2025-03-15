import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# 設定標定時的終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 創建標定板的三維座標點
objp = np.zeros((7 * 10, 3), np.float32)
square_size = 80  # 設定標定板方格的實際大小(mm)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square_size

# 創建存儲容器
objpoints = []      # 存儲標定板上點的三維座標
imgpointsLF = []    # 存儲右相機拍攝圖片中檢測到的角點二維座標
imgpointsL = []     # 存儲左相機拍攝圖片中檢測到的角點二維座標

# 確保輸出目錄存在
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 檢測統計數據
stats = {
    'total_images': 0,
    'detected_normal': 0,
    'detected_inverted': 0,
    'detected_both': 0,
    'detected_none': 0
}

# 檢測棋盤格函數，嘗試原始和倒置兩種方式
def detect_chessboard(image, pattern_size=(10, 7)):
    """
    嘗試使用原始圖像和倒置圖像檢測棋盤格
    
    Parameters:
    image: 輸入圖像
    pattern_size: 棋盤格大小 (寬, 高)
    
    Returns:
    tuple: (檢測結果, 角點, 檢測方式)
    """
    # 嘗試原始圖像
    ret_orig, corners_orig = cv2.findChessboardCorners(image, pattern_size, None)
    
    # 嘗試倒置圖像
    image_inv = cv2.bitwise_not(image)
    ret_inv, corners_inv = cv2.findChessboardCorners(image_inv, pattern_size, None)
    
    # 決定使用哪種結果
    if ret_orig and ret_inv:
        # 兩種方法都成功，選擇角點數量多的
        if len(corners_orig) >= len(corners_inv):
            return ret_orig, corners_orig, "原始"
        else:
            return ret_inv, corners_inv, "倒置"
    elif ret_orig:
        return ret_orig, corners_orig, "原始"
    elif ret_inv:
        return ret_inv, corners_inv, "倒置"
    else:
        return False, None, "無法檢測"

# 讀取並處理每一張標定圖片
for i in range(59):
    t = str(i)
    stats['total_images'] += 1

    # 讀取圖像
    lf_path = f'binocular_correction/indoor_0315/45/Indoor_{i}.JPG'
    l_path = f'binocular_correction/indoor_0315/side/Indoor_{i}.JPG'
    
    # 檢查文件是否存在
    if not os.path.exists(lf_path) or not os.path.exists(l_path):
        print(f"警告: 圖像 {i} 不存在，跳過")
        continue
    
    ChessImaLF = cv2.imread(lf_path, 0)
    ChessImaL = cv2.imread(l_path, 0)
    
    if ChessImaLF is None or ChessImaL is None:
        print(f"警告: 圖像 {i} 讀取失敗，跳過")
        continue

    # 在左前和左圖像中查找標定板角點，嘗試原始和倒置方法
    retLF, cornersLF, methodLF = detect_chessboard(ChessImaLF)
    retL, cornersL, methodL = detect_chessboard(ChessImaL)
    
    print(f"Image {t} - LeftFront: {retLF} ({methodLF}), Left: {retL} ({methodL})")
    
    # 更新統計信息
    if retLF and not retL:
        stats['detected_normal'] += 1
    elif not retLF and retL:
        stats['detected_inverted'] += 1
    elif retLF and retL:
        stats['detected_both'] += 1
    else:
        stats['detected_none'] += 1
    
    # 如果左和左前圖像都成功檢測到角點
    if retLF and retL:
        objpoints.append(objp)  # 添加三維座標點
        
        # 對檢測到的角點進行亞像素級精確化
        cv2.cornerSubPix(ChessImaLF, cornersLF, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        
        # 保存精確化後的角點座標
        imgpointsLF.append(cornersLF)
        imgpointsL.append(cornersL)
        
        # 創建彩色圖像用於繪製
        ChessImaLF_color = cv2.cvtColor(ChessImaLF, cv2.COLOR_GRAY2BGR)
        ChessImaL_color = cv2.cvtColor(ChessImaL, cv2.COLOR_GRAY2BGR)
        
        # 在圖像上繪製檢測到的角點，並保存結果
        cv2.drawChessboardCorners(ChessImaLF_color, (10, 7), cornersLF, retLF)
        cv2.imwrite(f'{output_dir}/R_{t}_{methodLF}.jpg', ChessImaLF_color)
        cv2.drawChessboardCorners(ChessImaL_color, (10, 7), cornersL, retL)
        cv2.imwrite(f'{output_dir}/L_{t}_{methodL}.jpg', ChessImaL_color)

# 打印檢測統計信息
print("\n==== 棋盤格檢測統計 ====")
print(f"總圖像數量: {stats['total_images']}")
print(f"兩個相機都成功檢測: {len(objpoints)} ({len(objpoints)/stats['total_images']*100:.1f}%)")
print(f"僅左前相機檢測成功: {stats['detected_normal']}")
print(f"僅左相機檢測成功: {stats['detected_inverted']}")
print(f"兩個相機都未檢測成功: {stats['detected_none']}")

# 如果沒有足夠的成功檢測，就結束程式
if len(objpoints) < 3:
    print("錯誤: 成功檢測的棋盤格數量不足，無法進行相機標定。至少需要 3 個棋盤格圖像。")
    exit()

# 獲取最後成功檢測的圖像尺寸，用於相機標定
img_size_LF = ChessImaLF.shape[::-1]
img_size_L = ChessImaL.shape[::-1]

# 左前相機標定
retLF, mtxLF, distLF, rvecsLF, tvecsLF = cv2.calibrateCamera(objpoints, imgpointsLF, img_size_LF, None, None)

# 獲取左前相機的最優新相機矩陣
hLF, wLF = ChessImaLF.shape[:2]
OmtxLF, roiLF = cv2.getOptimalNewCameraMatrix(mtxLF, distLF, (wLF, hLF), 1, (wLF, hLF))

# 左相機標定
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, img_size_L, None, None)

# 獲取左相機的最優新相機矩陣
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# 雙目相機標定
flags = 0
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsLF,
                                                           mtxL, distL, mtxLF, distLF,
                                                           img_size_LF, 
                                                           criteria_stereo, flags)

# 定義矩陣乘法函數
def create_RT_matrix(R, T):
    """建立4x4的RT矩陣"""
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = T.flatten()
    return RT

def project_matrix(camera_matrix, RT):
    """計算投影矩陣"""
    # 將相機矩陣擴展為3x4
    P = np.zeros((3, 4))
    P[:3, :3] = camera_matrix
    
    # 計算投影矩陣
    return np.dot(camera_matrix, RT[:3, :])

# 建立左相機的RT矩陣（單位矩陣）
RT_left = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
])

# 建立左前相機的RT矩陣
RT_leftfront = create_RT_matrix(R, T)

# 計算投影矩陣
P_left = project_matrix(MLS, RT_left)
P_leftfront = project_matrix(MRS, RT_leftfront)

# 在打印結果之前，加入以下設置：
np.set_printoptions(suppress=True,  # 禁用科學記號
                   precision=6,      # 設置小數位數
                   floatmode='fixed', # 使用固定小數點格式
                   threshold=np.inf)  # 顯示完整數組

print("\n==== 標定結果 ====")
print("左相機內參矩陣 MLS:")
print(MLS)
print("\n左前相機內參矩陣 MRS:")
print(MRS)
print("\n旋轉矩陣 R:")
print(R)
print("\n平移向量 T:")
print(T)

print("\n==== 投影矩陣計算結果 ====")
print("左相機投影矩陣:")
print(P_left)
print("\n左前相機投影矩陣:")
print(P_leftfront)

# Format and print the projection matrices
def format_projection_matrix(matrix):
    formatted = "[\n"
    for row in matrix:
        formatted += f"    [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}, {row[3]:12.6f}],\n"
    formatted += "]"
    return formatted

print("\n==== 投影矩陣計算結果（格式化） ====")
print("左相機投影矩陣:")
print(format_projection_matrix(P_left))
print("\n左前相機投影矩陣:")
print(format_projection_matrix(P_leftfront))

# 計算重投影誤差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsLF[i], tvecsLF[i], mtxLF, distLF)
    error = cv2.norm(imgpointsLF[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("\n左前相機重投影誤差: {}".format(mean_error / len(objpoints)))

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    error = cv2.norm(imgpointsL[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("左相機重投影誤差: {}".format(mean_error / len(objpoints)))

# 保存標定結果
calibration_result = {
    "左相機內參矩陣": MLS.tolist(),
    "左相機畸變係數": dLS.tolist(),
    "左前相機內參矩陣": MRS.tolist(),
    "左前相機畸變係數": dRS.tolist(),
    "旋轉矩陣": R.tolist(),
    "平移向量": T.tolist(),
    "左相機投影矩陣": P_left.tolist(),
    "左前相機投影矩陣": P_leftfront.tolist()
}

import json
with open(f'{output_dir}/calibration_results.json', 'w') as f:
    json.dump(calibration_result, f, indent=4)

print(f"\n標定結果已保存至 {output_dir}/calibration_results.json")