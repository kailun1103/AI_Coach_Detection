import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

# 設定標定時的終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 創建標定板的三維座標點（只需計算一次）
objp = np.zeros((7 * 10, 3), np.float32)
square_size = 80  # 設定標定板方格的實際大小(mm)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square_size

# 創建存儲容器
objpoints = []      # 存儲標定板上點的三維座標
imgpointsLF = []    # 存儲右相機拍攝圖片中檢測到的角點二維座標
imgpointsL = []     # 存儲左相機拍攝圖片中檢測到的角點二維座標

# 確保輸出目錄存在
os.makedirs('output', exist_ok=True)

# 定義處理單張圖像對的函數
def process_image_pair(i):
    t = str(i)
    ChessImaLF = cv2.imread(f'binocular_correction/indoor_0227/45/Indoor_{i}.JPG', 0)
    ChessImaL = cv2.imread(f'binocular_correction/indoor_0227/side/Indoor_{i}.JPG', 0)
    
    if ChessImaLF is None or ChessImaL is None:
        print(f"Image {t} - Could not read one or both images")
        return None
    
    # 在左和左前圖像中查找標定板角點（使用更快的檢測方法）
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    retLF, cornersLF = cv2.findChessboardCorners(ChessImaLF, (10, 7), flags)  
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (10, 7), flags)
    
    print(f"Image {t} - LeftFront: {retLF}, Left: {retL}")
    
    # 如果左和左前圖像都成功檢測到角點
    if (retLF and retL):
        # 對檢測到的角點進行亞像素級精確化
        cv2.cornerSubPix(ChessImaLF, cornersLF, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        
        # 在圖像上繪製檢測到的角點，並保存結果
        cv2.drawChessboardCorners(ChessImaLF, (10,7), cornersLF, retLF)
        cv2.imwrite(f'output/R_{t}.jpg', ChessImaLF)
        cv2.drawChessboardCorners(ChessImaL, (10,7), cornersL, retL)
        cv2.imwrite(f'output/L_{t}.jpg', ChessImaL)
        
        return (objp.copy(), cornersLF, cornersL, ChessImaLF.shape, ChessImaL.shape)
    
    return None

# 使用多線程並行處理圖像對
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_image_pair, range(190)))

# 過濾出有效結果並整理數據
valid_results = [r for r in results if r is not None]
for result in valid_results:
    objp_copy, cornersLF, cornersL, shapeLF, shapeL = result
    objpoints.append(objp_copy)
    imgpointsLF.append(cornersLF)
    imgpointsL.append(cornersL)

# 設置初始相機矩陣估計值（可選，有助於加速收斂）
shapeLF_rev = shapeLF[::-1]
shapeL_rev = shapeL[::-1]

print(f"成功處理的圖像對數量: {len(objpoints)}")

# 使用較少的迭代次數進行相機標定
flags_calib = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST

# 左前相機標定
retLF, mtxLF, distLF, rvecsLF, tvecsLF = cv2.calibrateCamera(
    objpoints, imgpointsLF, shapeLF_rev, None, None, 
    flags=flags_calib, criteria=criteria)

# 獲取左前相機的最優新相機矩陣
hLF, wLF = shapeLF[:2]
OmtxLF, roiLF = cv2.getOptimalNewCameraMatrix(mtxLF, distLF, (wLF, hLF), 1, (wLF, hLF))

# 左相機標定
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, shapeL_rev, None, None,
    flags=flags_calib, criteria=criteria)

# 獲取左相機的最優新相機矩陣
hL, wL = shapeL[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# 雙目相機標定 - 使用合適的標志來加速
stereo_flags = cv2.CALIB_FIX_INTRINSIC  # 使用已校準的相機參數
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsLF,
    mtxL, distL, mtxLF, distLF,
    shapeLF_rev, 
    criteria_stereo, stereo_flags)

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

# Format and print the projection matrices
def format_projection_matrix(matrix):
    formatted = "[\n"
    for row in matrix:
        formatted += f"    [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}, {row[3]:12.6f}],\n"
    formatted += "]"
    return formatted

print("\n==== 投影矩陣計算結果 ====")
print("左相機投影矩陣:")
print(format_projection_matrix(P_left))
print("\n左前相機投影矩陣:")
print(format_projection_matrix(P_leftfront))

# 保存結果到文件
np.savez('calibration_results.npz', 
         MLS=MLS, MRS=MRS, R=R, T=T, 
         P_left=P_left, P_leftfront=P_leftfront,
         dLS=dLS, dRS=dRS)

print("\n校準結果已保存到 calibration_results.npz")