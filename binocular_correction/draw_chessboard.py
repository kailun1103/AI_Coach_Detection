import numpy as np
import cv2

def create_ultra_high_quality_chessboard(rows=8, cols=11, square_size=500, anti_aliasing=True):
    """
    創建超高解析度棋盤格圖案（BMP格式）
    
    Parameters:
    rows (int): 棋盤格的行數
    cols (int): 棋盤格的列數
    square_size (int): 每個方格的像素大小
    anti_aliasing (bool): 是否啟用抗鋸齒
    
    Returns:
    numpy.ndarray: 超高解析度棋盤格圖像
    """
    # 使用更高的縮放因子來提升抗鋸齒效果
    scale = 4 if anti_aliasing else 1  # 提高到4倍
    working_size = square_size * scale
    
    # 計算完整圖像的大小
    width = cols * working_size
    height = rows * working_size
    
    # 創建空白圖像（使用float64以獲得更高的精度）
    chessboard = np.zeros((height, width), dtype=np.float64)
    
    # 填充棋盤格
    for i in range(rows):
        for j in range(cols):
            y = i * working_size
            x = j * working_size
            
            if (i + j) % 2 == 0:
                chessboard[y:y+working_size, x:x+working_size] = 255.0
    
    # 使用更高品質的抗鋸齒處理
    if anti_aliasing:
        target_size = (cols * square_size, rows * square_size)
        # 使用多步驟縮小來獲得更好的品質
        current_size = (width, height)
        while current_size[0] > target_size[0] * 1.5:
            current_size = (current_size[0] // 2, current_size[1] // 2)
            chessboard = cv2.resize(chessboard, current_size, 
                                  interpolation=cv2.INTER_LANCZOS4)
        
        # 最後一步縮放到目標大小
        chessboard = cv2.resize(chessboard, target_size, 
                              interpolation=cv2.INTER_LANCZOS4)
    
    # 確保邊緣清晰
    chessboard = np.clip(chessboard, 0, 255)
    
    # 銳化處理以提高邊緣清晰度
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]], dtype=np.float64)
    chessboard = cv2.filter2D(chessboard, -1, kernel)
    
    # 最終轉換為8位格式
    chessboard = np.clip(chessboard, 0, 255).astype(np.uint8)
    
    return chessboard

if __name__ == "__main__":
    # 創建超高解析度棋盤格
    chessboard = create_ultra_high_quality_chessboard(
        rows=8,
        cols=11,
        square_size=500,  # 大幅增加方格大小到500像素
        anti_aliasing=True
    )
    
    # 儲存為BMP格式
    cv2.imwrite('ultra_high_quality_chessboard.bmp', chessboard)
    
    # 縮小顯示圖像（僅用於預覽）
    display_size = (1200, 800)  # 適合顯示器的大小
    display_img = cv2.resize(chessboard, display_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Ultra High Quality Chessboard (Preview)', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()