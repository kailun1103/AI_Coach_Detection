import plotly.graph_objects as go
import numpy as np
import json

def create_3d_plots(data_file):
    with open(data_file, 'r') as f:
        trajectory_data = json.load(f)
    
    # 只提取網球座標，並調換 Y 和 Z 座標
    ball_data = [(point['tennis_ball']['x'], point['tennis_ball']['z'], point['tennis_ball']['y']) 
                 for point in trajectory_data 
                 if point['tennis_ball']['x'] is not None]
    
    if ball_data:
        X_ball, Y_ball, Z_ball = zip(*ball_data)
        
        # 計算網球軌跡的範圍
        x_min, x_max = min(X_ball), max(X_ball)
        y_min, y_max = min(Y_ball), max(Y_ball)
        z_min, z_max = min(Z_ball), max(Z_ball)
        
        # 添加邊界間距
        padding_x = (x_max - x_min) * 0.2
        padding_y = (y_max - y_min) * 0.2
        padding_z = (z_max - z_min) * 0.2
        
        # 創建佈局
        layout = dict(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Z',  # 改為 Z
                zaxis_title='Y',  # 改為 Y
                camera=dict(
                    eye=dict(x=2.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                xaxis=dict(
                    range=[x_min - padding_x, x_max + padding_x],
                    nticks=20,
                ),
                yaxis=dict(
                    range=[y_min - padding_y, y_max + padding_y],
                    nticks=15,
                ),
                zaxis=dict(
                    range=[z_min - padding_z, z_max + padding_z],
                    nticks=15,
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1, z=1)
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        # 創建圖形
        fig = go.Figure()
        
        # 添加網球軌跡
        fig.add_trace(go.Scatter3d(
            x=X_ball, y=Y_ball, z=Z_ball,  # Y 和 Z 已經在資料收集時調換
            mode='lines+markers',
            marker=dict(
                size=5,
                color=list(range(len(X_ball))),
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title='Frame')
            ),
            line=dict(color='red', width=2),
            name='Tennis Ball'
        ))
        
        fig.update_layout(title='Tennis Ball 3D Trajectory', **layout)
        
        # 儲存並顯示圖形
        fig.write_html("ball_trajectory_3d.html")
        fig.show()
    else:
        print("No valid ball trajectory data found")

# 使用更新後的 JSON 文件名稱
create_3d_plots('3D_ball_trajectory.json')