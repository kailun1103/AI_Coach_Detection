import plotly.graph_objects as go
import numpy as np
import json

def create_3d_plots(data_file):
    """
    創建具有互動範圍控制的3D軌跡圖
    
    Parameters:
    data_file (str): JSON檔案路徑
    """
    with open(data_file, 'r') as f:
        trajectory_data = json.load(f)
    
    # 提取座標
    wrist_data = {
        'x': [point['left_wrist']['x'] for point in trajectory_data],
        'y': [point['left_wrist']['y'] for point in trajectory_data],
        'z': [point['left_wrist']['z'] for point in trajectory_data]
    }
    
    # 提取網球數據
    ball_positions = [(point['tennis_ball']['x'], point['tennis_ball']['y'], point['tennis_ball']['z']) 
                     for point in trajectory_data 
                     if point['tennis_ball']['x'] is not None]
    
    ball_data = {
        'x': [pos[0] for pos in ball_positions],
        'y': [pos[1] for pos in ball_positions],
        'z': [pos[2] for pos in ball_positions]
    }

    total_frames = len(wrist_data['x'])
    
    # 創建圖形
    fig = go.Figure()

    # 添加初始軌跡
    fig.add_trace(go.Scatter3d(
        x=wrist_data['x'],
        y=wrist_data['y'],
        z=wrist_data['z'],
        mode='lines+markers',
        name='Wrist',
        marker=dict(
            size=5,
            color=list(range(total_frames)),
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
            colorbar=dict(title='Frame')
        ),
        line=dict(color='darkblue', width=2)
    ))

    if ball_data['x']:
        fig.add_trace(go.Scatter3d(
            x=ball_data['x'],
            y=ball_data['y'],
            z=ball_data['z'],
            mode='lines+markers',
            name='Ball',
            marker=dict(
                size=5,
                color=list(range(len(ball_data['x']))),
                colorscale='Plasma',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='Frame', x=1.1)
            ),
            line=dict(color='red', width=2)
        ))

    # 計算坐標範圍
    x_min = min(min(wrist_data['x']), min(ball_data['x']) if ball_data['x'] else float('inf'))
    x_max = max(max(wrist_data['x']), max(ball_data['x']) if ball_data['x'] else float('-inf'))
    y_min = min(min(wrist_data['y']), min(ball_data['y']) if ball_data['y'] else float('inf'))
    y_max = max(max(wrist_data['y']), max(ball_data['y']) if ball_data['y'] else float('-inf'))
    z_min = min(min(wrist_data['z']), min(ball_data['z']) if ball_data['z'] else float('inf'))
    z_max = max(max(wrist_data['z']), max(ball_data['z']) if ball_data['z'] else float('-inf'))

    padding_x = (x_max - x_min) * 0.2
    padding_y = (y_max - y_min) * 0.2
    padding_z = (z_max - z_min) * 0.2

    # 創建起始幀數滑桿的步驟
    start_steps = []
    for i in range(0, total_frames, max(1, total_frames//50)):
        step = dict(
            method="update",
            args=[{"x": [wrist_data['x'][i:], ball_data['x'][i:] if ball_data['x'] else []],
                  "y": [wrist_data['y'][i:], ball_data['y'][i:] if ball_data['y'] else []],
                  "z": [wrist_data['z'][i:], ball_data['z'][i:] if ball_data['z'] else []]
                 }],
            label=f"Frame {i}"
        )
        start_steps.append(step)

    # 創建結束幀數滑桿的步驟
    end_steps = []
    for i in range(0, total_frames, max(1, total_frames//50)):
        step = dict(
            method="update",
            args=[{"x": [wrist_data['x'][:i+1], ball_data['x'][:i+1] if ball_data['x'] else []],
                  "y": [wrist_data['y'][:i+1], ball_data['y'][:i+1] if ball_data['y'] else []],
                  "z": [wrist_data['z'][:i+1], ball_data['z'][:i+1] if ball_data['z'] else []]
                 }],
            label=f"Frame {i}"
        )
        end_steps.append(step)

    # 更新布局
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=2.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(range=[x_min - padding_x, x_max + padding_x]),
            yaxis=dict(range=[y_min - padding_y, y_max + padding_y]),
            zaxis=dict(range=[z_min - padding_z, z_max + padding_z]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)
        ),
        width=1200,
        height=1100,  # 增加整體高度
        title=dict(
            text='3D Trajectories',
            y=0.95
        ),
        margin=dict(t=100, b=200),  # 增加底部邊距
        showlegend=True,
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True, True]},
                           {"title": "Both Trajectories"}]),
                dict(label="Wrist Only",
                     method="update",
                     args=[{"visible": [True, False]},
                           {"title": "Wrist Trajectory"}]),
                dict(label="Ball Only",
                     method="update",
                     args=[{"visible": [False, True]},
                           {"title": "Ball Trajectory"}])
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            x=0.1,
            y=1.1,
        )],
        sliders=[
            # 起始幀數滑桿
            dict(
                active=0,
                currentvalue={"prefix": "Start Frame: "},
                pad={"t": 120, "b": 100},  # 增加上下間距
                steps=start_steps,
                yanchor="top",
                y=-0.15,  # 調整位置
                len=0.9,
                x=0.1
            ),
            # 結束幀數滑桿
            dict(
                active=len(end_steps)-1,
                currentvalue={"prefix": "End Frame: "},
                pad={"t": 100, "b": 120},  # 增加上下間距
                steps=end_steps,
                yanchor="top",
                y=-0.45,  # 調整位置
                len=0.9,
                x=0.1
            )
        ]
    )

    # 保存和顯示
    fig.write_html(
        "interactive_trajectories.html",
        include_plotlyjs=True,
        full_html=True,
        include_mathjax=False
    )
    
    fig.show()

if __name__ == "__main__":
    create_3d_plots('leftBackhand_3D_trajectory_corrected.json')