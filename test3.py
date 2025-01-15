import plotly.graph_objects as go
import json

def create_2d_plots(data_file):
    """
    創建2D姿態圖，可以顯示每一幀的骨架和關節點位置
    
    Parameters:
    data_file (str): JSON檔案路徑，包含關節點和網球的2D座標
    """
    with open(data_file, 'r') as f:
        trajectory_data = json.load(f)

    # 獲取總幀數
    total_frames = len(trajectory_data)

    # 定義要追蹤的關節點和它們的顏色
    joints = {
        'tennis_ball': '#ff0000',  # 紅色
        'nose': '#00ff00',         # 綠色
        'left_eye': '#0000ff',     # 藍色
        'right_eye': '#00ffff',    # 青色
        'left_ear': '#ff00ff',     # 洋紅色
        'right_ear': '#ffff00',    # 黃色
        'left_shoulder': '#800000', # 暗紅色
        'right_shoulder': '#008000',# 暗綠色
        'left_elbow': '#000080',   # 暗藍色
        'right_elbow': '#808000',  # 橄欖色
        'left_wrist': '#800080',   # 紫色
        'right_wrist': '#008080',  # 藍綠色
        'left_hip': '#ff8000',     # 橙色
        'right_hip': '#0080ff',    # 淺藍色
        'left_knee': '#ff0080',    # 粉紅色
        'right_knee': '#80ff00',   # 淺綠色
        'left_ankle': '#8000ff',   # 紫羅蘭色
        'right_ankle': '#00ff80'   # 青綠色
    }

    # 定義骨架連接
    skeleton_connections = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]

    # 找出所有有效座標的範圍
    all_x = []
    all_y = []
    for frame in trajectory_data:
        for joint in joints:
            if (frame[joint]['x'] is not None and 
                frame[joint]['y'] is not None and 
                frame[joint]['x'] != 0 and 
                frame[joint]['y'] != 0):
                all_x.append(frame[joint]['x'])
                all_y.append(frame[joint]['y'])

    # 確保有有效數據
    if not all_x or not all_y:
        raise ValueError("No valid coordinate data found")

    # 創建圖形
    fig = go.Figure()

    # 為每一幀創建關節點和骨架的trace（初始都設為隱藏）
    for frame_idx, frame in enumerate(trajectory_data):
        # 添加關節點
        for joint_name, color in joints.items():
            if (frame[joint_name]['x'] is not None and 
                frame[joint_name]['y'] is not None and 
                frame[joint_name]['x'] != 0 and 
                frame[joint_name]['y'] != 0):
                fig.add_trace(go.Scatter(
                    x=[frame[joint_name]['x']],
                    y=[frame[joint_name]['y']],
                    mode='markers',
                    name=f"{joint_name}_{frame_idx}",
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.8
                    ),
                    showlegend=False,
                    visible=frame_idx == 0
                ))
        
        # 添加骨架
        for start_joint, end_joint in skeleton_connections:
            if (frame[start_joint]['x'] is not None and 
                frame[start_joint]['y'] is not None and 
                frame[end_joint]['x'] is not None and 
                frame[end_joint]['y'] is not None and
                frame[start_joint]['x'] != 0 and 
                frame[start_joint]['y'] != 0 and
                frame[end_joint]['x'] != 0 and 
                frame[end_joint]['y'] != 0):
                fig.add_trace(go.Scatter(
                    x=[frame[start_joint]['x'], frame[end_joint]['x']],
                    y=[frame[start_joint]['y'], frame[end_joint]['y']],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.8)', width=2),
                    showlegend=False,
                    visible=frame_idx == 0,
                    hoverinfo='skip'
                ))

    # 計算每一幀的trace數量
    traces_per_frame = len(joints) + len(skeleton_connections)
    
    # 創建幀動畫
    frames = []
    for i in range(total_frames):
        frame_traces = []
        for trace_idx in range(len(fig.data)):
            if i == trace_idx // traces_per_frame:
                frame_traces.append(dict(visible=True))
            else:
                frame_traces.append(dict(visible=False))
        frames.append(dict(
            data=frame_traces,
            name=f'frame_{i}'
        ))
    
    # 創建滑桿的步驟
    steps = []
    for i in range(total_frames):
        visibility = [False] * len(fig.data)
        start_idx = i * traces_per_frame
        end_idx = start_idx + traces_per_frame
        for j in range(start_idx, end_idx):
            if j < len(visibility):
                visibility[j] = True
        step = dict(
            args=[{"visible": visibility}],
            label=f"{i:02d}",
            method="update"
        )
        steps.append(step)

    # 計算顯示範圍
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    aspect_ratio = x_range / y_range
    
    if aspect_ratio > 1.5:
        width = 1200
        height = int(width / aspect_ratio)
    else:
        height = 800
        width = int(height * aspect_ratio)

    # 更新布局
    fig.update_layout(
        xaxis=dict(
            title='X Position',
            gridcolor='lightgray',
            showgrid=True,
            showline=True,
            zeroline=False,
            range=[min(all_x) - x_range*0.1, max(all_x) + x_range*0.1]
        ),
        yaxis=dict(
            title='Y Position',
            gridcolor='lightgray',
            showgrid=True,
            showline=True,
            zeroline=False,
            range=[max(all_y) + y_range*0.1, min(all_y) - y_range*0.1],
            scaleanchor="x",
            scaleratio=1
        ),
        width=width,
        height=height + 200,
        title=dict(
            text='Body Pose Frame View',
            y=0.95
        ),
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=100)
    )

    # 添加播放按鈕和滑桿
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=100, redraw=True),
                    fromcurrent=True,
                    mode="immediate"
                )]
            )],
            x=0.1,
            y=-0.05,
            xanchor="right",
            yanchor="top",
            pad=dict(r=10, t=10)
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="Frame: ",
                suffix=f" / {total_frames-1}",
                visible=True,
                xanchor="right",
                font=dict(size=14, color="#666666")
            ),
            pad=dict(t=50, b=10),
            steps=steps,
            x=0.1,
            y=-0.1,
            xanchor="left",
            yanchor="top",
            len=0.8,
            bgcolor="#f0f0f0",
            bordercolor="#666666",
            borderwidth=1,
            ticklen=5,
            minorticklen=2,
            tickwidth=1,
            tickcolor="#666666",
            font=dict(size=10)
        )]
    )

    # 設置frames
    fig.frames = frames

    # 保存和顯示
    fig.write_html(
        "body_pose_frame_view.html",
        include_plotlyjs=True,
        full_html=True,
        include_mathjax=False
    )
    
    fig.show()

if __name__ == "__main__":
    create_2d_plots('leftBackhand_45_trajectory.json')