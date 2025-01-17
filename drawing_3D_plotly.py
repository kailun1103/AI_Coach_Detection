import plotly.graph_objects as go
import json

def create_3d_plots(data_file):
    with open(data_file, 'r') as f:
        trajectory_data = json.load(f)

    total_frames = len(trajectory_data)

    # 定義關節點顏色
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
        ('left_ear', 'left_shoulder'), ('right_ear', 'right_shoulder'),
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

    # 收集所有有效座標
    all_x, all_y, all_z = [], [], []
    ball_x, ball_y, ball_z = [], [], []
    left_wrist_x, left_wrist_y, left_wrist_z = [], [], []
    right_wrist_x, right_wrist_y, right_wrist_z = [], [], []
    
    for frame in trajectory_data:
        for joint in joints:
            if (frame[joint]['x'] is not None and 
                frame[joint]['y'] is not None and 
                frame[joint]['z'] is not None):
                all_x.append(frame[joint]['x'])
                all_y.append(frame[joint]['y'])
                all_z.append(frame[joint]['z'])
                
                if joint == 'tennis_ball':
                    ball_x.append(frame[joint]['x'])
                    ball_y.append(frame[joint]['z'])
                    ball_z.append(frame[joint]['y'])
                elif joint == 'left_wrist':
                    left_wrist_x.append(frame[joint]['x'])
                    left_wrist_y.append(frame[joint]['z'])
                    left_wrist_z.append(frame[joint]['y'])
                elif joint == 'right_wrist':
                    right_wrist_x.append(frame[joint]['x'])
                    right_wrist_y.append(frame[joint]['z'])
                    right_wrist_z.append(frame[joint]['y'])

    if not all_x or not all_y or not all_z:
        raise ValueError("No valid coordinate data found")

    # 創建圖形
    fig = go.Figure()

    # 添加球的軌跡線
    fig.add_trace(go.Scatter3d(
        x=ball_x,
        y=ball_y,
        z=ball_z,
        mode='lines',
        name='Ball Trajectory',
        line=dict(color='red', width=2),
        showlegend=True,
    ))

    # 添加左手腕軌跡線
    fig.add_trace(go.Scatter3d(
        x=left_wrist_x,
        y=left_wrist_y,
        z=left_wrist_z,
        mode='lines',
        name='Left Wrist Trajectory',
        line=dict(color='#800080', width=2),
        showlegend=True,
    ))

    # 添加右手腕軌跡線
    fig.add_trace(go.Scatter3d(
        x=right_wrist_x,
        y=right_wrist_y,
        z=right_wrist_z,
        mode='lines',
        name='Right Wrist Trajectory',
        line=dict(color='#008080', width=2),
        showlegend=True,
    ))

    # 為每一幀創建骨架和關節點
    frames = []
    for frame_idx, frame in enumerate(trajectory_data):
        frame_data = []
        
        # 添加球的完整軌跡（保持可見）
        frame_data.append(go.Scatter3d(
            x=ball_x,
            y=ball_y,
            z=ball_z,
            mode='lines',
            line=dict(color='red', width=2),
            name='Ball Trajectory',
            showlegend=True if frame_idx == 0 else False,
        ))
        
        # 添加左手腕軌跡
        frame_data.append(go.Scatter3d(
            x=left_wrist_x,
            y=left_wrist_y,
            z=left_wrist_z,
            mode='lines',
            line=dict(color='#800080', width=2),
            name='Left Wrist Trajectory',
            showlegend=True if frame_idx == 0 else False,
        ))
        
        # 添加右手腕軌跡
        frame_data.append(go.Scatter3d(
            x=right_wrist_x,
            y=right_wrist_y,
            z=right_wrist_z,
            mode='lines',
            line=dict(color='#008080', width=2),
            name='Right Wrist Trajectory',
            showlegend=True if frame_idx == 0 else False,
        ))
        
        # 添加當前幀的關節點
        for joint_name, color in joints.items():
            if (frame[joint_name]['x'] is not None and 
                frame[joint_name]['y'] is not None and 
                frame[joint_name]['z'] is not None):
                frame_data.append(go.Scatter3d(
                    x=[frame[joint_name]['x']],
                    y=[frame[joint_name]['z']],
                    z=[frame[joint_name]['y']],
                    mode='markers',
                    marker=dict(
                        size=15 if joint_name == 'tennis_ball' else 5,
                        color=color,
                        opacity=0.8
                    ),
                    showlegend=False,
                    hovertemplate=f"{joint_name}<br>" +
                                "X: %{x:.1f}<br>" +
                                "Y: %{z:.1f}<br>" +
                                "Z: %{y:.1f}<br>"
                ))

        # 添加骨架連接
        for start_joint, end_joint in skeleton_connections:
            if (frame[start_joint]['x'] is not None and 
                frame[start_joint]['y'] is not None and 
                frame[start_joint]['z'] is not None and 
                frame[end_joint]['x'] is not None and 
                frame[end_joint]['y'] is not None and 
                frame[end_joint]['z'] is not None):
                frame_data.append(go.Scatter3d(
                    x=[frame[start_joint]['x'], frame[end_joint]['x']],
                    y=[frame[start_joint]['z'], frame[end_joint]['z']],
                    z=[frame[start_joint]['y'], frame[end_joint]['y']],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.8)', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=f'frame_{frame_idx}'
        ))
    
    # 添加初始幀的數據
    for trace in frames[0].data:
        fig.add_trace(trace)

    # 計算坐標範圍
    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    )

    # 更新布局設置
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white',
                range=[min(all_x) - max_range*0.1, max(all_x) + max_range*0.1]
            ),
            yaxis=dict(
                title='Z',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white',
                range=[min(all_z) - max_range*0.1, max(all_z) + max_range*0.1]
            ),
            zaxis=dict(
                title='Y',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white',
                range=[min(all_y) - max_range*0.1, max(all_y) + max_range*0.1]
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0},
                        'mode': 'immediate',
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [[f'frame_{k}'], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(k),
                'method': 'animate'
            } for k in range(len(frames))]
        }],
        width=1000,
        height=800,
        title='3D Body Pose Animation with Wrist Trajectories',
    )

    fig.frames = frames

    output_path = data_file.replace('.json', '.html')

    # 配置顯示設置
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['orbitRotation'],
        'scrollZoom': True,
    }

    # 保存和顯示
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        include_mathjax='cdn',
        config=config,
        auto_play=False
    )
    
    # 顯示圖形
    fig.show(config=config)

if __name__ == "__main__":
    create_3d_plots('temp/junior_3D_trajectory.json')