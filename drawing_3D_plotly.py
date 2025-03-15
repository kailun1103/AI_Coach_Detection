import plotly.graph_objects as go
import json

def create_3d_plots(data_file):
    with open(data_file, 'r') as f:
        trajectory_data = json.load(f)

    total_frames = len(trajectory_data)

    joints = {
        'tennis_ball': '#ff0000',
        'nose': '#00ff00',
        'left_eye': '#0000ff',
        'right_eye': '#00ffff',
        'left_ear': '#ff00ff',
        'right_ear': '#ffff00',
        'left_shoulder': '#800000',
        'right_shoulder': '#008000',
        'left_elbow': '#000080',
        'right_elbow': '#808000',
        'left_wrist': '#800080',
        'right_wrist': '#008080',
        'left_hip': '#ff8000',
        'right_hip': '#0080ff',
        'left_knee': '#ff0080',
        'right_knee': '#80ff00',
        'left_ankle': '#8000ff',
        'right_ankle': '#00ff80'
    }

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

    all_x, all_y, all_z = [], [], []
    ball_x, ball_y, ball_z = [], [], []
    left_wrist_x, left_wrist_y, left_wrist_z = [], [], []
    right_wrist_x, right_wrist_y, right_wrist_z = [], [], []
    frame_labels = []
    
    for frame_idx, frame in enumerate(trajectory_data):
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
                    frame_labels.append(f'F{frame_idx}')
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

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=ball_x,
        y=ball_y,
        z=ball_z,
        mode='lines+markers+text',
        name='Ball Trajectory',
        line=dict(color='red', width=2),
        text=frame_labels,
        textposition='top center',
        textfont=dict(size=8),
        showlegend=True,
    ))

    fig.add_trace(go.Scatter3d(
        x=left_wrist_x,
        y=left_wrist_y,
        z=left_wrist_z,
        mode='lines',
        name='Left Wrist Trajectory',
        line=dict(color='#800080', width=2),
        showlegend=True,
    ))

    fig.add_trace(go.Scatter3d(
        x=right_wrist_x,
        y=right_wrist_y,
        z=right_wrist_z,
        mode='lines',
        name='Right Wrist Trajectory',
        line=dict(color='#008080', width=2),
        showlegend=True,
    ))

    frames = []
    for frame_idx, frame in enumerate(trajectory_data):
        frame_data = []
        
        frame_data.append(go.Scatter3d(
            x=ball_x,
            y=ball_y,
            z=ball_z,
            mode='lines+markers+text',
            line=dict(color='red', width=2),
            text=frame_labels,
            textposition='top center',
            textfont=dict(size=8),
            name='Ball Trajectory',
            showlegend=True if frame_idx == 0 else False,
        ))
        
        frame_data.append(go.Scatter3d(
            x=left_wrist_x,
            y=left_wrist_y,
            z=left_wrist_z,
            mode='lines',
            line=dict(color='#800080', width=2),
            name='Left Wrist Trajectory',
            showlegend=True if frame_idx == 0 else False,
        ))
        
        frame_data.append(go.Scatter3d(
            x=right_wrist_x,
            y=right_wrist_y,
            z=right_wrist_z,
            mode='lines',
            line=dict(color='#008080', width=2),
            name='Right Wrist Trajectory',
            showlegend=True if frame_idx == 0 else False,
        ))
        
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
    
    for trace in frames[0].data:
        fig.add_trace(trace)

    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    )

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
        title='3D Body Pose Animation with Frame Labels',
    )

    fig.frames = frames

    output_path = data_file.replace('.json', '.html')

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['orbitRotation'],
        'scrollZoom': True,
    }

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        include_mathjax='cdn',
        config=config,
        auto_play=False
    )
    
    # fig.show(config=config)

if __name__ == "__main__":
    create_3d_plots("凱倫__1(3D_trajectory_smoothed)_only_swing.json")