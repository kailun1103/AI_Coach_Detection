import plotly.graph_objects as go
import numpy as np
import json

def create_3d_plots(data_file):
    with open(data_file, 'r') as f:
        trajectory_data = json.load(f)
    
    # Extract coordinates
    X = [point['left_wrist']['x'] for point in trajectory_data]
    Y = [point['left_wrist']['y'] for point in trajectory_data]
    Z = [point['left_wrist']['z'] for point in trajectory_data]
    
    ball_data = [(point['tennis_ball']['x'], point['tennis_ball']['y'], point['tennis_ball']['z']) 
                 for point in trajectory_data 
                 if point['tennis_ball']['x'] is not None]
    
    if ball_data:
        X_ball, Y_ball, Z_ball = zip(*ball_data)
        
        # Calculate ranges including both wrist and ball data
        x_min = min(min(X), min(X_ball))
        x_max = max(max(X), max(X_ball))
        y_min = min(min(Y), min(Y_ball))
        y_max = max(max(Y), max(Y_ball))
        z_min = min(min(Z), min(Z_ball))
        z_max = max(max(Z), max(Z_ball))
    else:
        x_min, x_max = min(X), max(X)
        y_min, y_max = min(Y), max(Y)
        z_min, z_max = min(Z), max(Z)
    
    # Add padding to ranges
    padding_x = (x_max - x_min) * 0.2
    padding_y = (y_max - y_min) * 0.2
    padding_z = (z_max - z_min) * 0.2
    
    # Create modified layout with wider dimensions
    layout = dict(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
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
    
    # Combined plot with modified dimensions
    fig_combined = go.Figure()
    
    # Add wrist trajectory
    fig_combined.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='lines+markers',
        marker=dict(
            size=5,
            color=list(range(len(X))),
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Frame')
        ),
        line=dict(color='darkblue', width=2),
        name='Wrist'
    ))
    
    # Add ball trajectory
    if ball_data:
        fig_combined.add_trace(go.Scatter3d(
            x=X_ball, y=Y_ball, z=Z_ball,
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
    
    fig_combined.update_layout(title='Combined 3D Trajectories', **layout)
    
    # Save and display the plot
    fig_combined.write_html("combined_trajectories_3d_wide.html")
    fig_combined.show()

create_3d_plots('leftBackhand_3D_trajectory_smoothed.json')