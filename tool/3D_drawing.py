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

    # Create base layout
    def get_layout(title):
        return dict(
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
            showlegend=True,
            title=title
        )

    # Create wrist trajectory plot
    fig_wrist = go.Figure()
    frame_numbers = list(range(len(X)))
    fig_wrist.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='lines+markers+text',
        marker=dict(
            size=5,
            color=frame_numbers,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Frame')
        ),
        text=frame_numbers,
        textposition='top center',
        textfont=dict(size=10),
        line=dict(color='darkblue', width=2),
        name='Wrist'
    ))
    fig_wrist.update_layout(**get_layout('Wrist Trajectory'))
    fig_wrist.write_html("wrist_trajectory_3d.html")
    fig_wrist.show()

    # Create ball trajectory plot
    if ball_data:
        fig_ball = go.Figure()
        ball_frame_numbers = list(range(len(X_ball)))
        fig_ball.add_trace(go.Scatter3d(
            x=X_ball, y=Y_ball, z=Z_ball,
            mode='lines+markers+text',
            marker=dict(
                size=5,
                color=ball_frame_numbers,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title='Frame')
            ),
            text=ball_frame_numbers,
            textposition='top center',
            textfont=dict(size=10),
            line=dict(color='red', width=2),
            name='Tennis Ball'
        ))
        fig_ball.update_layout(**get_layout('Ball Trajectory'))
        fig_ball.write_html("ball_trajectory_3d.html")
        fig_ball.show()

    # Create combined trajectory plot
    fig_combined = go.Figure()
    
    # Add wrist trajectory
    fig_combined.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='lines+markers+text',
        marker=dict(
            size=5,
            color=frame_numbers,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Frame')
        ),
        text=frame_numbers,
        textposition='top center',
        textfont=dict(size=10),
        line=dict(color='darkblue', width=2),
        name='Wrist'
    ))

    # Add ball trajectory
    if ball_data:
        fig_combined.add_trace(go.Scatter3d(
            x=X_ball, y=Y_ball, z=Z_ball,
            mode='lines+markers+text',
            marker=dict(
                size=5,
                color=ball_frame_numbers,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title='Frame')
            ),
            text=ball_frame_numbers,
            textposition='top center',
            textfont=dict(size=10),
            line=dict(color='red', width=2),
            name='Tennis Ball'
        ))
    
    fig_combined.update_layout(**get_layout('Combined Trajectories'))
    fig_combined.write_html("combined_trajectories_3d.html")
    fig_combined.show()

if __name__ == "__main__":
    create_3d_plots('leftBackhand_3D_trajectory_smoothed_preserved.json')