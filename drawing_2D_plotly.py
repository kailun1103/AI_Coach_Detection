import plotly.graph_objects as go
import json

def create_2d_plots(file_path):
    def load_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def extract_coordinates(data, key):
        return [(point['frame'], point[key]['x'], point[key]['y']) 
                for point in data if key in point and point[key]['x'] is not None and point[key]['y'] is not None]

    trajectory_data = load_data(file_path)
    wrist_data = extract_coordinates(trajectory_data, 'right_wrist')
    ball_data = extract_coordinates(trajectory_data, 'tennis_ball')
    
    def get_common_layout(title):
        return dict(
            title=title,
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            width=1800,
            height=800,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                dtick=100,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                minor=dict(
                    dtick=50,
                    gridwidth=0.5,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    ticks='inside'
                ),
            ),
            showlegend=True,
            margin=dict(r=150)
        )

    def create_trace(coords_data, color_scale, line_color, name_label, colorbar_x):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            marker=dict(
                size=5,
                color=frames,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title=f'Frame ({name_label})',
                    x=colorbar_x
                )
            ),
            line=dict(color=line_color, width=2),
            name=name_label
        )

    def create_frame_labels(coords_data, line_color):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='text',
            text=[f'F{frame}' for frame in frames],
            textposition='top center',
            textfont=dict(size=8, color=line_color),
            showlegend=False
        )

    # Tennis Ball Trajectory
    fig_ball_name = file_path.replace('.json','_2d_ball_trajectory.html')
    fig_ball = go.Figure()
    fig_ball.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.1))
    fig_ball.add_trace(create_frame_labels(ball_data, 'red'))
    fig_ball.update_layout(get_common_layout('Tennis Ball Trajectory'))
    fig_ball.write_html(fig_ball_name)
    # fig_ball.show()

    # Wrist Trajectory
    fig_wrist_name = file_path.replace('.json','_2d_wrist_trajectory.html')
    fig_wrist = go.Figure()
    fig_wrist.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_wrist.add_trace(create_frame_labels(wrist_data, 'blue'))
    fig_wrist.update_layout(get_common_layout('Wrist Trajectory'))
    fig_wrist.write_html(fig_wrist_name)
    fig_wrist.show()

    # Combined Trajectory
    fig_combined = go.Figure()
    fig_combined.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_combined.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.2))
    fig_combined.update_layout(get_common_layout('Combined Wrist and Tennis Ball Trajectory'))
    fig_combined.write_html('2d_combined_trajectory.html')
    fig_combined.show()

if __name__ == "__main__":
    create_2d_plots("凱倫__3_side(2D_trajectory_smoothed).json")