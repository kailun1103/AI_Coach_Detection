import plotly.graph_objects as go
import json

# 通用設定：從檔案載入資料
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 提取座標資料
def extract_coordinates(data, key):
    return [(point['frame'], point[key]['x'], point[key]['y']) 
            for point in data if key in point and point[key]['x'] is not None and point[key]['y'] is not None]

# 取得通用圖表佈局
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

# 繪製 2D 圖表
def create_2d_plot(data, key, color_scale, line_color, title, output_file, name_label):
    fig = go.Figure()
    frames = [frame for frame, _, _ in data]
    x_coords = [x for _, x, _ in data]
    y_coords = [y for _, _, y in data]
    
    # 主資料軌跡
    fig.add_trace(go.Scatter(
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
                x=1.1
            )
        ),
        line=dict(color=line_color, width=2),
        name=name_label
    ))
    
    # 帶有框架編號的標籤
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='text',
        text=[f'F{frame}' for frame in frames],
        textposition='top center',
        textfont=dict(size=8, color=line_color),
        showlegend=False
    ))

    fig.update_layout(get_common_layout(title))
    fig.write_html(output_file)
    fig.show()

# 主程式
if __name__ == "__main__":
    # 載入資料
    trajectory_data = load_data('leftBackhand_45_trajectory.json')
    
    # 提取手腕和網球座標
    wrist_data = extract_coordinates(trajectory_data, 'left_wrist')
    ball_data = extract_coordinates(trajectory_data, 'tennis_ball')

    # 1. 網球軌跡
    create_2d_plot(
        data=ball_data,
        key='tennis_ball',
        color_scale='Plasma',
        line_color='red',
        title='Tennis Ball Trajectory',
        output_file='2d_ball_trajectory.html',
        name_label='Tennis Ball'
    )

    # 2. 手腕軌跡
    create_2d_plot(
        data=wrist_data,
        key='left_wrist',
        color_scale='Viridis',
        line_color='blue',
        title='Wrist Trajectory',
        output_file='2d_wrist_trajectory.html',
        name_label='Wrist'
    )

    # 3. 合併圖
    fig_combined = go.Figure()

    # 手腕資料
    frames_wrist = [frame for frame, _, _ in wrist_data]
    x_coords_wrist = [x for _, x, _ in wrist_data]
    y_coords_wrist = [y for _, _, y in wrist_data]
    fig_combined.add_trace(go.Scatter(
        x=x_coords_wrist,
        y=y_coords_wrist,
        mode='lines+markers',
        marker=dict(
            size=5,
            color=frames_wrist,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Frame (Wrist)',
                x=1.1
            )
        ),
        line=dict(color='blue', width=2),
        name='Wrist'
    ))

    # 網球資料
    frames_ball = [frame for frame, _, _ in ball_data]
    x_coords_ball = [x for _, x, _ in ball_data]
    y_coords_ball = [y for _, _, y in ball_data]
    fig_combined.add_trace(go.Scatter(
        x=x_coords_ball,
        y=y_coords_ball,
        mode='lines+markers',
        marker=dict(
            size=5,
            color=frames_ball,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title='Frame (Ball)',
                x=1.2
            )
        ),
        line=dict(color='red', width=2),
        name='Tennis Ball'
    ))

    fig_combined.update_layout(get_common_layout('Combined Wrist and Tennis Ball Trajectory'))
    fig_combined.write_html('2d_combined_trajectory.html')
    fig_combined.show()
