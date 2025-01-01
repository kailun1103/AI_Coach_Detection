import plotly.graph_objects as go
import json

# Read data
with open('leftBackhand_45_trajectory.json', 'r') as f:
    trajectory_data = json.load(f)

# Extract coordinates
wrist_data = [(point['frame'], point['left_wrist']['x'], point['left_wrist']['y']) 
              for point in trajectory_data]

ball_data = [(point['frame'], point['tennis_ball']['x'], point['tennis_ball']['y']) 
             for point in trajectory_data 
             if point['tennis_ball']['x'] is not None and point['tennis_ball']['y'] is not None]

# Combined trajectory plot with wider width and more detailed x-axis
fig_combined = go.Figure()

# Add wrist trajectory
fig_combined.add_trace(go.Scatter(
    x=[x for _, x, _ in wrist_data],
    y=[y for _, _, y in wrist_data],
    mode='lines+markers',
    marker=dict(
        size=5,
        color=[frame for frame, _, _ in wrist_data],
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

fig_combined.add_trace(go.Scatter(
    x=[x for _, x, _ in wrist_data],
    y=[y for _, _, y in wrist_data],
    mode='text',
    text=[f'F{frame}' for frame, _, _ in wrist_data],
    textposition='top center',
    textfont=dict(size=8, color='blue'),
    name='Wrist Frame Labels',
    showlegend=False
))

# Add tennis ball trajectory
fig_combined.add_trace(go.Scatter(
    x=[x for _, x, _ in ball_data],
    y=[y for _, _, y in ball_data],
    mode='lines+markers',
    marker=dict(
        size=5,
        color=[frame for frame, _, _ in ball_data],
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

fig_combined.add_trace(go.Scatter(
    x=[x for _, x, _ in ball_data],
    y=[y for _, _, y in ball_data],
    mode='text',
    text=[f'F{frame}' for frame, _, _ in ball_data],
    textposition='bottom center',
    textfont=dict(size=8, color='red'),
    name='Ball Frame Labels',
    showlegend=False
))

# Update layout with wider width and more detailed x-axis
fig_combined.update_layout(
    title='Combined Wrist and Tennis Ball Trajectory',
    xaxis_title='X coordinate',
    yaxis_title='Y coordinate',
    width=1800,  # Increased width from 1200 to 1800
    height=800,
    yaxis=dict(autorange="reversed"),
    xaxis=dict(
        dtick=100,  # Set x-axis tick interval to 100
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        minor=dict(
            dtick=50,  # Set minor tick interval to 50
            gridwidth=0.5,
            gridcolor='rgba(128, 128, 128, 0.1)',
            ticks='inside'
        ),
    ),
    showlegend=True,
    margin=dict(r=150)
)

# Save the plot to HTML
fig_combined.write_html("2d__trajectory_wide.html")

# Display the plot
fig_combined.show()