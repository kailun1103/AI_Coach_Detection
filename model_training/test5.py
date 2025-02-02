import plotly.graph_objects as go
import json

# Read data
with open('model_training/leftFront_trajectory.json', 'r') as f:
    trajectory_data = json.load(f)

# Extract only ball coordinates
ball_data = [(point['frame'], point['tennis_ball']['x'], point['tennis_ball']['y'])
           for point in trajectory_data
           if point['tennis_ball']['x'] is not None and point['tennis_ball']['y'] is not None]

# Create figure
fig = go.Figure()

# Add tennis ball trajectory
fig.add_trace(go.Scatter(
    x=[x for _, x, _ in ball_data],
    y=[y for _, _, y in ball_data],
    mode='lines+markers',
    marker=dict(
        size=5,
        color=[frame for frame, _, _ in ball_data],
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(
            title='Frame'
        )
    ),
    line=dict(color='red', width=2),
    name='Tennis Ball'
))

# Add frame labels
fig.add_trace(go.Scatter(
    x=[x for _, x, _ in ball_data],
    y=[y for _, _, y in ball_data],
    mode='text',
    text=[f'F{frame}' for frame, _, _ in ball_data],
    textposition='bottom center',
    textfont=dict(size=8, color='red'),
    name='Frame Labels',
    showlegend=False
))

# Update layout
fig.update_layout(
    title='Tennis Ball Trajectory',
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
    showlegend=True
)

# Save the plot to HTML
fig.write_html("ball_trajectory.html")

# Display the plot
fig.show()