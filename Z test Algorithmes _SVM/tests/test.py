import plotly.graph_objects as go
import numpy as np

# Sample 3D vectors
X = np.array([
    [1, 0, 1],
    [0, 2, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 2],
])
labels = [1, -1, 1, -1, 1]

# Assign colors based on sentiment
colors = ['green' if label == 1 else 'red' for label in labels]

# Create 3D scatter plot of the points
scatter = go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=X[:, 2],
    mode='markers+text',
    marker=dict(size=6, color=colors),
    text=[str(list(x)) for x in X],
    textposition='top center'
)

# Define a sample plane: x + 2y + 3z = 4 => z = -(x + 2y - 4)/3
w = np.array([1, 2, 3])
b = -4
xx, yy = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 2, 10))
zz = -(w[0] * xx + w[1] * yy + b) / w[2]

# Plot the plane
plane = go.Surface(
    x=xx, y=yy, z=zz,
    opacity=0.5,
    showscale=False,
    colorscale='Blues'
)

# Build the figure
fig = go.Figure(data=[scatter, plane])

# Layout settings: fixed axis ranges, clean cube view
fig.update_layout(
    title="3D Vector Space (Fixed Interactive View)",
    width=900,
    height=800,
    scene=dict(
        xaxis=dict(title='good', range=[0, 2]),
        yaxis=dict(title='bad', range=[0, 2]),
        zaxis=dict(title='amazing', range=[0, 2]),
        aspectmode='cube'
    ),
    scene_camera=dict(
        eye=dict(x=1.8, y=1.8, z=1.2)  # Fixed starting view angle
    )
)

# Save as interactive HTML file
fig.write_html("svm_3d_plot_fixed_static.html")
