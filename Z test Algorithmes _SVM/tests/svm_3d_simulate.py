import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import cvxpy as cp
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go

# === Dataset
sentences = [
    "good good good neutral",
    "bad bad bad neutral",
    "good neutral neutral bad",
    "good good bad",
    "bad bad good neutral",
    "neutral neutral neutral",
    "good good neutral bad bad",
    "good neutral",
    "bad neutral",
    "good good good bad bad neutral"
]

# === Labeling
def label_sentence(s):
    g = s.count("good")
    b = s.count("bad")
    return 1 if g > b else -1 if b > g else 0

labels = np.array([label_sentence(s) for s in sentences])
sentences = [s for i, s in enumerate(sentences) if labels[i] != 0]
labels = labels[labels != 0]

# === TF-IDF
vectorizer = TfidfVectorizer(vocabulary=["good", "bad", "neutral"])
X = vectorizer.fit_transform(sentences).toarray()

# === Feasible SVM with OR-Tools
solver = pywraplp.Solver.CreateSolver('GLOP')
w1 = solver.NumVar(-10, 10, 'w1')
w2 = solver.NumVar(-10, 10, 'w2')
w3 = solver.NumVar(-10, 10, 'w3')
b_lin = solver.NumVar(-10, 10, 'b')
for i in range(len(X)):
    solver.Add(labels[i] * (w1 * X[i,0] + w2 * X[i,1] + w3 * X[i,2] + b_lin) >= 1)
solver.Minimize(0)
solver.Solve()
w_feasible = np.array([w1.solution_value(), w2.solution_value(), w3.solution_value()])
b_feasible = b_lin.solution_value()

# === Optimized SVM with cvxpy
w = cp.Variable(3)
b = cp.Variable()
constraints = [labels[i] * (X[i] @ w + b) >= 1 for i in range(len(X))]
objective = cp.Minimize(cp.norm(w, 2))
cp.Problem(objective, constraints).solve()
w_opt = w.value
b_opt = b.value

# === Plotting setup
xx, yy = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))

def plane_z(w, b):
    return -(w[0]*xx + w[1]*yy + b) / w[2]

fig = go.Figure()

# === Plot points
for i in range(len(X)):
    fig.add_trace(go.Scatter3d(
        x=[X[i,0]], y=[X[i,1]], z=[X[i,2]],
        mode='markers+text',
        marker=dict(size=5, color='green' if labels[i]==1 else 'red'),
        text=[str(i+1)],
        textposition="top center",
        showlegend=False
    ))

# === Add all 4 planes
fig.add_trace(go.Surface(
    x=xx, y=yy, z=plane_z(w_feasible, b_feasible),
    opacity=0.15, showscale=False,
    name='Feasible Plane',
    colorscale=[[0, 'orange'], [1, 'orange']]
))
fig.add_trace(go.Surface(
    x=xx, y=yy, z=plane_z(w_opt, b_opt),
    opacity=0.2, showscale=False,
    name='Optimized Plane',
    colorscale=[[0, 'blue'], [1, 'blue']]
))
fig.add_trace(go.Surface(
    x=xx, y=yy, z=plane_z(w_opt, b_opt - 1),
    opacity=0.1, showscale=False,
    name='Margin +1',
    colorscale=[[0, 'lightblue'], [1, 'lightblue']]
))
fig.add_trace(go.Surface(
    x=xx, y=yy, z=plane_z(w_opt, b_opt + 1),
    opacity=0.1, showscale=False,
    name='Margin -1',
    colorscale=[[0, 'lightblue'], [1, 'lightblue']]
))

fig.update_layout(
    scene=dict(
        xaxis=dict(
            title=dict(text="TF-IDF('good')", font=dict(color='lightgray')),
            tickfont=dict(color='lightgray'),
            range=[-4, 4],
            backgroundcolor='white',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text="TF-IDF('bad')", font=dict(color='lightgray')),
            tickfont=dict(color='lightgray'),
            range=[-4, 4],
            backgroundcolor='white',
            showgrid=True,
            gridcolor='lightgray'
        ),
        zaxis=dict(
            title=dict(text="TF-IDF('neutral')", font=dict(color='lightgray')),
            tickfont=dict(color='lightgray'),
            range=[-4, 4],
            backgroundcolor='white',
            showgrid=True,
            gridcolor='lightgray'
        ),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    ),
    title="SVM in 3D: Fixed Axes, Light Labels, Full Planes",
    showlegend=True
)


# === Save
fig.write_html("svm_3d_fixed_cubic_light.html")
print("✅ Saved to 'svm_3d_fixed_cubic_light.html'")
