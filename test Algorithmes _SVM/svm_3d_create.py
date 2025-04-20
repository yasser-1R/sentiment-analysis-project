import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import cvxpy as cp
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go

# === Sentences
sentences = [
    "good good good neutral",
    "bad bad bad neutral",
    "good neutral neutral good bad",
    "good good bad bad bad bad",
    "bad neutral",
    "good good bad bad bad bad bad bad neutral",
    "good good bad",
    "bad bad good neutral",
    "neutral good neutral neutral",
    "good good good neutral bad bad",
    "good neutral",
    "good good good good bad bad neutral"
]

# === Labels
def label_sentence(s):
    g, b = s.count("good"), s.count("bad")
    return 1 if g > b else -1 if b > g else 0

labels = np.array([label_sentence(s) for s in sentences])
sentences = [s for i, s in enumerate(sentences) if labels[i] != 0]
labels = labels[labels != 0]

# === TF-IDF
vectorizer = TfidfVectorizer(vocabulary=["good", "bad", "neutral"])
X = vectorizer.fit_transform(sentences).toarray()

# === Feasible SVM (Google OR-Tools)
solver = pywraplp.Solver.CreateSolver('GLOP')
w1, w2, w3 = [solver.NumVar(-10, 10, f'w{i}') for i in range(1, 4)]
b_lin = solver.NumVar(-10, 10, 'b')
for i in range(len(X)):
    solver.Add(labels[i] * (w1 * X[i, 0] + w2 * X[i, 1] + w3 * X[i, 2] + b_lin) >= 1)
solver.Minimize(0)
solver.Solve()
w_feas = np.array([w1.solution_value(), w2.solution_value(), w3.solution_value()])
b_feas = b_lin.solution_value()

# === Optimized SVM (cvxpy)
w = cp.Variable(3)
b = cp.Variable()
constraints = [labels[i] * (X[i] @ w + b) >= 1 for i in range(len(X))]
cp.Problem(cp.Minimize(cp.norm(w, 2)), constraints).solve()
w_opt, b_opt = w.value, b.value

# === Meshgrid (BIG enough to not clip planes)
xx, yy = np.meshgrid(np.linspace(-6, 6, 10), np.linspace(-6, 6, 10))
def get_plane(w, b): return -(w[0]*xx + w[1]*yy + b) / w[2]

# === Plotly figure
fig = go.Figure()

# === Add data points
for i in range(len(X)):
    fig.add_trace(go.Scatter3d(
        x=[X[i, 0]], y=[X[i, 1]], z=[X[i, 2]],
        mode='markers+text',
        marker=dict(size=5, color='green' if labels[i] == 1 else 'red'),
        text=[str(i+1)],
        textposition="top center",
        showlegend=False
    ))

# === Add 4 planes with distinct IDs (index positions)
num_data_points = len(X)
fig.add_trace(go.Surface(
    x=xx, y=yy, z=get_plane(w_feas, b_feas),
    opacity=0.4, showscale=False, name="Feasible Plane",
    colorscale=[[0, 'orange'], [1, 'orange']]
))  # Index: num_data_points

fig.add_trace(go.Surface(
    x=xx, y=yy, z=get_plane(w_opt, b_opt),
    opacity=0.4, showscale=False, name="Optimized Plane",
    colorscale=[[0, 'blue'], [1, 'blue']]
))  # Index: num_data_points + 1

fig.add_trace(go.Surface(
    x=xx, y=yy, z=get_plane(w_opt, b_opt - 1),
    opacity=0.2, showscale=False, name="Margin +1",
    colorscale=[[0, 'darkblue'], [1, 'darkblue']]
))  # Index: num_data_points + 2

fig.add_trace(go.Surface(
    x=xx, y=yy, z=get_plane(w_opt, b_opt + 1),
    opacity=0.15, showscale=False, name="Margin -1",
    colorscale=[[0, 'navy'], [1, 'navy']]
))  # Index: num_data_points + 3

# === Axis layout
axis_template = lambda title: dict(
    title=dict(text=title, font=dict(color='silver')),
    tickfont=dict(color='silver'),
    range=[-1.3, 1.3],  # Changed from [-4, 4]
    backgroundcolor='white',
    gridcolor='lightgray',
    showgrid=True
)

fig.update_layout(
    scene=dict(
        xaxis=axis_template("TF-IDF('good')"),
        yaxis=axis_template("TF-IDF('bad')"),
        zaxis=axis_template("TF-IDF('neutral')"),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    ),
    title="SVM in 3D: Full Planes + Axis Style + Multiple Plane Toggles",
    showlegend=True
)

# === Create all possible combinations of plane visibility
# Each combination is represented as a visibility array and a button label
combinations = []

# All planes visible (default)
all_visible = [True] * len(fig.data)
combinations.append((all_visible, "Show All Planes"))

# All planes hidden
all_planes_hidden = [True] * num_data_points + [False] * 4
combinations.append((all_planes_hidden, "Hide All Planes"))

# Individual planes hidden
plane_labels = ["Feasible Plane", "Optimized Plane", "Margin +1", "Margin -1"]
for i in range(4):
    visibility = [True] * len(fig.data)
    visibility[num_data_points + i] = False
    combinations.append((visibility, f"Hide {plane_labels[i]}"))

# Combinations of planes hidden
combinations.append(([True] * num_data_points + [True, False, False, False], "Show Feasible"))
combinations.append(([True] * num_data_points + [False, False, True, True], "Show Margins"))
combinations.append(([True] * num_data_points + [False, True, True, True], "Show Optimized & Margins"))
combinations.append(([True] * num_data_points + [False, True, False, False], "Show Optimized"))
combinations.append(([True] * num_data_points + [True, True, False, False], "Show Feasible & Optimized"))

# Combine buttons into the update menu
buttons = [dict(label=label, method="update", args=[{"visible": visibility}]) for visibility, label in combinations]

fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        direction="down",
        x=1.15, xanchor="left", y=0.9, yanchor="top",
        buttons=buttons
    )]
)

# === Save HTML
fig.write_html("svm_3d_simulate.html")
print("✅ Plot saved to 'svm_3d_simulate.html'")