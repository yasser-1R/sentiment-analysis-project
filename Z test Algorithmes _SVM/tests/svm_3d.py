import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import cvxpy as cp
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go

# === Sentences using 3 keywords
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

# === Feasible weights using OR-Tools
solver = pywraplp.Solver.CreateSolver('GLOP')
w1 = solver.NumVar(-10, 10, 'w1')
w2 = solver.NumVar(-10, 10, 'w2')
w3 = solver.NumVar(-10, 10, 'w3')
b_lin = solver.NumVar(-10, 10, 'b')

for i in range(len(X)):
    x1, x2, x3 = X[i]
    y = labels[i]
    solver.Add(y * (w1 * x1 + w2 * x2 + w3 * x3 + b_lin) >= 1)

solver.Minimize(0)
solver.Solve()
w_feasible = np.array([w1.solution_value(), w2.solution_value(), w3.solution_value()])
b_feasible = b_lin.solution_value()

# === Optimized weights using CVXPY
w_opt = cp.Variable(3)
b_opt = cp.Variable()
constraints = [labels[i] * (X[i] @ w_opt + b_opt) >= 1 for i in range(len(X))]
objective = cp.Minimize(cp.norm(w_opt, 2))
problem = cp.Problem(objective, constraints)
problem.solve()
w_opt = w_opt.value
b_opt = b_opt.value

# === 3D Visualization with Plotly
fig = go.Figure()

# === Points
for i in range(len(X)):
    color = 'green' if labels[i] == 1 else 'red'
    fig.add_trace(go.Scatter3d(
        x=[X[i, 0]], y=[X[i, 1]], z=[X[i, 2]],
        mode='markers+text',
        marker=dict(size=5, color=color),
        name=f"Sentence {i+1}",
        text=[str(i+1)],
        textposition="top center",
        legendgroup="points",
        showlegend=(i == 0)
    ))

# === Planes
xx, yy = np.meshgrid(np.linspace(0, 1.2, 10), np.linspace(0, 1.2, 10))

# Feasible plane
zz_feasible = -(w_feasible[0] * xx + w_feasible[1] * yy + b_feasible) / w_feasible[2]
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz_feasible, showscale=False, opacity=0.3,
    colorscale=[[0, 'orange'], [1, 'orange']],
    name='Feasible Plane',
    visible=True,
    legendgroup='feasible'
))

# Optimized plane
zz_opt = -(w_opt[0] * xx + w_opt[1] * yy + b_opt) / w_opt[2]
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz_opt, showscale=False, opacity=0.4,
    colorscale=[[0, 'blue'], [1, 'blue']],
    name='Optimized Plane',
    visible=True,
    legendgroup='opt'
))

# Margin +1
zz_plus = -(w_opt[0] * xx + w_opt[1] * yy + (b_opt - 1)) / w_opt[2]
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz_plus, showscale=False, opacity=0.2,
    colorscale=[[0, 'lightblue'], [1, 'lightblue']],
    name='Margin +1',
    visible=True,
    legendgroup='plus'
))

# Margin -1
zz_minus = -(w_opt[0] * xx + w_opt[1] * yy + (b_opt + 1)) / w_opt[2]
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz_minus, showscale=False, opacity=0.2,
    colorscale=[[0, 'lightblue'], [1, 'lightblue']],
    name='Margin -1',
    visible=True,
    legendgroup='minus'
))

# === Buttons for visibility toggles
buttons = [
    dict(label="Show All", method="update",
         args=[{"visible": [True]*len(fig.data)}]),
    dict(label="Hide All Planes", method="update",
         args=[{"visible": [True]*len(X) + [False, False, False, False]}]),
    dict(label="Feasible Only", method="update",
         args=[{"visible": [True]*len(X) + [True, False, False, False]}]),
    dict(label="Optimized Only", method="update",
         args=[{"visible": [True]*len(X) + [False, True, False, False]}]),
    dict(label="Optimized + Margin", method="update",
         args=[{"visible": [True]*len(X) + [False, True, True, True]}]),
]

# === Layout
fig.update_layout(
    scene=dict(
        xaxis_title="TF-IDF('good')",
        yaxis_title="TF-IDF('bad')",
        zaxis_title="TF-IDF('neutral')"
    ),
    title="SVM: Toggle Feasible, Optimized & Margin Planes",
    showlegend=True,
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            showactive=True,
            x=1.15,
            xanchor="left",
            y=0.9,
            yanchor="top",
            buttons=buttons
        )
    ]
)

# Save to HTML
fig.write_html("svm_3d_toggle_planes.html")
print("✅ Interactive 3D plot with toggle options saved as 'svm_3d_toggle_planes.html'")
