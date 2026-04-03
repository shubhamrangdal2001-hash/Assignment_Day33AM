"""
Part D: AI-Augmented Task
Visualise how the SVM decision boundary changes as C varies from 0.01 to 100.
Uses a simple 2D dataset so we can actually plot the boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def make_2d_dataset(n_samples=200, random_state=42):
    """Create a slightly overlapping 2-class 2D dataset for visualisation."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.05,          # small noise so boundary is interesting
        random_state=random_state,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def plot_decision_boundary(ax, svm_model, X, y, title):
    """
    Draw the SVM decision boundary and margins on a given axes.
    Support vectors are circled.
    """
    h = 0.02  # mesh resolution
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Decision region shading
    ax.contourf(xx, yy, Z, levels=[-999, 0, 999],
                colors=["#FADADD", "#D4EDDA"], alpha=0.4)

    # Decision boundary (Z=0) and margin lines (Z=±1)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=["--", "-", "--"],
               colors=["#e74c3c", "#2c3e50", "#e74c3c"], linewidths=[1, 2, 1])

    # Data points
    scatter_colors = ["#e74c3c" if label == 0 else "#2980b9" for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=18, alpha=0.8, zorder=3)

    # Circle support vectors
    ax.scatter(svm_model.support_vectors_[:, 0],
               svm_model.support_vectors_[:, 1],
               s=80, facecolors="none", edgecolors="black",
               linewidths=1.2, zorder=4, label=f"SVs={len(svm_model.support_vectors_)}")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])


def visualise_c_effect():
    """Plot SVM decision boundary for 6 different C values."""
    X, y = make_2d_dataset()

    c_values = [0.01, 0.1, 1, 10, 50, 100]
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("SVM Decision Boundary: Effect of C Parameter (RBF Kernel)",
                 fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    for idx, c_val in enumerate(c_values):
        svm = SVC(kernel="rbf", C=c_val, gamma="scale")
        svm.fit(X, y)

        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        n_sv = len(svm.support_vectors_)
        plot_decision_boundary(
            ax, svm, X, y,
            title=f"C = {c_val}  |  Support Vectors = {n_sv}"
        )

    # Annotation box
    note = ("Low C → wide margin, more misclassifications (underfitting)\n"
            "High C → narrow margin, fewer misclassifications (may overfit)")
    fig.text(0.5, -0.03, note, ha="center", fontsize=10,
             bbox=dict(facecolor="#fffde7", edgecolor="#f0c040", boxstyle="round,pad=0.4"))

    plt.savefig("svm_c_boundary.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved → svm_c_boundary.png")


if __name__ == "__main__":
    visualise_c_effect()
    print("""
Kernel Trick Analogy (Part D — AI-generated & verified):
─────────────────────────────────────────────────────────
Imagine you have red and blue marbles scattered on a table
(2D space) and they are mixed together so no straight line
can separate them.

Now imagine lifting the table and giving it a gentle shake —
some marbles jump up higher than others based on their colour.
In that 3D view a flat sheet (hyperplane) can now separate the
two colours cleanly.

The kernel trick does exactly this mathematically: it computes
dot products in a higher-dimensional space WITHOUT explicitly
transforming the data. The RBF kernel is like an infinite-
dimensional lift — it can always find a separating hyperplane
no matter how tangled the original data is.

Evaluation:
  ✓ Accurate  — captures the core idea of implicit feature mapping.
  ✓ Helpful   — the "lifting marbles" image is easy to remember.
  ✓ The key point (no explicit transformation, just kernel matrix)
    is preserved, so students won't be misled.
""")
