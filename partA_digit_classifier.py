"""
Part A: Handwritten Digit Classifier using SVM and KNN
Dataset: sklearn digits (8x8 pixel images, 10 classes 0-9)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ── helpers ──────────────────────────────────────────────────────────────────

def load_and_split(test_size=0.2, random_state=42):
    """Load digits dataset, scale it, and return train/test splits."""
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_svm_with_grid_search(X_train, y_train):
    """
    Train SVM with RBF kernel using GridSearchCV to find best C and gamma.
    Returns the best estimator.
    """
    param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
    }
    svm = SVC(kernel="rbf", random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best SVM params : {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def find_best_k(X_train, y_train, k_range=range(1, 21)):
    """Try different K values and return the one with best cross-val accuracy."""
    from sklearn.model_selection import cross_val_score

    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_score = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        scores.append(cv_score.mean())

    best_k = list(k_range)[np.argmax(scores)]
    print(f"Best K = {best_k}  (CV accuracy = {max(scores):.4f})")

    # Plot K vs accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), scores, marker="o", color="steelblue")
    plt.xlabel("K")
    plt.ylabel("CV Accuracy")
    plt.title("KNN: K vs Cross-Validation Accuracy")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("knn_k_selection.png", dpi=120)
    plt.close()

    return best_k


def evaluate_model(model, X_test, y_test, model_name):
    """Print accuracy, classification report, and return confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    return y_pred, cm, acc


def plot_confusion_matrix(cm, title, filename):
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved → {filename}")


def find_confused_pairs(cm, top_n=5):
    """Return the most commonly confused digit pairs (off-diagonal entries)."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)
    print(f"\nTop {top_n} confused digit pairs:")
    for count, true_d, pred_d in pairs[:top_n]:
        print(f"  True={true_d}  Predicted={pred_d}  Count={count}")
    return pairs[:top_n]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading and scaling digits dataset...")
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

    # --- SVM ---
    print("\nTraining SVM with GridSearchCV (RBF kernel)...")
    best_svm = train_svm_with_grid_search(X_train, y_train)
    y_pred_svm, cm_svm, acc_svm = evaluate_model(best_svm, X_test, y_test, "SVM (RBF)")
    plot_confusion_matrix(cm_svm, "SVM Confusion Matrix", "svm_confusion_matrix.png")

    # --- KNN ---
    print("\nFinding optimal K for KNN...")
    best_k = find_best_k(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred_knn, cm_knn, acc_knn = evaluate_model(knn, X_test, y_test, f"KNN (K={best_k})")
    plot_confusion_matrix(cm_knn, f"KNN Confusion Matrix (K={best_k})", "knn_confusion_matrix.png")

    # --- Comparison ---
    print("\n" + "="*50)
    print("  MODEL COMPARISON")
    print("="*50)
    print(f"SVM (RBF)  accuracy: {acc_svm:.4f}")
    print(f"KNN (K={best_k})  accuracy: {acc_knn:.4f}")

    print("\nMost confused pairs (SVM):")
    find_confused_pairs(cm_svm)

    print("\nMost confused pairs (KNN):")
    find_confused_pairs(cm_knn)


if __name__ == "__main__":
    main()
