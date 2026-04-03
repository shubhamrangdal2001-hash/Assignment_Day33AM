"""
Part C: Interview Ready
  Q2 - KNN from scratch using only NumPy
  Q3 - Debug the broken SVM (feature scaling issue)
"""

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ── Q2: KNN from scratch ──────────────────────────────────────────────────────

def euclidean_distances(X_train, x_query):
    """
    Compute Euclidean distance between one query point and all training points.
    Returns 1D array of distances.
    """
    diff = X_train - x_query          # broadcast: shape (n_train, n_features)
    return np.sqrt(np.sum(diff ** 2, axis=1))


def knn_from_scratch(X_train, y_train, X_test, k):
    """
    KNN classifier implemented with only NumPy.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
    y_train : ndarray, shape (n_train,)
    X_test  : ndarray, shape (n_test, n_features)
    k       : int, number of neighbours

    Returns
    -------
    predictions : ndarray, shape (n_test,)
    """
    predictions = []

    for query_point in X_test:
        # Step 1: compute distance to every training point
        distances = euclidean_distances(X_train, query_point)

        # Step 2: find K nearest neighbour indices
        k_nearest_indices = np.argsort(distances)[:k]

        # Step 3: majority vote among K neighbours
        k_nearest_labels = y_train[k_nearest_indices]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]

        predictions.append(predicted_label)

    return np.array(predictions)


def demo_knn_from_scratch():
    """Quick demo comparing scratch KNN vs sklearn KNN on digits."""
    from sklearn.neighbors import KNeighborsClassifier

    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Use small test subset so it runs quickly
    X_small = X_test_sc[:100]
    y_small = y_test[:100]

    # Scratch KNN
    preds_scratch = knn_from_scratch(X_train_sc, y_train, X_small, k=3)
    acc_scratch = accuracy_score(y_small, preds_scratch)

    # sklearn KNN
    knn_sk = KNeighborsClassifier(n_neighbors=3)
    knn_sk.fit(X_train_sc, y_train)
    preds_sk = knn_sk.predict(X_small)
    acc_sk = accuracy_score(y_small, preds_sk)

    print("="*45)
    print("  Q2: KNN from Scratch Demo (100 samples)")
    print("="*45)
    print(f"  Scratch KNN accuracy : {acc_scratch:.4f}")
    print(f"  sklearn KNN accuracy : {acc_sk:.4f}")
    print(f"  Results match        : {np.array_equal(preds_scratch, preds_sk)}")


# ── Q3: Debug broken SVM ──────────────────────────────────────────────────────

def demo_debug_svm():
    """
    Reproduce the bug (no feature scaling) and show the fix.
    Features: salary (50K-200K range) vs age (20-60 range).
    Without scaling, salary completely dominates the RBF kernel distance.
    """
    np.random.seed(42)
    n_samples = 500

    # Simulate salary (50000-200000) and age (20-60)
    salary = np.random.uniform(50000, 200000, n_samples)
    age    = np.random.uniform(20, 60, n_samples)
    # Label based on simple rule
    label  = ((salary > 125000) & (age > 35)).astype(int)

    X = np.column_stack([salary, age])
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

    # --- BUGGY: no scaling ---
    svm_buggy = SVC(kernel="rbf", C=1.0)
    svm_buggy.fit(X_train, y_train)
    acc_buggy = svm_buggy.score(X_test, y_test)

    # --- FIXED: with StandardScaler ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    svm_fixed = SVC(kernel="rbf", C=1.0)
    svm_fixed.fit(X_train_sc, y_train)
    acc_fixed = svm_fixed.score(X_test_sc, y_test)

    print("\n" + "="*45)
    print("  Q3: Debug - SVM Scaling Fix")
    print("="*45)
    print(f"  Buggy SVM accuracy (no scaling) : {acc_buggy:.4f}  ← ~random!")
    print(f"  Fixed SVM accuracy (scaled)     : {acc_fixed:.4f}  ← much better")
    print("""
Root cause:
  - Salary range (~150K) >> Age range (~40).
  - RBF kernel: exp(-gamma * ||x_i - x_j||^2)
  - Without scaling, salary dominates ||x||^2 entirely.
  - The kernel effectively ignores the age feature.
  Fix: Always apply StandardScaler before training an SVM.
""")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_knn_from_scratch()
    demo_debug_svm()
