"""
Part B: Approximate Nearest Neighbors using FAISS
Comparing sklearn KNN vs FAISS on the digits dataset
"""

import time
import numpy as np

import faiss
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ── data prep ────────────────────────────────────────────────────────────────

def prepare_data():
    """Load and scale the digits dataset. Returns float32 arrays (FAISS needs float32)."""
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def build_faiss_index(X_train):
    """Build a flat L2 FAISS index from training vectors."""
    dim = X_train.shape[1]
    index = faiss.IndexFlatL2(dim)   # exact search (brute force over L2 distance)
    index.add(X_train)
    print(f"FAISS index built. Total vectors: {index.ntotal}, Dimension: {dim}")
    return index


def faiss_predict(index, X_train_labels, X_query, k=3):
    """
    Use FAISS index to find K nearest neighbours and do majority-vote prediction.
    Returns predicted labels array.
    """
    # distances shape: (n_queries, k), indices shape: (n_queries, k)
    _, indices = index.search(X_query, k)

    predictions = []
    for neighbor_ids in indices:
        neighbor_labels = X_train_labels[neighbor_ids]
        # majority vote
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])

    return np.array(predictions)


# ── speed benchmark ───────────────────────────────────────────────────────────

def benchmark_sklearn_knn(X_train, y_train, X_query, k=3, n_runs=3):
    """Time sklearn KNN prediction on X_query."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        knn.predict(X_query)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    preds = knn.predict(X_query)
    return preds, avg_time


def benchmark_faiss(faiss_index, y_train, X_query, k=3, n_runs=3):
    """Time FAISS KNN search on X_query."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        preds = faiss_predict(faiss_index, y_train, X_query, k=k)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    return preds, avg_time


# ── accuracy helper ───────────────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()

    # Use 1000 query samples for fair benchmark
    n_queries = min(1000, X_test.shape[0])
    X_query = X_test[:n_queries]
    y_query = y_test[:n_queries]
    print(f"Running benchmark on {n_queries} queries, K=3\n")

    # --- sklearn KNN ---
    print("Benchmarking sklearn KNN...")
    sk_preds, sk_time = benchmark_sklearn_knn(X_train, y_train, X_query)
    sk_acc = accuracy(y_query, sk_preds)
    print(f"  sklearn KNN  | Time: {sk_time:.4f}s | Accuracy: {sk_acc:.4f}")

    # --- FAISS ---
    print("\nBuilding FAISS index and benchmarking...")
    faiss_index = build_faiss_index(X_train)
    faiss_preds, faiss_time = benchmark_faiss(faiss_index, y_train, X_query)
    faiss_acc = accuracy(y_query, faiss_preds)
    print(f"  FAISS KNN    | Time: {faiss_time:.4f}s | Accuracy: {faiss_acc:.4f}")

    # --- Summary ---
    speedup = sk_time / faiss_time if faiss_time > 0 else float("inf")
    print("\n" + "="*55)
    print("  FAISS vs sklearn KNN — Benchmark Summary")
    print("="*55)
    print(f"  {'Method':<15} {'Time (s)':>10} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    print(f"  {'sklearn KNN':<15} {sk_time:>10.4f} {sk_acc:>10.4f}")
    print(f"  {'FAISS KNN':<15} {faiss_time:>10.4f} {faiss_acc:>10.4f}")
    print(f"\n  FAISS speedup: {speedup:.2f}x faster than sklearn")
    print("="*55)

    print("""
Findings:
- FAISS uses optimised C++/SIMD under the hood, so it is faster especially
  at larger scale.
- Accuracy is identical here (both use exact L2 search).
- At production scale (millions of vectors), FAISS switches to approximate
  indexes (IVF, HNSW) trading a tiny accuracy drop for massive speed gains.
- This makes FAISS ideal for recommendation and RAG similarity search.
""")


if __name__ == "__main__":
    main()
