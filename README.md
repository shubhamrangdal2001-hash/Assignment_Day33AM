# D33 | Take-Home Assignment: SVM & KNN

**IIT Gandhinagar – PG Diploma in AI-ML & Agentic AI Engineering**  
Day 33 | AM Session | Week 6  
Topics: SVM (hard/soft margin, C, gamma, RBF kernel) · KNN (distance metrics, K selection, FAISS)

---

## Folder Structure

```
svm_knn_assignment/
├── README.md                          ← you are here
├── requirements.txt                   ← all dependencies
├── .gitignore
│
├── D33_SVM_KNN_Assignment.ipynb       ← complete Jupyter notebook (all parts)
├── D33_SVM_KNN_Solution.docx          ← part-wise written solution document
│
├── part_a/
│   ├── digit_classifier.py            ← SVM + KNN on sklearn digits dataset
│   ├── svm_confusion_matrix.png       ← generated output
│   ├── knn_confusion_matrix.png       ← generated output
│   └── knn_k_selection.png            ← K vs CV accuracy plot
│
├── part_b/
│   └── faiss_knn.py                   ← FAISS vs sklearn KNN speed benchmark
│
├── part_c/
│   └── interview_ready.py             ← KNN from scratch + SVM debug fix
│
└── part_d/
    ├── svm_visualization.py           ← SVM decision boundary (C parameter effect)
    └── svm_c_boundary.png             ← generated output
```

---

## What Each Part Does

| Part | File | Task |
|------|------|------|
| A | `part_a/digit_classifier.py` | SVM (RBF + GridSearchCV) and KNN (optimal K) on handwritten digits. Compares accuracy, plots confusion matrices. |
| B | `part_b/faiss_knn.py` | FAISS approximate nearest-neighbour search vs sklearn KNN. Benchmarks speed and accuracy on 1000 queries. |
| C | `part_c/interview_ready.py` | (Q2) KNN built from scratch using only NumPy. (Q3) Reproduce and fix a broken SVM caused by missing feature scaling. |
| D | `part_d/svm_visualization.py` | Visualise how SVM decision boundary changes as C varies from 0.01 to 100. Includes kernel trick analogy. |

---

## Setup

**Python 3.8+ required.**

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/svm_knn_assignment.git
cd svm_knn_assignment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` is the CPU-only build. If you have a GPU you can swap it for `faiss-gpu`.

---

## How to Run

### Run individual scripts
```bash
python part_a/digit_classifier.py
python part_b/faiss_knn.py
python part_c/interview_ready.py
python part_d/svm_visualization.py
```

### Run the Jupyter notebook
```bash
jupyter notebook D33_SVM_KNN_Assignment.ipynb
```
Run all cells top-to-bottom. The notebook reproduces all outputs from all four parts in one place.

---

## Key Results

| Model | Test Accuracy | Notes |
|-------|-------------|-------|
| SVM (RBF, C=100, gamma=0.01) | **0.9833** | Best params found via GridSearchCV 5-fold CV |
| KNN (K=1) | 0.9667 | Best K found by cross-validating K=1..20 |
| sklearn KNN (1000 queries) | 0.9667 — 0.0495s | Baseline |
| FAISS KNN (1000 queries) | 0.9667 — 0.0066s | **~7.5x faster**, identical accuracy |

**Most confused digit pairs (both models):** `8↔1`, `4↔7`, `9↔6` — makes visual sense at 8×8 resolution.

---

## Git Commit Strategy

Suggested commits for clean git history:
```
git commit -m "Initial project structure and README"
git commit -m "Add Part A: SVM and KNN digit classifier with GridSearchCV"
git commit -m "Add Part B: FAISS approximate KNN benchmark vs sklearn"
git commit -m "Add Part C: KNN from scratch (NumPy) and SVM debug fix"
git commit -m "Add Part D: SVM decision boundary C-parameter visualization"
git commit -m "Add Jupyter notebook combining all parts"
git commit -m "Add requirements.txt and .gitignore"
```

---

## Dependencies

See `requirements.txt`. Main packages:

- `scikit-learn` — SVM, KNN, datasets, preprocessing, metrics  
- `faiss-cpu` — approximate nearest neighbour search  
- `numpy` — numerical operations (KNN from scratch)  
- `matplotlib` / `seaborn` — plots and confusion matrices  
- `jupyter` — notebook environment  

---

## Author

Student submission — IIT Gandhinagar PG Diploma in AI-ML  
Week 6 · Day 33 Take-Home Assignment
