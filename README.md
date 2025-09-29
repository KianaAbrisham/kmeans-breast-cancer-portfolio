# K-means Clustering on Breast Cancer Data (Portfolio Sample)

This repository demonstrates **unsupervised learning** with **K-means** on the Breast Cancer dataset from `scikit-learn`.
It mirrors the kind of data-quality thinking and evaluation used in clinical research—mapping clusters to ground truth,
reporting metrics, and producing clear, reproducible figures.

## What this shows
- End-to-end **reproducible analysis**: data loading, scaling, clustering, evaluation, and visualization
- **Metrics** for unsupervised evaluation: silhouette score; label mapping for precision/recall/accuracy (for interpretability)
- Clean, publication-style plots (PCA projection of clusters; K vs. silhouette)
- A simple **from-scratch K-means** implementation (for transparency), alongside a `scikit-learn` reference

## Repo structure
```
.
├── notebooks
│   └── kmeans_breast_cancer.ipynb     # Reproducible notebook with code + plots
├── src
│   └── kmeans_scratch.py              # Minimal from-scratch K-means (NumPy)
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quickstart
```bash
# 1) Create a virtual environment (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the notebook
jupyter notebook notebooks/kmeans_breast_cancer.ipynb
```

## Notes
- This project uses **public, non-sensitive** data.
- It focuses on **clarity, evaluation, and reproducibility**, which are critical when preparing real-world research datasets.

## License
MIT — see `LICENSE`.
