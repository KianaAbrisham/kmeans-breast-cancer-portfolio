# K-means on the Wisconsin Diagnostic Breast Cancer Dataset

[![Checks](https://github.com/KianaAbrisham/kmeans-breast-cancer-portfolio/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/KianaAbrisham/kmeans-breast-cancer-portfolio/actions/workflows/checks.yml)

Explore clustering in the 569-observation, 30-feature dataset bundled with scikit-learn.
The notebook compares a NumPy implementation of Lloyd's algorithm with scikit-learn KMeans.

## Analysis

- Standardize the feature measurements, then cluster without diagnosis labels.
- Run one random initialization in the scratch implementation and 20 k-means++ starts in the reference.
- Compare inertia, silhouette score and adjusted Rand agreement with recorded diagnoses.
- Visualize clusters in a two-component PCA projection.
- Explore silhouette scores for cluster counts from two through seven.

Scaling and clustering use the full dataset. Diagnosis labels are used afterward for descriptive
comparison, including a best two-cluster label mapping. **These are in-sample exploratory results**,
not held-out diagnostic accuracy or evidence of clinical usefulness. PCA is used only for visualization.
The different initialization budgets also prevent treating this as an equal-budget algorithm benchmark.

## Implementation

The scratch function keeps an empty cluster's previous centroid and always recomputes assignments
against the returned centroids. It does not guarantee a global optimum or that every cluster is occupied.

| Path | Purpose |
|---|---|
| [notebooks/kmeans_breast_cancer.ipynb](notebooks/kmeans_breast_cancer.ipynb) | Executed clustering comparison and plots |
| [src/kmeans_scratch.py](src/kmeans_scratch.py) | NumPy Lloyd iteration |
| [tests/test_kmeans.py](tests/test_kmeans.py) | Final-assignment, empty-cluster and invalid-input checks |

Dataset documentation: [scikit-learn breast cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).

## Run locally

Use Python 3.12 and a separate environment for this project. From the repository folder:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter notebook notebooks/kmeans_breast_cancer.ipynb
```

The notebook finds the repository from either its root folder or `notebooks/`.
The saved outputs come from CPU execution with the included data; see
[validation](docs/VALIDATION.md) for the checks and limits.

[Development notes](https://github.com/KianaAbrisham/KianaAbrisham/blob/main/docs/DEVELOPMENT.md)

## License

MIT — see [LICENSE](LICENSE).
