"""Lloyd's K-means with one random initialization and explicit empty clusters."""
import numpy as np

def kmeans(X, k=2, max_iter=100, tol=1e-4, random_state=42):
    """Return centroids and their nearest-centroid labels.

    Empty clusters keep their previous centroid. This simple implementation
    does not guarantee k occupied clusters or a global optimum.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError('X must be a nonempty finite two-dimensional matrix.')
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError('k must be an integer between 1 and the number of rows.')
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer.')
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('tol must be finite and nonnegative.')
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(max_iter):
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        updated = np.array([
            X[labels == cluster].mean(axis=0) if np.any(labels == cluster) else centroids[cluster]
            for cluster in range(k)
        ])
        shift = np.linalg.norm(updated - centroids)
        centroids = updated
        if shift <= tol:
            break
    # The last update can move a boundary, especially when max_iter is reached.
    distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = distances.argmin(axis=1)
    return centroids, labels
