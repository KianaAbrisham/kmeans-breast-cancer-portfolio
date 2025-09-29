import numpy as np

def kmeans(X, k=2, max_iter=100, tol=1e-4, random_state=42):
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    # Init centroids by random choice
    centroids = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n, k)
        new_labels = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.array([X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else centroids[j]
                                  for j in range(k)])

        # Check convergence
        shift = np.linalg.norm(new_centroids - centroids)
        centroids, labels = new_centroids, new_labels
        if shift < tol:
            break
    return centroids, labels
