import unittest
import numpy as np
from src.kmeans_scratch import kmeans


class KMeansTests(unittest.TestCase):
    def test_labels_match_returned_centroids_after_one_update(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        centers, labels = kmeans(X, k=3, max_iter=1, random_state=42)
        nearest = np.sum((X[:, None] - centers[None, :]) ** 2, axis=2).argmin(axis=1)
        np.testing.assert_array_equal(labels, nearest)

    def test_empty_cluster_does_not_produce_nan(self):
        centers, labels = kmeans(np.ones((5, 2)), k=3)
        self.assertTrue(np.isfinite(centers).all())
        np.testing.assert_array_equal(labels, np.zeros(5, dtype=int))

    def test_reject_invalid_inputs(self):
        for kwargs in [{"k": 0}, {"k": 4}, {"max_iter": 0}, {"tol": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kmeans([[1], [2], [3]], **kwargs)
        with self.assertRaises(ValueError):
            kmeans([[np.nan], [1]])


if __name__ == "__main__":
    unittest.main()
