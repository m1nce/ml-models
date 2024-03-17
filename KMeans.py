import pandas as pd
import numpy as np

"""
KMeans classifer
By: Minchan Kim
"""

class KMeans():
    def __init__(self, k=3, num_iter=1000):
        """
        Initialize the KMeans model
        """
        self.model_name = 'KMeans'
        self.k = k
        self.num_iter = num_iter
        self.centers = None
        self.RM = None


    def train(self, X):
        """
        Train the KMeans model
        ---
        Parameters:
            X: Input feature matrix
        ---
        Return:
            self
        """
        r, c = X.shape
        # Randomly choose the initial centers
        centers = X[np.random.choice(r, self.k, replace=False)]

        for _ in range(self.num_iter):
            RM = np.zeros((r, self.k))

            # Assign data points to the nearest centroid
            for i in range(r):
                distances = np.linalg.norm(X[i] - centers, axis=1)
                closest = np.argmin(distances)
                RM[i, closest] = 1
            
            new_centers = np.zeros((self.k, c))
            # Recalculate the centroids
            for j in range(self.k):
                points_in_cluster = X[RM[:, j] == 1]
                if len(points_in_cluster) > 0:
                    new_centers[j] = np.mean(points_in_cluster, axis=0)
            
            # Check for convergence
            if np.all(centers == new_centers):
                break

            centers = new_centers

        self.centers = centers
        self.RM = RM
        return self
