import pandas as pd
import numpy as np
from KMeans import * # Import the KMeans classifer

"""
Gaussian Mixture Model classifer
By: Minchan Kim
"""

def gaussian(X, mu, cov):
    """ 
    Function to create mixtures using the given matrix X, given covariance and mu.
    ---
    Parameter:
        X: Input feature matrix
        mu: Mean of the cluster
        cov: Covariance of the cluster
    ---
    Return:
    transformed x.
    """
    # X should be matirx-like
    # X should be matirx-like
    n = X.shape[1]
    diff = (X - mu).T
    # Ensure cov is 2-dimensional. If it's 1D, convert it to 2D diagonal matrix.
    if cov.ndim == 1:
        cov = np.diag(cov)
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)


def initialize_clusters(X, n_clusters):
    """ 
    Initialize the clusters by storing the information in the data matrix X into the clusters
    ---
    Parameter:
        X: Input feature matrix
        n_clusters: Number of clusters we are trying to classify
    ---
    Return:
        cluster: List of clusters. Each cluster center is calculated by the KMeans algorithm above.
    """
    clusters = []
    index = np.arange(X.shape[0])
    
    # We use the KMeans centroids to initialize the GMM
    kmeans = KMeans().train(X)
    mu_k = kmeans.centers
    
    for i in range(n_clusters):
        clusters.append({
            'w_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })
        
    return clusters


def expectation_step(X, clusters):
    """ 
    "E-Step" for the GM algorithm
    ---
    Parameter:
        X: Input feature matrix
        clusters: List of clusters
    """
    N = X.shape[0]
    K = len(clusters)  # Number of clusters
    totals = np.zeros((N, 1), dtype=np.float64)  # Initialize totals for each data point
    
    # Compute the probability of each data point in each cluster
    for cluster in clusters:
        pi_k = cluster['w_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        
        # Calculate the Gaussian probability for each data point for this cluster
        prob = gaussian(X, mu_k, cov_k) * pi_k  # Weighted by the cluster weight
        cluster['prob'] = prob  # Store the weighted probability for later use
        totals += prob  # Summing the weighted probabilities for normalization
    
    # Calculate and store the posterior probabilities for each cluster
    for cluster in clusters:
        cluster['posterior'] = cluster['prob'] / totals  # Normalize to get posterior probabilities

    # Update totals in each cluster dictionary for consistency
    for cluster in clusters:
        cluster['totals'] = totals


def maximization_step(X, clusters):
    """ 
    "M-Step" for the GM algorithm
    ---
    Parameter:
        X: Input feature matrix
        clusters: List of clusters
    """
    N = float(X.shape[0])
  
    for cluster in clusters:
        posterior = cluster['posterior'].reshape(-1)  # Ensure posterior is a 1D array for easier handling
        N_k = posterior.sum()
        
        # Update weights
        w_k = N_k / N
        
        # Update means
        mu_k = np.dot(posterior, X) / N_k
        
        # Update covariance
        diff = X - mu_k
        cov_k = np.dot(posterior * diff.T, diff) / N_k  # Adjusted for correct broadcasting
        
        cluster['w_k'] = w_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = np.diag(cov_k)  # Ensure covariance is correctly shaped, assuming independence


def get_likelihood(X, clusters):
    """
    Function to calculate the likelihood of the data given the clusters
    ---
    Parameter:
        X: Input feature matrix
        clusters: List of clusters
    """
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods


def train_gmm(X, n_clusters, n_epochs):
    """
    Function to train the GMM model
    ---
    Parameter:
        X: Input feature matrix
        n_clusters: Number of clusters we are trying to classify
        n_epochs: Number of epochs
    ---
    Return:
        clusters: List of clusters
        likelihoods: List of likelihoods
        scores: List of scores
        sample_likelihoods: List of sample likelihoods
    """
    clusters = initialize_clusters(X, n_clusters)
    likelihoods = np.zeros((n_epochs, ))
    scores = np.zeros((X.shape[0], n_clusters))

    for i in range(n_epochs):
      
        expectation_step(X, clusters)
        maximization_step(X, clusters)

        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood
        
    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['w_k']).reshape(-1)
        
    return clusters, likelihoods, scores, sample_likelihoods