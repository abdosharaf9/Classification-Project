import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Get the mean value to make the data centered
        mean = np.mean(X, axis=0)
        X = X - mean

        # Compute the covariance matrix of the data
        cov_matrix = np.cov(X.T)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Select the top n_components eigenvectors based on their corresponding eigenvalues
        idx = eigenvalues.argsort()[::-1][:self.n_components]
        self.components = eigenvectors[:, idx]

    def transform(self, X):
        # Project the data onto the top n_components principal components
        return X.dot(self.components)