import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal


class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None, learning_rate=0.01):
        self.k = k
        self.dim = dim
        self.learning_rate = learning_rate
        if init_mu is None:
            init_mu = np.random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        if init_sigma is None:
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if init_pi is None:
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi
        if colors is None:
            colors = np.random.rand(k, 3)
        self.colors = colors
    def init_gibbs(self, X):
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))
    def gibbs_update(self):
        # Update cluster assignments z_i based on posterior probabilities
        for i in range(self.num_points):
            posteriors = np.zeros(self.k)
            for j in range(self.k):
                posteriors[j] = self.pi[j] * multivariate_normal.pdf(self.data[i], mean=self.mu[j], cov=self.sigma[j])
            # Check for valid posteriors and normalize them
            total_posterior = np.sum(posteriors)
            if total_posterior > 0:
                posteriors /= total_posterior  # Normalize posteriors
                # Use multinomial sampling for cluster assignment
                self.z[i] = np.zeros(self.k)
                chosen_cluster = np.random.multinomial(1, posteriors).argmax()
                self.z[i, chosen_cluster] = 1
            else:
                # If total_posterior is zero, assign uniformly or handle appropriately
                self.z[i] = np.random.dirichlet(np.ones(self.k))  # Uniform assignment
    def sgld_update(self):
        epsilon = 1e-6  # Small constant for numerical stability
        for i in range(self.k):
            # Select points assigned to cluster i
            assigned_points = self.data[self.z[:, i] > 0]
            if len(assigned_points) > 0:
                # Mean update using SGLD with preconditioning
                gradient_mu = np.zeros(self.dim)
                for x in assigned_points:
                    gradient_mu += multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma[i]) * (x - self.mu[i])
                # Fisher Information Matrix (FIM) for mu_k is simply sigma_k^{-1}
                fim_mu_inv = np.linalg.inv(self.sigma[i])
                noise_mu = np.random.normal(0, np.sqrt(2 * self.learning_rate), size=self.mu[i].shape)
                self.mu[i] += (self.learning_rate * fim_mu_inv @ gradient_mu / len(assigned_points)) + noise_mu
                # Covariance update using SGLD with preconditioning
                gradient_sigma = np.zeros((self.dim, self.dim))
                for x in assigned_points:
                    diff = (x - self.mu[i]).reshape(-1, 1)
                    gradient_sigma += multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma[i]) * (diff @ diff.T)
                noise_sigma = np.random.normal(0, np.sqrt(2 * self.learning_rate), size=self.sigma[i].shape)
                # FIM for sigma_k is more complex; here we assume a simple structure.
                fim_sigma_inv = 2 * np.linalg.inv(self.sigma[i])  # Simplified assumption
                self.sigma[i] += (self.learning_rate * fim_sigma_inv @ gradient_sigma / len(assigned_points)) + noise_sigma
                # Ensure covariance is symmetric and positive definite
                self.sigma[i] = (self.sigma[i] + self.sigma[i].T) / 2 + epsilon * np.eye(self.dim)
                # Adjust negative eigenvalues if necessary to ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(self.sigma[i])
                eigvals[eigvals < 0] = epsilon  # Set negative eigenvalues to epsilon
                self.sigma[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    def log_likelihood(self, X):
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)

    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
        Utility function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        '''
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        '''
        if (self.dim != 2):
            print("Drawing available only for 2D case.")
            return
        for i in range(self.k):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)
