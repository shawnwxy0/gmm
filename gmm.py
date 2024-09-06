import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal


class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None, learning_rate=0.01):
        '''
        Define a model with known number of clusters and dimensions.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension
            - init_mu: initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                       (default) equal value to all cluster i.e. 1/k
            - colors: Color valu for plotting each cluster (k, 3)
                      (default) random from uniform[0, 1]
        '''
        self.k = k
        self.dim = dim
        self.learning_rate = learning_rate
        if (init_mu is None):
            init_mu = np.random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        if (init_sigma is None):
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if (init_pi is None):
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi
        if (colors is None):
            colors = np.random.rand(k, 3)
        self.colors = colors

    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))

    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)

    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i])
            self.sigma[i] /= sum_z[i]

    def sgd_update_mu_sigma(self):
        ''' SGD update for means and covariances. '''
        epsilon = 1e-6  # Small regularization term for numerical stability

        for i in range(self.k):
            # Gradient for mu
            gradient_mu = np.zeros(self.dim)
            for j in range(self.num_points):
                z_ij = self.z[j, i]  # Responsibility of cluster i for data point j
                diff = self.data[j] - self.mu[i]  # x_j - mu_i
                # No need for covariance inverse here, just use diff
                gradient_mu += z_ij * diff  # Weighted gradient

            # Update the mean with SGD
            self.mu[i] += self.learning_rate * gradient_mu

            # Gradient for sigma (updating the covariance matrix)
            gradient_sigma = np.zeros((self.dim, self.dim))
            for j in range(self.num_points):
                z_ij = self.z[j, i]  # Responsibility of cluster i for data point j
                diff = (self.data[j] - self.mu[i]).reshape(-1, 1)
                # Covariance update based on the outer product of diff
                gradient_sigma += z_ij * (np.matmul(diff, diff.T))

            # Update sigma using SGD
            self.sigma[i] = (1 - self.learning_rate) * self.sigma[i] + self.learning_rate * gradient_sigma

            # Ensure symmetry (though this should usually be symmetric)
            self.sigma[i] = (self.sigma[i] + self.sigma[i].T) / 2

            # Add small regularization to ensure positive semidefiniteness
            self.sigma[i] += epsilon * np.eye(self.dim)
    # def sgd_update_mu_sigma(self):
    #     ''' SGD update for means and covariances. '''
    #     for i in range(self.k):
    #         # Gradient for mu
    #         gradient_mu = np.zeros(self.dim)
    #         for j in range(self.num_points):
    #             z_ij = self.z[j, i]  # Responsibility of cluster i for data point j
    #             diff = self.data[j] - self.mu[i]  # x_j - mu_i
    #             gradient_mu += z_ij * np.matmul(np.linalg.inv(self.sigma[i]), diff)  # Weighted gradient
    #
    #         # Update the mean with SGD
    #         self.mu[i] += self.learning_rate * gradient_mu
    #
    #         # Gradient for sigma
    #         gradient_sigma = np.zeros((self.dim, self.dim))
    #         for j in range(self.num_points):
    #             z_ij = self.z[j, i]  # Responsibility of cluster i for data point j
    #             diff = (self.data[j] - self.mu[i]).reshape(-1, 1)
    #             gradient_sigma += z_ij * (np.matmul(diff, diff.T) - self.sigma[i])  # Covariance update
    #
    #         # Update sigma using SGD
    #         self.sigma[i] += self.learning_rate * gradient_sigma

    def gibbs_sampler(self):
        '''
        Gibbs sampling step to update the z_i (cluster assignments for each x_i)
        based on the current parameters (mu, sigma, pi).
        '''
        for i in range(self.num_points):
            # Compute the posterior probability for each cluster z_i
            posteriors = np.zeros(self.k)
            for j in range(self.k):
                posteriors[j] = self.pi[j] * multivariate_normal.pdf(self.data[i], mean=self.mu[j], cov=self.sigma[j])
            # Normalize to get valid probabilities
            posteriors /= np.sum(posteriors)

            # Sample z_i from the categorical distribution given by posteriors
            self.z[i] = np.zeros(self.k)  # Reset the vector
            self.z[i, np.random.multinomial(1, posteriors).argmax()] = 1  # Set one-hot encoding

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
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