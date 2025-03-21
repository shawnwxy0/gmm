import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

class PSDMap:
    def __init__(self, dim, epsilon=1e-6):
        self.dim = dim
        self.epsilon = epsilon
    def map(self, matrix):
        matrix = (matrix + matrix.T) / 2
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, self.epsilon)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

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
        for i in range(self.num_points):
            posteriors = np.zeros(self.k)
            for j in range(self.k):
                posteriors[j] = self.pi[j] * multivariate_normal.pdf(self.data[i], mean=self.mu[j], cov=self.sigma[j])
            total_posterior = np.sum(posteriors)
            if total_posterior > 0:
                posteriors /= total_posterior
                self.z[i] = np.zeros(self.k)
                chosen_cluster = np.random.multinomial(1, posteriors).argmax()
                self.z[i, chosen_cluster] = 1
            else:
                self.z[i] = np.random.dirichlet(np.ones(self.k))
                
    def sgld_update(self):
        epsilon = 1e-6
        for i in range(self.k):
            assigned_points = self.data[self.z[:, i] > 0]
            if len(assigned_points) > 0:
                gradient_mu = np.zeros(self.dim)
                for x in assigned_points:
                    gradient_mu += multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma[i]) * (x - self.mu[i])
                noise_mu = np.random.normal(0, np.sqrt(2 * self.learning_rate), size=self.mu[i].shape)
                self.mu[i] += (self.learning_rate * self.sigma[i] @ gradient_mu / len(assigned_points)) + noise_mu
                gradient_sigma = np.zeros((self.dim, self.dim))
                for x in assigned_points:
                    diff = (x - self.mu[i]).reshape(-1, 1)
                    gradient_sigma += multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma[i]) * (diff @ diff.T)
                noise_sigma = np.random.normal(0, np.sqrt(2 * self.learning_rate), size=self.sigma[i].shape)
                preconditioned_update_sigma = (2 * self.sigma[i] @ gradient_sigma) / len(assigned_points)
                self.sigma[i] += preconditioned_update_sigma + noise_sigma
                self.sigma[i] = (self.sigma[i] + self.sigma[i].T) / 2 + epsilon * np.eye(self.dim)
                eigvals, eigvecs = np.linalg.eigh(self.sigma[i])
                eigvals[eigvals < 0] = epsilon
                self.sigma[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T
                self.sigma[i] = PSDMap(self.dim).map(self.sigma[i])
                
    def log_likelihood(self, X):
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)
        
    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
        
    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        if self.dim != 2:
            print("Drawing available only for 2D case.")
            return
        for i in range(self.k):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)

def main():
    np.random.seed(42)
    k = 2
    dim = 2
    n_points = 500
    true_mu = np.array([[2, 2], [-2, -2]])
    true_sigma = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.3], [-0.3, 1]]])
    true_pi = np.array([0.6, 0.4])
    data = []
    labels = []
    for i in range(n_points):
        comp = np.random.choice(k, p=true_pi)
        sample = np.random.multivariate_normal(true_mu[comp], true_sigma[comp])
        data.append(sample)
        labels.append(comp)
    data = np.array(data)
    model = GMM(k, dim)
    model.init_gibbs(data)
    n_iter = 50
    for i in range(n_iter):
        model.gibbs_update()
        model.sgld_update()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    model.draw(ax, n_std=2.0, facecolor='none', linewidth=2)
    ax.set_title("GMM Clustering and Learned Gaussian Contours")
    plt.show()

if __name__ == "__main__":
    main()
