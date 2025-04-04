import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from paragami import PatternDict, NumericVectorPattern, NumericArrayPattern


class SGLD_GMM:
    def __init__(self, num_samples=1000, num_clusters=3, dim=2, eta=0.01, num_iterations=5000, burn_in=1000, k=1):
        np.random.seed(42)
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.dim = dim
        self.eta = eta
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.k = k

        self.true_means = np.array([[2, 2], [-2, -2], [2, -2]])
        self.true_covs = np.array([np.eye(dim) * 0.5 for _ in range(num_clusters)])
        self.true_weights = np.array([0.4, 0.4, 0.2])

        self.z = np.random.choice(num_clusters, size=num_samples, p=self.true_weights)
        self.x = np.array([np.random.multivariate_normal(self.true_means[k], self.true_covs[k]) for k in self.z])

        self.means = np.random.randn(num_clusters, dim)
        self.cov_update_pattern = self._get_covariance_update_pattern(dim, k)
        self.covs = [self.cov_update_pattern.flatten({
            'log_sigma': np.ones(dim),
            'low_rank': np.random.randn(dim, k)
        }) for _ in range(num_clusters)]
        self.weights = np.ones(num_clusters) / num_clusters

        self.samples_means = []
        self.samples_covs = []
        self.mixing_times = []

    def _get_covariance_update_pattern(self, dim, k):
        cov_pattern = PatternDict(free_default=True)
        cov_pattern['log_sigma'] = NumericVectorPattern(length=dim)
        cov_pattern['low_rank'] = NumericArrayPattern(shape=(dim, k))
        return cov_pattern

    def _get_full_covariance(self, param_dict):
        diag = np.diag(np.exp(param_dict['log_sigma'])**2)
        low_rank = param_dict['low_rank']
        return diag + low_rank @ low_rank.T

    def run(self):
        for t in range(self.num_iterations):
            self._gibbs_sampling()
            self._sgld_update()
            self._track_statistics(t)

    def _gibbs_sampling(self):
        responsibilities = np.zeros((self.num_clusters, self.num_samples))
        for k in range(self.num_clusters):
            param_dict = self.cov_update_pattern.fold(self.covs[k])
            cov_k = self._get_full_covariance(param_dict)
            responsibilities[k] = self.weights[k] * multivariate_normal.pdf(self.x, self.means[k], cov_k)
        responsibilities /= responsibilities.sum(axis=0)
        self.z = np.array([np.random.choice(self.num_clusters, p=responsibilities[:, i]) for i in range(self.num_samples)])

    def _sgld_update(self):
        for k in range(self.num_clusters):
            cluster_points = self.x[self.z == k]
            if len(cluster_points) > 0:
                param_dict = self.cov_update_pattern.fold(self.covs[k])
                grad = np.linalg.solve(np.diag(np.exp(2 * param_dict['log_sigma'])) + 1e-6,
                                       (cluster_points.mean(axis=0) - self.means[k]))
                self.means[k] += self.eta * grad + np.sqrt(2 * self.eta) * np.random.randn(self.dim)

                param_dict['log_sigma'] += self.eta * np.random.randn(self.dim)
                param_dict['low_rank'] += self.eta * np.random.randn(self.dim, self.k)
                param_dict['log_sigma'] = np.clip(param_dict['log_sigma'], -10, 10)
                self.covs[k] = self.cov_update_pattern.flatten(param_dict)

    def _track_statistics(self, t):
        if t >= self.burn_in:
            self.samples_means.append(self.means.copy())
            self.samples_covs.append(self.covs.copy())
        if len(self.samples_means) >= 100 and t % 100 == 0:
            mixing_time = np.linalg.norm(self.means - self.samples_means[-100], ord='fro')
            self.mixing_times.append(mixing_time)

    def plot_results(self):
        samples_means = np.array(self.samples_means)
        mixing_times = np.array(self.mixing_times)

        plt.figure(figsize=(8, 6))
        plt.hist(samples_means[:, :, 0].flatten(), bins=30, alpha=0.5, label='Mean (Dim 1)')
        plt.hist(samples_means[:, :, 1].flatten(), bins=30, alpha=0.5, label='Mean (Dim 2)')
        plt.legend()
        plt.title("Empirical Density of Global Parameters")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(mixing_times, label="Mixing Time")
        plt.xlabel("Iteration (x100)")
        plt.ylabel("Frobenius Norm Difference")
        plt.title("Mixing Time of Global Parameter and Latent Variable")
        plt.legend()
        plt.show()

        # -------- Plot: Estimated vs True Means --------
        plt.figure(figsize=(8, 6))
        plt.scatter(self.true_means[:, 0], self.true_means[:, 1], c='red', marker='x', s=100, label='True Means')
        plt.scatter(self.means[:, 0], self.means[:, 1], c='blue', marker='o', s=100, label='Estimated Means')
        plt.legend()
        plt.title("True vs Estimated Cluster Means")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.show()

        # -------- Plot: Contour of True vs Estimated GMM --------
        x_min, x_max = self.x[:, 0].min() - 2, self.x[:, 0].max() + 2
        y_min, y_max = self.x[:, 1].min() - 2, self.x[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # True GMM Density
        true_density = np.zeros(len(grid))
        for k in range(self.num_clusters):
            true_density += self.true_weights[k] * multivariate_normal.pdf(grid, self.true_means[k], self.true_covs[k])
        true_density = true_density.reshape(xx.shape)

        # Estimated GMM Density
        est_density = np.zeros(len(grid))
        for k in range(self.num_clusters):
            param_dict = self.cov_update_pattern.fold(self.covs[k])
            cov_k = self._get_full_covariance(param_dict)
            est_density += self.weights[k] * multivariate_normal.pdf(grid, self.means[k], cov_k)
        est_density = est_density.reshape(xx.shape)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, true_density, levels=20, cmap='Reds')
        plt.title("True GMM Density")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.scatter(self.true_means[:, 0], self.true_means[:, 1], c='red', marker='x')

        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, est_density, levels=20, cmap='Blues')
        plt.title("Estimated GMM Density")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.scatter(self.means[:, 0], self.means[:, 1], c='blue', marker='o')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    model = SGLD_GMM()
    model.run()
    model.plot_results()
