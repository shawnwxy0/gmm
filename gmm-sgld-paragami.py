import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

class GMM:
    def __init__(
        self,
        k,
        dim,
        init_mu=None,
        init_sigma=None,
        init_pi=None,
        colors=None,
        learning_rate=0.005,
        decay=0.01,
        update_pi=True,
        burn_in=50,
        max_iter=200,
        cov_reg=1e-6
    ):
        self.k = k
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay = decay
        self.update_pi_flag = update_pi
        self.burn_in = burn_in
        self.max_iter = max_iter
        self.cov_reg = cov_reg
        
        if init_mu is None:
            init_mu = np.random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        
        if init_sigma is None:
            init_sigma = np.array([np.eye(dim) for _ in range(k)])
        self.sigma = init_sigma
        
        if init_pi is None:
            init_pi = np.ones(k) / k
        self.pi = init_pi
        
        if colors is None:
            colors = np.random.rand(k, 3)
        self.colors = colors
        
        self.cov_map = PSDMap(dim)
        
        self.iteration = 0
        self.avg_count = 0
        self.mu_accum = np.zeros_like(self.mu)
        self.sigma_accum = np.zeros_like(self.sigma)
        
        self.mu_avg = np.zeros_like(self.mu)
        self.sigma_avg = np.zeros_like(self.sigma)

    def init_gibbs(self, X):
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))

    def gibbs_update(self):
        for i in range(self.num_points):
            posteriors = np.zeros(self.k)
            for j in range(self.k):
                posteriors[j] = self.pi[j] * multivariate_normal.pdf(
                    self.data[i], mean=self.mu[j], cov=self.sigma[j]
                )
            total = np.sum(posteriors)
            if total > 0:
                posteriors /= total
                chosen = np.random.multinomial(1, posteriors).argmax()
                self.z[i] = np.zeros(self.k)
                self.z[i, chosen] = 1
            else:
                self.z[i] = np.random.dirichlet(np.ones(self.k))
        if self.update_pi_flag:
            cluster_counts = np.sum(self.z, axis=0)
            self.pi = cluster_counts / self.num_points
            self.pi = np.maximum(self.pi, 1e-10)
            self.pi /= np.sum(self.pi)

    def sgld_update(self):
        lr = self.learning_rate / (1.0 + self.decay * self.iteration)
        for i in range(self.k):
            assigned = self.data[self.z[:, i] > 0]
            if len(assigned) == 0:
                continue
            grad_mu = np.zeros(self.dim)
            for x in assigned:
                grad_mu += multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma[i]) * (x - self.mu[i])
            noise_mu = np.random.normal(0, np.sqrt(2 * lr), size=self.mu[i].shape)
            self.mu[i] += (lr * self.sigma[i] @ grad_mu / len(assigned)) + noise_mu
            grad_sigma = np.zeros((self.dim, self.dim))
            for x in assigned:
                diff = (x - self.mu[i]).reshape(-1, 1)
                grad_sigma += multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma[i]) * (diff @ diff.T)
            noise_sigma = np.random.normal(0, np.sqrt(2 * lr), size=self.sigma[i].shape)
            precond_update = (2 * self.sigma[i] @ grad_sigma) / len(assigned)
            self.sigma[i] += precond_update + noise_sigma
            self.sigma[i] += self.cov_reg * np.eye(self.dim)
            self.sigma[i] = self.cov_map.map(self.sigma[i])
        if self.iteration >= self.burn_in:
            self.avg_count += 1
            self.mu_accum += self.mu
            self.sigma_accum += self.sigma
        self.iteration += 1

    def finalize_averages(self):
        if self.avg_count > 0:
            self.mu_avg = self.mu_accum / self.avg_count
            self.sigma_avg = self.sigma_accum / self.avg_count
        else:
            self.mu_avg = np.copy(self.mu)
            self.sigma_avg = np.copy(self.sigma)

    def run(self, X):
        self.init_gibbs(X)
        log_likelihoods = []
        for it in range(self.max_iter):
            self.gibbs_update()
            self.sgld_update()
            ll = self.log_likelihood(X, use_avg=False)
            log_likelihoods.append(ll)
        self.finalize_averages()
        return log_likelihoods

    def log_likelihood(self, X, use_avg=False):
        if use_avg:
            mus = self.mu_avg
            sigmas = self.sigma_avg
        else:
            mus = self.mu
            sigmas = self.sigma
        ll = 0
        for x in X:
            p = 0
            for j in range(self.k):
                p += self.pi[j] * multivariate_normal.pdf(x, mean=mus[j], cov=sigmas[j])
            ll += np.log(max(p, 1e-30))
        return ll

    def plot_gaussian(self, mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, facecolor=facecolor, **kwargs)
        ax.add_patch(ellipse)
    
    def draw(self, ax, n_std=2.0, use_avg=False, facecolor='none', **kwargs):
        if use_avg:
            mus = self.mu_avg
            sigmas = self.sigma_avg
        else:
            mus = self.mu
            sigmas = self.sigma
        for i in range(self.k):
            self.plot_gaussian(mus[i], sigmas[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)

if __name__ == "__main__":
    np.random.seed(42)
    k = 2
    dim = 2
    n_points = 500
    true_mu = np.array([[2, 2], [-2, -2]])
    true_sigma = np.array([
        [[1, 0.5], [0.5, 1]],
        [[1, -0.3], [-0.3, 1]]
    ])
    true_pi = np.array([0.6, 0.4])
    data = []
    labels = []
    for _ in range(n_points):
        comp = np.random.choice(k, p=true_pi)
        sample = np.random.multivariate_normal(true_mu[comp], true_sigma[comp])
        data.append(sample)
        labels.append(comp)
    data = np.array(data)
    labels = np.array(labels)
    gmm = GMM(
        k=k,
        dim=dim,
        learning_rate=0.005,
        decay=0.01,
        update_pi=True,
        burn_in=50,
        max_iter=200,
        cov_reg=1e-6
    )
    log_likelihoods = gmm.run(data)
    final_ll_avg = gmm.log_likelihood(data, use_avg=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    gmm.draw(ax, n_std=2.0, use_avg=False, facecolor='none', linewidth=2)
    ax.set_title('GMM Clustering (Final Iteration) and Gaussian Contours')
    plt.savefig("gmm_clustering_final.png")
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    gmm.draw(ax, n_std=2.0, use_avg=True, facecolor='none', linewidth=2)
    ax.set_title('GMM Clustering (Averaged) and Gaussian Contours')
    plt.savefig("gmm_clustering_avg.png")
    plt.show()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(log_likelihoods, marker='o')
    ax2.set_title('Log Likelihood Progression')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log Likelihood')
    plt.savefig("log_likelihood.png")
    plt.show()
    print("=== True Parameters ===")
    print("True mu:\n", true_mu)
    print("True sigma:\n", true_sigma)
    print("True pi:\n", true_pi)
    print("\n=== Final Iteration Learned Parameters ===")
    print("Final mu:\n", gmm.mu)
    print("Final sigma:\n", gmm.sigma)
    print("Final pi:\n", gmm.pi)
    print("Final log likelihood (current params):", log_likelihoods[-1])
    print("\n=== Averaged Learned Parameters ===")
    print("Averaged mu:\n", gmm.mu_avg)
    print("Averaged sigma:\n", gmm.sigma_avg)
    print("Log likelihood (averaged params):", final_ll_avg)
