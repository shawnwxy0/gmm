# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import GMM
from gmm import GMM

def main():


    def gen_data(k=3, dim=2, points_per_cluster=200, lim=[-10, 10]):
        '''
        Generates data from a random mixture of Gaussians in a given range.
        Will also plot the points in case of 2D.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension of generated points
            - points_per_cluster: Number of points to be generated for each cluster
            - lim: Range of mean values
        output:
            - X: Generated points (points_per_cluster*k, dim)
        '''
        x = []
        mean = np.random.rand(k, dim) * (lim[1] - lim[0]) + lim[0]
        for i in range(k):
            cov = np.random.rand(dim, dim + 10)
            cov = np.matmul(cov, cov.T)
            _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
            x += list(_x)
        x = np.array(x)
        if (dim == 2):
            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4)
            ax.autoscale(enable=True)
        return x

    def plot(title):
        '''
        Draw the data points and the fitted mixture model.
        input:
            - title: title of plot and name with which it will be saved.
        '''
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.4)
        ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c=gmm.colors)
        gmm.draw(ax, lw=3)
        ax.set_xlim((-12, 12))
        ax.set_ylim((-12, 12))

        plt.title(title)
        plt.savefig(title.replace(':', '_'))
        plt.show()
        plt.clf()
    X = gen_data(k=3, dim=2, points_per_cluster=1000)
    gmm = GMM(3, 2)
    gmm.init_em(X)
    num_iters = 30
    # Saving log-likelihood
    log_likelihood = [gmm.log_likelihood(X)]
    # plotting
    # plot("Iteration:  0")
    for e in range(num_iters):
        # E-step
        # gmm.e_step()
        # M-step
        gmm.gibbs_sampler()
        gmm.sgd_update_mu_sigma()
        # gmm.m_step()
        # Computing log-likelihood
        log_likelihood.append(gmm.log_likelihood(X))
        print("Iteration: {}, log-likelihood: {:.4f}".format(e + 1, log_likelihood[-1]))
        # plotting
        # plot(title="Iteration: " + str(e + 1))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
