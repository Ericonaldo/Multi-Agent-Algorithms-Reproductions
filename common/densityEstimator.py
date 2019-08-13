import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

class Estimator(object):
    def __init__(self, name, agent_id):
        self.name = name
        self.agent_id = agent_id

    def fit(self, **kwargs):
        raise NotImplementedError

    def prob(self, **kwargs):
        raise NotImplementedError

class KDEEstimator(Estimator):
    def __init__(self, name, agent_id, kernel="gaussian", bandwidth=0.1):
        super().__init__(name, agent_id)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.pca = None
        self.kde = None

    def fit(self, x):
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(x) # x.shape: [None, act+obs]

    def prob(self, x):

        return np.exp(self.kde.score_samples(x))+1e-4 # x.shape: [None, dim]


class GMMEstimator(Estimator):
    def __init__(self, name, agent_id):
        super().__init__(name, agent_id)
        self.pca = None
        self.gmm = None

    def fit(self, x):

        self.gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(x) # x.shape: [None, act+obs]

    def prob(self, x):

        return np.exp(self.gmm.score_samples(x))+1e-4 # x.shape: [None, dim]

