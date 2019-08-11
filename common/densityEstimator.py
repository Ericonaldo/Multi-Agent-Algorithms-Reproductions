import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

class Estimator(object):
    def __init__(self, name, agent_id, lower_dimension):
        self.name = name
        self.agent_id = agent_id
        self.lower_dimension = lower_dimension

    def fit(self, **kwargs):
        raise NotImplementedError

    def prob(self, **kwargs):
        raise NotImplementedError

class KDEEstimator(Estimator):
    def __init__(self, name, agent_id, lower_dimension=None, kernel="gaussian", bandwidth=0.1):
        super().__init__(name, agent_id, lower_dimension)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.pca = None
        self.kde = None

    def fit(self, s, a):
        x = np.concatenate((s, a), axis=-1)

        if (self.lower_dimension is not None) and (np.shape(x)[-1]>self.lower_dimension):
            self.pca = PCA(n_components=self.lower_dimension, whiten=False)
            x = self.pca.fit_transform(x)

        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(x) # x.shape: [None, act+obs]

    def prob(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        if (self.lower_dimension is not None) and (np.shape(x)[-1]>self.lower_dimension):
            x = self.pca.transform(x)

        return np.exp(self.kde.score_samples(x))+1e-4 # x.shape: [None, act+obs]

class VAEEstimator(Estimator):
    def __init__(self, name, agent_id, lower_dimension, kernel="gaussian", bandwidth=0.2):
        super().__init__(name, agent_id, lower_dimension)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.kde = None

    def fit(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        self.kde = kde.KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(x) # x.shape: [None, act+obs]

    def prob(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        return np.exp(self.kde.score_samples(x))+1e-4 # x.shape: [None, act+obs]
