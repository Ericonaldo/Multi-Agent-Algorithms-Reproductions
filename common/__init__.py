from densityEstimator import KDEEstimator, GMMEstimator
from dimensioner import PCADimensioner, FWDimensioner, AEDimensioner


DIMENSIONER = {
    "pca": PCADimensioner,
    "ae": AEDimensioner,
    "fw": FWDimensioner
}


ESTIMATOR = {
    "kde": KDEEstimator,
    "gmm": GMMEstimator
}
