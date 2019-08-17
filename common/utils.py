import numpy as np
import operator
from functools import reduce

try:
    from scipy.misc import imresize
except:
    import cv2
    imresize = cv2.resize

class BaseAgent(object):
    """
    A random agent.
    """
    def __init__(self, env, name):
        self.env = env
        self.name = name

    def act(self, obs_n):
        act_n = [softmax(np.random.random(self.env.action_space[i].n)) for i in range(self.env.n)]

        return act_n

class BaseModel(object):
    def __init__(self, name):
        self.name = name
        self._saver = None
        self._sess = None

    def train(self, **kwargs):
        raise NotImplementedError

    def _construct(self, **kwargs):
        raise NotImplementedError


def flatten(_tuple):
    return reduce(operator.mul, _tuple)

# ------ scheduler --------
def constant(p):
    return 1

def linear(p):
    return 1-p


def middle_drop(p):
    eps = 0.75
    if 1-p<eps:
        return eps*0.1
    return 1-p

def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1-p<eps:
        return eps
    return 1-p

def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1-p<eps1:
        if 1-p<eps2:
            return eps2*0.5
        return eps1*0.1
    return 1-p

def softmax(X, theta=1.0, axis=None):
    """ Compute the softmax of each element along an axis of X.

    :param X: ND-Array. Probably should be floats.
    :param theta: float, used as a multiplier prior to exponentiation. Default = 1.0
    :param axis: axis to compute values along. Default is the first non-singleton axis.
    :return: an array the same size as X. The result will sum to 1.0 along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(np.shape(X)) == 1: p = p.flatten()

    return p

schedules = {
    'linear':linear,
    'constant':constant,
    'double_linear_con':double_linear_con,
    'middle_drop':middle_drop,
    'double_middle_drop':double_middle_drop
}

class Scheduler(object):
    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v*self.schedule(self.n/self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps/self.nvalues)
