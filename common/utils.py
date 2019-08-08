import numpy as np
import operator
from functools import reduce
from collections import namedtuple
import random
import time
import os


try:
    from scipy.misc import imresize
except:
    import cv2
    imresize = cv2.resize


Transition = namedtuple("Transition", "state, action, next_state, reward, done")


class Buffer:
    def __init__(self, capacity, batch_size):
        self._data = []
        self._capacity = capacity
        self._flag = 0
        self.batch_size = batch_size

    def __len__(self):
        return len(self._data)

    def push(self, *args):
        """args: state, action, next_state, reward, done"""

        if len(self._data) < self._capacity:
            self._data.append(None)

        self._data[self._flag] = Transition(*args)
        self._flag = (self._flag + 1) % self._capacity

    def sample(self):
        if len(self._data) < self.batch_size:
            return None

        samples = random.sample(self._data, batch_size)

        return Transition(*zip(*samples))

    @property
    def capacity(self):
        return self._capacity

class BunchBuffer(Buffer):
    def __init__(self, n_agent, capacity, batch_size):
        super().__init__(capacity, batch_size)

        self.n_agent = n_agent
        self._data = [[] for _ in range(self.n_agent)]
        self._size = 0
        self._next_idx = 0

    def __len__(self):
        return self._size

    def clear(self):
        self._data = [[] for _ in range(self.n_agent)]
        self._size = 0

    def push(self, *args):
        """ Append coming transition into inner dataset

        :param args: ordered tuple (state, action, next_state, reward, done)
        """

        for i, (state, action, next_state, reward, done) in enumerate(zip(*args)):
            if self._next_idx >= self._size:
                self._data[i].append(Transition(state, action, next_state, reward, done))
            else:
                self._data[i][self._next_idx] = Transition(state, action, next_state, reward, done)

        self._next_idx = (self._next_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

        return self._size < self.capacity

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def sample(self):
        """ Sample mini-batch data with given size

        :param batch_size: int, indicates the size of mini-batch
        :return: a list of batch data for N agents
        """

        if self._size < self.batch_size:
            return None

        samples = [None for _ in range(self.n_agent)]

        random.seed(time.time())
        ind = random.sample(range(self._size), self.batch_size)
        for i in range(self.n_agent):
            tmp = [self._data[i][j] for j in ind]
            samples[i] = Transition(*zip(*tmp))

        return samples

class Dataset(object):
    def __init__(self, name, agent_num, batch_size=512, capacity=65536):
        self.agent_num = agent_num
        self._name = name
        self._capacity = capacity
        self._size = 0
        self._observations = [[] for _ in range(agent_num)]
        self._actions = [[] for _ in range(agent_num)]
        self._point = 0
        self.batch_size = batch_size

    def __len__(self):
        return self._size

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    def clear(self):
        self._size = 0
        self._observations = [[] for _ in range(self.agent_num)]
        self._actions = [[] for _ in range(self.agent_num)]
        self._point = 0

    def push(self, obs_n, act_n):
        if self._size<self._capacity:
            self._observations = list(map(lambda x, y: y + [x], obs_n, self._observations))
            self._actions = list(map(lambda x, y: y + [x], act_n, self._actions))
        
        self._size = min(self._size+1, self._capacity)
        
        return self._size<self._capacity

    def shuffle(self):
        indices = list(range(self._size)) 
        random.shuffle(indices)
        # tmp_o = [[] for _ in range(self.agent_num)]
        # tmp_a = [[] for _ in range(self.agent_num)]
        for i in range(self.agent_num):
            self._observations[i] = [self._observations[i][j] for j in indices]
            self._actions[i] = [self._actions[i][j] for j in indices]
        # self._observations = tmp_o
        # self._actions = tmp_a
        self._point = 0

    def next(self):
        batch_obs_n = list(map(lambda x: x[self._point:self._point+self.batch_size], self._observations))
        batch_act_n = list(map(lambda x: x[self._point:self._point+self.batch_size], self._actions))

        return batch_obs_n, batch_act_n

    def save_data(self, save_dir):
        save_dir = os.path.join(save_dir, self._name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(self.agent_num):
            obs_path = save_dir + "/expert_obs-{}.csv".format(i)
            act_path = save_dir + "/expert_act-{}.csv".format(i)
            try:
                with open(obs_path, 'ab') as f_handle:
                    np.savetxt(f_handle, self._observations[i], fmt='%s')
                with open(act_path, 'ab') as f_handle:
                    np.savetxt(f_handle, self._actions[i], fmt='%s')
            except FileNotFoundError:
                with open(obs_path, 'wb') as f_handle:
                    np.savetxt(f_handle, self._observations[i], fmt='%s')
                with open(act_path, 'wb') as f_handle:
                    np.savetxt(f_handle, self._actions[i], fmt='%s')
        print("saved data to {}".format(save_dir))

    def load_data(self, load_dir):
        load_dir = os.path.join(load_dir, self._name)
        self.clear()
        if not os.path.exists(load_dir):
            print("Load dir is not exist!")
            exit(0)
        for i in range(self.agent_num):
            obs_path = load_dir + "/expert_obs-{}.csv".format(i)
            act_path = load_dir + "/expert_act-{}.csv".format(i)
            self._observations[i] = np.genfromtxt(obs_path)
            self._actions[i] = np.genfromtxt(act_path)
        self._size = len(self._actions[0])
        print("loaded data from {}".format(load_dir))


class BaseModel(object):
    def __init__(self, name):
        self.name = name
        self._saver = None
        self._sess = None

    def train(self, **kwargs):
        raise NotImplementedError

    def _construct(self, **kwargs):
        raise NotImplementedError


class Meta(object):
    obs = None
    action = None
    terminate = None
    obs_next = None
    reward = None


class Record(object):
    def __init__(self, n):
        self.loss = [0. for _ in range(n)]
        self.reward = [0. for _ in range(n)]


class Matrix(object):
    def __init__(self):
        self.data = []  # dims: (id, memory_size, inner_shape)
        self.dims = []

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.data[item[1]][item[0]]  # item: (memory, id) -> (id, memory) -> (item[1], item[0])
        else:
            return [self.data[i][item] for i in range(len(self.data))]  # item at here is memory index

    def __setitem__(self, key, value):
        if isinstance(key, int):
            for i, v in enumerate(value):
                self.data[i][key] = v
        else:
            raise Exception("Cannot accept this index type:", type(key), "You can make key as int.")

    @property
    def shape(self):
        return self.dims[0]


class ObsMatrix(Matrix):
    def __init__(self, size, obs_spaces, dtype=np.float32):
        super().__init__()
        self.dims = [(size,) + obs_spaces[i].shape for i in range(len(obs_spaces))]

        for i in range(len(obs_spaces)):
            self.data.append(np.zeros((size,) + obs_spaces[i].shape, dtype=dtype))


class ActionMatrix(Matrix):
    def __init__(self, size, action_spaces, dtype=np.float32):
        super().__init__()
        self.dims = [(size, action_spaces[i].n) for i in range(len(action_spaces))]


def flatten(_tuple):
    return reduce(operator.mul, _tuple)


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
    if len(X.shape) == 1: p = p.flatten()

    return p
