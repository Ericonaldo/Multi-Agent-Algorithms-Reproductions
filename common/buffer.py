import numpy as np
import operator
from collections import namedtuple
import random
import time
import os

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
    def __init__(self, n_agent, capacity=10**4, batch_size=512):
        super().__init__(capacity, batch_size)

        self.n_agent = n_agent
        self._data = [[] for _ in range(self.n_agent)]
        self._size = 0
        self._next_idx = 0

    def __len__(self):
        return self._size

    def clear(self):
        self._data = [[] for _ in range(self.n_agent)]
        self._next_idx = 0
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

        return self._size <= self.capacity

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data
        len_data = len(data[0])
        if len_data > self._capacity:
            print("the size of data is larger than the capacity of the buffer!")
            exit(0)
        self._size = min(len_data, self._capacity)

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

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self._point+batch_size > self._size:
            batch_obs_n = list(map(lambda x: x[-batch_size:], self._observations))
            batch_act_n = list(map(lambda x: x[-batch_size:], self._actions))
            self._point = 0
        else:
            batch_obs_n = list(map(lambda x: x[self._point:self._point+batch_size], self._observations))
            batch_act_n = list(map(lambda x: x[self._point:self._point+batch_size], self._actions))
        self._point += batch_size

        return batch_obs_n, batch_act_n

    def save_data(self, save_dir):
        save_dir = os.path.join(save_dir, self._name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(self.agent_num):
            obs_path = save_dir + "/expert_obs-{}.csv".format(i)
            act_path = save_dir + "/expert_act-{}.csv".format(i)
            """
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
            """
            with open(obs_path, 'wb') as f_handle:
                np.savetxt(f_handle, self._observations[i], fmt='%s')
            with open(act_path, 'wb') as f_handle:
                np.savetxt(f_handle, self._actions[i], fmt='%s')

        print("saved data to {}".format(save_dir))

    def load_data(self, load_dir):
        load_dir = os.path.join(load_dir, self._name)
        self.clear()
        if not os.path.exists(load_dir):
            print("Load dir {} is not exist!".format(load_dir))
            exit(0)
        for i in range(self.agent_num):
            obs_path = load_dir + "/expert_obs-{}.csv".format(i)
            act_path = load_dir + "/expert_act-{}.csv".format(i)
            self._observations[i] = np.genfromtxt(obs_path)
            self._actions[i] = np.genfromtxt(act_path)
            if len(self.actions[i])>self._capacity:
                self._observations[i] = self._observations[i][:self._capacity]
                self._actions[i] = self._actions[i][:self._capacity]
            if len(self.actions[i]) == 1:
                self._observations[i] = [self._observations[i]]
                self._actions[i] = [self._actions[i]]
        self._size = min(len(self._actions[0]), self._capacity)
        print("loaded data from {}".format(load_dir))

