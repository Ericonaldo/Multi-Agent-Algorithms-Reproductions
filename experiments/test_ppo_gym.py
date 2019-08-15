import tensorflow as tf
import os
import sys
import gym
import numpy as np
import collections
import random
import tensorflow.contrib.layers as layers
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")
from math import *
import math

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../ma_env/multiagent-particle-envs'))

from common.utils import softmax
from algo.ppo import PPO
from common.buffer import PPODataset

ENV = "CartPole-v0"

EPISODES = 1000
MAX_STEP = 200
GAMMA = 0.95
BATCH_SIZE = 32
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002     # learning rate for critic
S_DIM, A_DIM = 3, 1
EPSILON = 0.2
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10


if __name__ == "__main__":
    env = gym.make(ENV).unwrapped 
    env.seed(1)
    all_ep_r = []
    dataset = PPODataset('ppo', 1, batch_size=32)
    with tf.Session() as sess:
        ppo = PPO(sess, env, 'ppo', 0, a_lr=1e-4, c_lr=1e-4, num_units=32)
        ppo.init()
        for ep in range(EPISODES):
            ep_r = 0
            s = env.reset()
            while True:
                env.render()
                a,v = ppo.act(s)

                act=np.argmax(a)
                s_, r, done, _ = env.step(act)
                dataset.push([s], [a], [r], [v])

                s = s_
                ep_r += r

                if done:
                    dataset.compute(GAMMA)
                    break

            # update ppo
            for epoch in range(6):
                batch_obs_n, batch_act_n, batch_reward_n, batch_values_n, batch_next_values_n, batch_gaes_n = dataset.sample()
                i=0
                ppo.train(batch_obs_n[i], batch_act_n[i], batch_reward_n[i], batch_next_values_n[i], batch_gaes_n[i])
            dataset.clear()

            if ep==0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

            print(
                'Ep: ', ep,
                '|Ep_r: ', ep_r,
            )
    env.close()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("ppo_reward.png")
    plt.show()
