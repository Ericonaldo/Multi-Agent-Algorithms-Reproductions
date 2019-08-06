import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel, Dataset
from algo.maddpg import MADDPG

class Discriminator(BaseModel):
    def __init__(self, sess, state_space, act_space, expert_dataset, lr=1e-2, name=None, agent_id=None):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id

        self._action_space = act_space
        self._observation_space = state_space

        self.expert_dataset = expert_dataset
        self.learning_dataset = None

        self._loss = None
        self._train_op = None

        self.act_dim = flatten(self._action_space)

        self.expert_obs = tf.placeholder(dtype=tf.float32, shape=[None] + self.observation_space)
        self.expert_act = tf.placeholder(dtype=tf.int32, shape=[None])
        self.expert_s_a = tf.concat([self.expert_act, self.expert_obs], axis=1)

        self.agent_obs = tf.placeholder(dtype=tf.float32, shape=[None] + self.observation_space)
        self.agent_act = tf.placeholder(dtype=tf.int32, shape=[None])
        self.agent_s_a = tf.concat([self.expert_act, self.expert_obs], axis=1)

        with tf.variable_scope('network') as network_scope:
            prob_1 = self.construct_network(input_ph=expert_s_a)
            network_scope.reuse_variables()  # share parameter
            prob_2 = self.construct_network(input_ph=agent_s_a)

        with tf.variable_scope('loss'):
            loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
            loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
            loss = loss_expert + loss_agent
            self._loss = -loss
            optimizer = tf.train.AdamOptimizer(lr)
            grad_vars = optimizer.compute_gradients(self._loss, self.e_variables)
            self._train_op = optimizer.apply_gradients(grad_vars)

    def _construct(self, input_ph, norm=False):
        l1 = tf.layers.dense(inputs=input_ph, units=100, activation=tf.nn.leaky_relu, name='l1')
        if norm: l1 = tc.layers.layer_norm(l1)
        l2 = tf.layers.dense(inputs=l1, units=100, activation=tf.nn.leaky_relu, name='l2')
        if norm: l2 = tc.layers.layer_norm(l2)
        l3 = tf.layers.dense(inputs=l2, units=50, activation=tf.nn.leaky_relu, name='l3')
        out = tf.layers.dense(inputs=l3, units=1, activation=tf.sigmoid, name='prob')

        return out

    def train(self, feed_dict):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=feed_dict)

        return loss


class MADiscriminator(object):
    def __init__(self, sess, env, name, expert_dataset, batch_size=512, lr=1e-2, name=None):
        self.name = name
        self.env = env

        self.sess = sess

        self.expert_dataset = expert_dataset
        self.learning_dataset = Dataset()
        self.discriminators = []

        self._loss = None
        self._train_op = None

        # Discriminator for each agents
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize discriminator for agent {} ...".format(i))
                obs_space, act_space = env.observation_space[i].shape, (env.action_space[i].n,)
                with tf.variable_scope("dicriminator_{}_{}".format(name, i)):
                    self.dicriminators.append(Discriminator(self.sess, state_space=obs_space, act_space=act_space, batch_size=batch_size, lr=lr, name=name, agent_id=i))


    def get_reward(self, obs_n, act_n):
        #TODO
        reward = sess.run(feed_dict=xx)
        return reward_n

    def store_data(self, obs_n, act_n):
        """
        Restore state-action dataset of learning agents.
        """
        return self.learning_dataset.push(obs_n, act_n)

    def train(self):
        train_epochs = len(learning_dataset) // self.batch_size
        # shuffle dataset
        self.expert_dataset.shuffle()
        self.learning_dataset.shuffle()
        # train
        for epoch in range(train_epochs):
            feed_dict = dict()
            feed_dict.update()
            loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=feed_dict)

        return loss

class MAIAIL:
    def __init__(self, sess, env, name, n_agent, batch_size=512, p_lr=1e-2, d_lr=1e-2, gamma=0.99, tau=0.01, memory_size=10**4):
        self.name = name
        self.sess = sess
        self.env = env
        self.n_agent = n_agent

        self.maddpg = None # agents 
        self.madcmt = None # discriminators
        self.action_dims = []
        self.batch_size = batch_size
        obs_space, act_space = env.observation_space[i].shape, (env.action_space[i].n,)

        # == Construct Network for Each Agent ==
        # Policy
        with tf.variable_scope(self.name):
            print("initialize policy agents ...".format(i))
            with tf.variable_scope("policy_{}".format(name)):
                policy_name = "maddpg"
                self.maddpg = MADDPG(sess, env, name=policy_name, n_agent, batch_size, actor_lr=p_lr, critic_lr = p_lr, gamma=gamma, tau=tau, memory_size=memory_size)

        # Discriminator
        with tf.variable_scope(self.name):
            print("initialize discriminators ...".format(i))
            discri_name = "madiscriminator"
            self.madcmt = MADiscriminator(self.sess, env, name=discri_name, n_agent, batch_size, lr=d_lr)

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs_set):
        """ Accept a observation list, return action list of all agents. """
        actions = self.maddpg.act(obs_set)

        return actions

    def save(self, dir_path, epoch, max_to_keep):
        """ Save model
        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        :param max_to_keep: the maximum of keeping models
        """

        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars, max_to_keep=max_to_keep)
        save_path = saver.save(self.sess, dir_name + "/{}".format(self.name), global_step=epoch)
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, dir_path, epoch=0):
        """ Load model from local storage.

        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        file_path = None

        dir_name = os.path.join(dir_path, self.name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_name, "{}-{}".format(self.name, epoch))
        saver.restore(self.sess, file_path)

    def train(self, expert_obs, expert_act, agent_obs, agent_act):
        
        # p_loss = [0.] * self.n_agent
        d_loss = [0.] * self.n_agent
        a_loss, c_loss = [0.] * self.n_agent, [0.] * self.n_agent

        for _ in range(2): # train Discriminators 2 times
            d_loss = self.madcmt.train()

        for _ in range(6): # train policy 6 times
            a_loss, c_loss = self.maddpg.train().values()

        return {'d_loss': d_loss, '(policy)a_loss': a_loss, '(policy)c_loss': c_loss} 
