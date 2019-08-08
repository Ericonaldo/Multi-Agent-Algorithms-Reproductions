import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel, Dataset, Transition
from algo.maddpg import MADDPG

train_policy_times = 6
train_discriminator_times = 2
units = 128
class Discriminator(BaseModel):
    def __init__(self, sess, expert_s_a, agent_s_a, lr=1e-2, name=None, agent_id=None):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id

        self._loss = None
        self._train_op = None

        self.scope = tf.get_variable_scope().name

        self.expert_s_a = expert_s_a
        self.agent_s_a = agent_s_a

        with tf.variable_scope('network') as network_scope:
            prob_1 = self._construct(input_ph=expert_s_a)
            network_scope.reuse_variables()  # share parameter
            prob_2 = self._construct(input_ph=agent_s_a)

        with tf.variable_scope('loss'):
            loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
            loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
            loss = loss_expert + loss_agent
            self._loss = -loss
            optimizer = tf.train.AdamOptimizer(lr)
            grad_vars = optimizer.compute_gradients(self._loss, self.trainable_variables)
            self._train_op = optimizer.apply_gradients(grad_vars)

        self.reward = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))

    def _construct(self, input_ph, norm=False):
        l1 = tf.layers.dense(inputs=input_ph, units=units, activation=tf.nn.leaky_relu, name='l1')
        if norm: l1 = tc.layers.layer_norm(l1)
        l2 = tf.layers.dense(inputs=l1, units=units, activation=tf.nn.leaky_relu, name='l2')
        if norm: l2 = tc.layers.layer_norm(l2)
        l3 = tf.layers.dense(inputs=l2, units=units // 2, activation=tf.nn.leaky_relu, name='l3')
        out = tf.layers.dense(inputs=l3, units=1, activation=tf.sigmoid, name='prob')

        return out

    def train(self, feed_dict):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=feed_dict)

        return loss

    def get_reward(self, feed_dict):
        reward = self.sess.run(self.reward, feed_dict=feed_dict)
        
        return reward[0]

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MADiscriminator(object):
    def __init__(self, sess, env, scenario, name, n_agent, expert_dataset, batch_size=512, lr=1e-2, memory_size = 10**4):
        self.name = name
        self.env = env
        self.sess = sess
        self.n_agent = n_agent
        self.expert_dataset = expert_dataset
        self.learning_dataset = Dataset(scenario, n_agent, batch_size)
        self.batch_size = batch_size

        self.discriminators = []
        self._loss = None
        self._train_op = None

        obs_space = [env.observation_space[i].shape for i in range(n_agent)]
        act_space = [(env.action_space[i].n,) for i in range(n_agent)]

        self.expert_obs_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + obs_space[i]) for i in range(n_agent)]
        self.expert_act_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + act_space[i]) for i in range(n_agent)]
        self.expert_obs_n = tf.concat(self.expert_obs_phs_n, axis=1)
        self.expert_act_n = tf.concat(self.expert_act_phs_n, axis=1)
        self.expert_s_a = tf.concat([self.expert_obs_n, self.expert_act_n], axis=1)

        self.agent_obs_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + obs_space[i]) for i in range(n_agent)]
        self.agent_act_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + act_space[i]) for i in range(n_agent)]
        self.agent_obs_n = tf.concat(self.agent_obs_phs_n, axis=1)
        self.agent_act_n = tf.concat(self.agent_act_phs_n, axis=1)
        self.agent_s_a = tf.concat([self.agent_obs_n, self.agent_act_n], axis=1)

        # Discriminator for each agents
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize discriminator for agent {} ...".format(i))
                with tf.variable_scope("dicriminator_{}_{}".format(name, i)):
                    self.discriminators.append(Discriminator(self.sess, self.expert_s_a, self.agent_s_a, lr=lr, name=name, agent_id=i))

    def get_reward(self, obs_n, act_n):
        reward_n = [None] * self.n_agent
        feed_dict = dict()
        feed_dict.update(zip(self.agent_obs_phs_n, obs_n))
        feed_dict.update(zip(self.agent_act_phs_n, act_n))
        for i in range(self.n_agent):
            reward_n[i] = self.discriminators[i].get_reward(feed_dict)
        return reward_n

    def store_dataset(self, obs_n, act_n):
        """
        Restore state-action dataset of learning agents.
        """
        return self.learning_dataset.push(obs_n, act_n)

    def clear_dataset(self):
        self.learning_dataset.clear()

    def train(self):
        loss = [0.0] * self.n_agent
        train_epochs = len(self.learning_dataset) // self.batch_size
        # shuffle dataset
        self.expert_dataset.shuffle()
        self.learning_dataset.shuffle()
        # train
        for epoch in range(train_epochs):
            expert_batch_obs_n, expert_batch_act_n = self.expert_dataset.next()
            agent_batch_obs_n, agent_batch_act_n = self.learning_dataset.next() 

            feed_dict = dict()
            feed_dict.update(zip(self.agent_obs_phs_n, agent_batch_obs_n))
            feed_dict.update(zip(self.agent_act_phs_n, agent_batch_act_n))
            feed_dict.update(zip(self.expert_obs_phs_n, expert_batch_obs_n))
            feed_dict.update(zip(self.expert_act_phs_n, expert_batch_act_n))

            for i in range(self.n_agent):
                loss[i]+=self.discriminators[i].train(feed_dict)
            
        return loss

class MAIAIL:
    def __init__(self, sess, env, scenario, name, n_agent, expert_dataset, batch_size=512, p_lr=1e-2, d_lr=1e-2, gamma=0.99, tau=0.01, memory_size=10**4):
        self.name = name
        self.sess = sess
        self.env = env
        self.n_agent = n_agent
        self.batch_size = batch_size

        self.maddpg = None # agents 
        self.madcmt = None # discriminators

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            # Policy
            print("initialize policy agents ...")
            policy_name = "maddpg"
            self.maddpg = MADDPG(sess, env, policy_name, n_agent, batch_size, actor_lr=p_lr, critic_lr=p_lr, gamma=gamma, tau=tau, memory_size=memory_size)

            # Discriminator
            print("initialize discriminators ...")
            discri_name = "ma-discriminator"
            self.madcmt = MADiscriminator(self.sess, env, scenario=scenario, name=discri_name, n_agent=n_agent, expert_dataset=expert_dataset, batch_size=batch_size, lr=d_lr, memory_size=memory_size)

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

    def load(self, dir_path, epoch=None):
        """ Load model from local storage.

        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        file_path = None

        dir_name = os.path.join(dir_path, self.name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        if epoch is not None:
            file_path = os.path.join(dir_name, "{}-{}".format(self.name, epoch))
            saver.restore(self.sess, file_path)
        else:
            file_path = dir_name
            saver.restore(self.sess, tf.train.latest_checkpoint(file_path))

    def train(self):
        # p_loss = [0.] * self.n_agent
        d_loss = [0.] * self.n_agent

        print("train discriminators for {} times".format(train_discriminator_times))
        for _ in range(train_discriminator_times): # train Discriminators 2 times
            loss = self.madcmt.train()
            d_loss = list(map(operator.add, loss, d_loss))

        d_loss = [_  / train_discriminator_times for _ in d_loss]

        # add reward to maddpg's replay buffer
        buffer_data = self.maddpg.replay_buffer.get_data()
        buffer_obs_n = [None for _ in range(self.n_agent)]
        buffer_act_n = [None for _ in range(self.n_agent)]
        for i in range(self.n_agent):
            buffer_obs_n[i] = list(map(lambda x:x[0], buffer_data[i]))
            buffer_act_n[i] = list(map(lambda x:x[1], buffer_data[i]))
        reward_n = self.madcmt.get_reward(buffer_obs_n, buffer_act_n)
        for i in range(self.n_agent):
            for j in range(len(reward_n[i])):
                buffer_data[i][j] = Transition(buffer_data[i][j][0], buffer_data[i][j][1], buffer_data[i][j][2], reward_n[i][j], buffer_data[i][j][4])
            # tmp = zip(*zip(*buffer_data[i]))
            # buffer_data[i] = list(map(lambda x,y:Transition(x[0], x[1], x[2], y, x[4]), tmp, reward_n[i]))
        self.maddpg.replay_buffer.set_data(buffer_data)

        pa_loss = [0.0] * self.n_agent
        pc_loss = [0.0] * self.n_agent
        for _ in range(train_policy_times): # train policy 6 times
            t_info = self.maddpg.train()
            pa_loss = map(operator.add, pa_loss, t_info['a_loss'])
            pc_loss = map(operator.add, pc_loss, t_info['c_loss'])

        pa_loss = [_/2 for _ in pa_loss]
        pc_loss = [_/2 for _ in pc_loss]
        print("train policy for {} times".format(train_policy_times))
        # clear buffer and dataset
        self.maddpg.clear_buffer()
        self.madcmt.clear_dataset()

        return {'d_loss': d_loss, 'pa_loss': pa_loss, 'pc_loss': pc_loss} 

    def store_data(self, state_n, action_n, next_state_n, done_n):
        flag1 = self.maddpg.store_trans(state_n, action_n, next_state_n, np.zeros((self.n_agent,)), done_n)
        flag2 = self.madcmt.store_dataset(state_n, action_n)

        return (flag1 or flag2)
