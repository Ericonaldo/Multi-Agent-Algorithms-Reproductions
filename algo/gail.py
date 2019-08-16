import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel, BaseAgent
from common.buffer import Dataset, Transition
from algo.ppo import PPO

# train_policy_times = 6
# train_discriminator_times = 2
# units = 128

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class Discriminator(BaseModel):
    def __init__(self, sess, env, name=None, agent_id=None, entcoeff=0.001, lr=1e-2, units=128):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id
        self.units = units

        self._loss = None
        self._train_op = None

        self.scope = tf.get_variable_scope().name

        obs_space = env.observation_space[agent_id].shape
        act_space = (env.action_space[agent_id].n,)

        self.expert_obs_phs = tf.placeholder(dtype=tf.float32, shape=(None,) + obs_space)
        self.expert_act_phs = tf.placeholder(dtype=tf.float32, shape=(None,) + act_space)
        self.expert_si_ai = tf.concat([self.expert_obs_phs, self.expert_act_phs], axis=1)

        self.agent_obs_phs = tf.placeholder(dtype=tf.float32, shape=(None,) + obs_space)
        self.agent_act_phs = tf.placeholder(dtype=tf.float32, shape=(None,) + act_space)
        self.agent_si_ai = tf.concat([self.agent_obs_phs, self.agent_act_phs], axis=1)

        with tf.variable_scope('network') as network_scope:
            self.logit_1 = self._construct(input_ph=self.expert_si_ai)
            network_scope.reuse_variables()  # share parameter
            self.logit_2 = self._construct(input_ph=self.agent_si_ai)

        with tf.variable_scope('loss'):
            self.loss_expert = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_1, labels=tf.ones_like(self.logit_1)))
            # loss_expert = tf.reduce_mean(tf.log(tf.nn.sigmoid(self.prob_1)+1e-8))
            self.loss_agent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_2, labels=tf.zeros_like(self.logit_2)))
            # loss_agent = tf.reduce_mean(tf.log(1 - self.prob_2+1e-8))
            logits = tf.concat([self.logit_1, self.logit_2], 0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = -entcoeff*entropy
            self._loss = self.loss_expert + self.loss_agent + entropy_loss
            # self._loss = loss_expert + loss_agent
            # self._loss = loss_expert


            optimizer = tf.train.AdamOptimizer(lr)
            # grad_vars = optimizer.compute_gradients(self._loss, self.trainable_variables)
            # self._train_op = optimizer.apply_gradients(grad_vars)
            self._train_op = optimizer.minimize(self._loss)

        # self.expert_reward = tf.log(tf.nn.sigmoid(self.logit_1)+1e-8)
        self.expert_reward = tf.nn.sigmoid(self.logit_1)*2.0 - 1
        self.reward = tf.nn.sigmoid(self.logit_2)*2.0 - 1
        # self.reward = tf.log(tf.nn.sigmoid(self.logit_2)+1e-8)
        # self.reward = -tf.log(1-tf.nn.sigmoid(self.logit_2)+1e-8)

    def _construct(self, input_ph, norm=False):
        l1 = tf.layers.dense(inputs=input_ph, units=self.units, activation=tf.nn.leaky_relu, name='l1')
        if norm: l1 = tc.layers.layer_norm(l1)
        l2 = tf.layers.dense(inputs=l1, units=self.units, activation=tf.nn.leaky_relu, name='l2')
        if norm: l2 = tc.layers.layer_norm(l2)
        l3 = tf.layers.dense(inputs=l2, units=self.units // 2, activation=tf.nn.leaky_relu, name='l3')
        out = tf.layers.dense(inputs=l3, units=1, activation=tf.identity, name='prob')

        return out

    def train(self, expert_obs, expert_act, agent_obs, agent_act):
        feed_dict = dict()
        feed_dict[self.agent_obs_phs] = agent_obs
        feed_dict[self.agent_act_phs] = agent_act
        feed_dict[self.expert_obs_phs] = expert_obs
        feed_dict[self.expert_act_phs] = expert_act
        loss, e_loss, a_loss, _ = self.sess.run([self._loss, self.loss_expert, self.loss_agent, self._train_op], feed_dict=feed_dict)

        return loss, e_loss, a_loss

    def get_reward(self, obs, act):
        feed_dict = {self.agent_obs_phs:obs, self.agent_act_phs:act}
        reward = self.sess.run(self.reward, feed_dict=feed_dict)
        # print(self.sess.run(tf.nn.sigmoid(self.logit_2), feed_dict=feed_dict))
        
        return reward.reshape((-1,))

    def get_expert_reward(self, obs, act):
        feed_dict = {self.expert_obs_phs:obs, self.expert_act_phs:act}
        reward = self.sess.run(self.expert_reward, feed_dict=feed_dict)
        # print(self.sess.run(tf.nn.sigmoid(self.logit_2), feed_dict=feed_dict))
        
        return reward.reshape((-1,))

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class GAIL(BaseAgent):
    def __init__(self, sess, env, scenario, name, n_agent, batch_size=128, entcoeff=0.001, lr=1e-2, gamma=0.99, p_step=3, d_step=1, units=128):
        super().__init__(env, name)
        # == Initialize ==
        self.sess = sess
        self.n_agent = n_agent
        self.batch_size = batch_size
        self.p_step = p_step
        self.d_step = d_step
        self.gamma = gamma

        self.ppo = [] # ppo agents
        self.disc = [] # discriminators

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            # Policy
            print("initialize policy agents ...")
            for i in range(self.n_agent):
                with tf.variable_scope("agent_{}".format(i)):
                    self.ppo.append(PPO(self.sess, env, name=name, agent_id=i, a_lr=lr, c_lr=lr, gamma=gamma, num_units=units))

            # Discriminator
            print("initialize discriminators ...")
            # Discriminator for each agents
            for i in range(self.n_agent):
                print("initialize discriminator for agent {} ...".format(i))
                with tf.variable_scope("dicriminator_{}".format(i)):
                    self.disc.append(Discriminator(self.sess, env, name=name, agent_id=i, entcoeff=entcoeff, lr=lr, units=units))

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def bc_init(self, init_iter, expert_dataset):
        print("bc initialization for {} iterations".format(init_iter))
        epoch = len(expert_dataset) // self.batch_size
        for _ in range(init_iter): 
            bc_loss = [0.] * self.n_agent
            for __ in range(epoch):
                obs_en, act_en = expert_dataset.sample(self.batch_size)
                for i in range(self.n_agent):
                    bc_loss[i] += self.ppo[i].bc_init(obs_en[i], act_en[i])

            bc_loss = list(map(lambda x:x/epoch, bc_loss))
            print("bc loss in iter {} : {}".format(_, bc_loss))

        return 

    def act(self, obs_set):
        """ Accept a observation list, return action list of all agents. """
        actions = [None for _ in range(self.n_agent)]
        values = [None for _ in range(self.n_agent)]
        for i in range(self.n_agent):
            actions[i], values[i] = self.ppo[i].act(obs_set[i])

        return actions, values

    def save(self, dir_path, iteration, max_to_keep):
        """ Save model
        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        :param max_to_keep: the maximum of keeping models
        """

        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        saver = tf.train.Saver(model_vars, max_to_keep=max_to_keep)
        save_path = saver.save(self.sess, dir_name + "/{}".format(self.name), global_step=iteration)
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

    def train(self, expert_dataset, learning_dataset):
        ## d step
        d_loss = [0.] * self.n_agent
        de_loss = [0.] * self.n_agent
        da_loss = [0.] * self.n_agent

        print("train discriminators for {} times".format(self.d_step))
        for _ in range(self.d_step): # train Discriminators 2 times
            expert_obs, expert_act = expert_dataset.sample(self.batch_size)
            agent_obs, agent_act, _, _, _, _ = learning_dataset.sample(self.batch_size)
            for i in range(self.n_agent):
                loss, e_loss, a_loss = self.disc[i].train(expert_obs[i], expert_act[i], agent_obs[i], agent_act[i])
                d_loss[i] += loss
                de_loss += e_loss
                da_loss += a_loss

        d_loss = [_  / self.d_step for _ in d_loss]
        de_loss = [_  / self.d_step for _ in de_loss]
        da_loss = [_  / self.d_step for _ in da_loss]

        obs_n, act_n, _, _, _, _ = learning_dataset.sample(5)
        for i in range(self.n_agent):
            # print("agent-(s,a): ({},{})".format(obs_n, act_n))
            print("agent-reward-D: {}".format(self.disc[i].get_reward(obs_n[i], act_n[i])))
            obs_en, act_en = expert_dataset.sample(5)
            # print("expert-(s,a): ({},{})".format(obs_en, act_en))
            print("expert-reward-D: {}".format(self.disc[i].get_expert_reward(obs_en[i], act_en[i])))

        # compute rewards and gaes
        learning_dataset.compute(self.gamma, [self.disc[i].get_reward for i in range(self.n_agent)])

        ## p step
        # p_loss = [0.] * self.n_agent
        pa_loss = [0.0] * self.n_agent
        pc_loss = [0.0] * self.n_agent
        print("train policy for {} times".format(self.p_step))
        for _ in range(self.p_step): # train policy 6 times
            obs, actions, rewards, v_preds, v_preds_next, gaes = learning_dataset.sample(self.batch_size)
            for i in range(self.n_agent):
                t_info = self.ppo[i].train(obs[i], actions[i], rewards[i], v_preds_next[i], gaes[i])
                pa_loss[i] += t_info['a_loss']
                pc_loss[i] += t_info['c_loss']

        pa_loss = [_/self.p_step for _ in pa_loss]
        pc_loss = [_/self.p_step for _ in pc_loss]

        return {'d_loss': d_loss, 'de_loss': de_loss,'da_loss': da_loss, 'pa_loss': pa_loss, 'pc_loss': pc_loss} 
