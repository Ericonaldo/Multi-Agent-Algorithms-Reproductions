import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel, BaseAgent
from common.buffer import Dataset, Transition
from algo.maddpg import MADDPG

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
    def __init__(self, sess, expert_si_an, agent_si_an, alpha_i, entcoeff=0.001, lr=1e-2, name=None, agent_id=None, units=128, e_w=1, a_w=1):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id
        self.units = units

        self._loss = None
        self._train_op = None

        self.scope = tf.get_variable_scope().name

        self.expert_si_an = expert_si_an
        self.agent_si_an = agent_si_an
        self.alpha_i = alpha_i

        with tf.variable_scope('network') as network_scope:
            self.logit_1 = self._construct(input_ph=expert_si_an)
            network_scope.reuse_variables()  # share parameter
            self.logit_2 = self._construct(input_ph=agent_si_an)

        with tf.variable_scope('loss'):
            self.loss_expert = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_1, labels=tf.ones_like(self.logit_1)))
            # loss_expert = tf.reduce_mean(tf.log(tf.nn.sigmoid(self.prob_1)+1e-8))
            self.loss_agent = tf.reduce_mean(tf.expand_dims(self.alpha_i,-1) * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_2, labels=tf.zeros_like(self.logit_2)))
            # loss_agent = tf.reduce_mean(tf.log(1 - self.prob_2+1e-8))
            logits = tf.concat([self.logit_1, self.logit_2], 0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = -entcoeff*entropy
            self._loss = e_w * self.loss_expert + a_w * self.loss_agent + entropy_loss
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
        # self.reward = tf.expand_dims(self.alpha_i,-1) * tf.log(tf.nn.sigmoid(self.logit_2)+1e-8)
        # self.reward = -tf.expand_dims(self.alpha_i,-1) * tf.log(1-tf.nn.sigmoid(self.logit_2)+1e-8)
        # self.reward = tf.expand_dims(self.alpha_i,-1) * tf.nn.sigmoid(self.logit_2)*2.0-1

    def _construct(self, input_ph, norm=False):
        l1 = tf.layers.dense(inputs=input_ph, units=self.units, activation=tf.nn.leaky_relu, name='l1')
        if norm: l1 = tc.layers.layer_norm(l1)
        l2 = tf.layers.dense(inputs=l1, units=self.units, activation=tf.nn.leaky_relu, name='l2')
        if norm: l2 = tc.layers.layer_norm(l2)
        l3 = tf.layers.dense(inputs=l2, units=self.units // 2, activation=tf.nn.leaky_relu, name='l3')
        out = tf.layers.dense(inputs=l3, units=1, activation=tf.identity, name='prob')

        return out

    def train(self, feed_dict):
        loss, e_loss, a_loss, _ = self.sess.run([self._loss, self.loss_expert, self.loss_agent, self._train_op], feed_dict=feed_dict)

        return loss, e_loss, a_loss

    def get_reward(self, feed_dict):
        reward = self.sess.run(self.reward, feed_dict=feed_dict)
        # print(self.sess.run(tf.nn.sigmoid(self.logit_2), feed_dict=feed_dict))
        
        return reward.reshape((-1,))

    def get_expert_reward(self, feed_dict):
        reward = self.sess.run(self.expert_reward, feed_dict=feed_dict)
        # print(self.sess.run(tf.nn.sigmoid(self.logit_2), feed_dict=feed_dict))
        
        return reward.reshape((-1,))

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class MADiscriminator(object):
    def __init__(self, sess, env, scenario, name, n_agent, expert_dataset, batch_size=512, entcoeff=0.001, lr=1e-2, memory_size = 10**4, units=128, lbd=1, e_w=1, a_w=1, lower_dimension=None):
        self.name = name
        self.env = env
        self.sess = sess
        self.n_agent = n_agent
        self.expert_dataset = expert_dataset
        self.learning_dataset = Dataset(scenario, n_agent, batch_size)
        self.batch_size = batch_size
        self.lbd = lbd

        self.discriminators = []
        self._loss = None
        self._train_op = None

        obs_space = [env.observation_space[i].shape for i in range(n_agent)]
        act_space = [(env.action_space[i].n,) for i in range(n_agent)]

        
        self.expert_obs_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + obs_space[i]) for i in range(n_agent)]
        self.expert_act_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + act_space[i]) for i in range(n_agent)]
        self.expert_obs_n = self.expert_obs_phs_n
        self.expert_act_n = tf.concat(self.expert_act_phs_n, axis=1)
        self.expert_si_an = [tf.concat([self.expert_obs_n[i], self.expert_act_n], axis=1) for i in range(n_agent)]

        self.agent_obs_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + obs_space[i]) for i in range(n_agent)]
        self.agent_act_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + act_space[i]) for i in range(n_agent)]
        self.agent_obs_n = self.agent_obs_phs_n
        self.agent_act_n = tf.concat(self.agent_act_phs_n, axis=1)
        self.agent_si_an = [tf.concat([self.agent_obs_n[i], self.agent_act_n], axis=1) for i in range(n_agent)]
        
        """
        if lower_dimension is None:
            lower_dimension = [(np.sum((obs_space[i],act_space[i])), ) for i in range(n_agent)] 
        else:
            lower_dimension = [(lower_dimension,) for _ in range(n_agent)]

        self.expert_s_a_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + lower_dimension[i]) for i in range(n_agent)]
        self.agent_s_a_phs_n = [tf.placeholder(dtype=tf.float32, shape=(None,) + lower_dimension[i]) for i in range(n_agent)]
        self.expert_s_a_n = tf.concat(self.expert_s_a_phs_n, axis=1)
        self.agent_s_a_n = tf.concat(self.agent_s_a_phs_n, axis=1)
        """

        self.alpha_n = [tf.placeholder(dtype=tf.float32, shape=(None,)) for i in range(n_agent)]

        # Discriminator for each agents
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize discriminator for agent {} ...".format(i))
                with tf.variable_scope("dicriminator_{}".format(i)):
                    self.discriminators.append(Discriminator(self.sess, self.expert_si_an[i], self.agent_si_an[i], self.alpha_n[i], lr=lr, name=name, agent_id=i, units=units, e_w=e_w, a_w=a_w))

    def get_reward(self, obs_n, act_n, expert_pdf, agent_pdf):
        reward_n = [None] * self.n_agent
        sa_n = [None for _ in range(self.n_agent)]
        for i in range(self.n_agent):
            sa_n[i] = agent_pdf[i].pca_transform(obs_n[i], act_n[i])

        feed_dict = dict()
        feed_dict.update(zip(self.agent_obs_phs_n, obs_n))
        feed_dict.update(zip(self.agent_act_phs_n, act_n))
        """
        feed_dict.update(zip(self.agent_s_a_phs_n, sa_n))
        """
        for i in range(self.n_agent):
            rho_1 = 1.0 * agent_pdf[i].prob(obs_n[i], act_n[i]) / expert_pdf[i].prob(obs_n[i], act_n[i])
            rho_2 = 1.0
            for j in range(self.n_agent):
                rho_1 *= expert_pdf[j].prob(obs_n[j], act_n[j])
                rho_2 *= agent_pdf[j].prob(obs_n[j], act_n[j])
            alpha = self.lbd * np.clip(rho_1 / rho_2, 1e-1, 1)
            # alpha = self.lbd * np.minimum(rho_1 / rho_2, 1)
            # alpha = self.lbd * np.maximum(rho_1 / rho_2, 1e-1)
            # alpha = self.lbd * 1.0 * rho_1 / rho_2
            # print("rho_1:{} | rho_2:{} | alpha:{}".format(rho_1, rho_2, alpha))
            feed_dict.update({self.alpha_n[i]: alpha})
            reward_n[i] = self.discriminators[i].get_reward(feed_dict)
        return reward_n

    def get_expert_reward(self, obs_n, act_n, expert_pdf, agent_pdf):
        reward_n = [None] * self.n_agent
        sa_n = [None for _ in range(self.n_agent)]
        for i in range(self.n_agent):
            sa_n[i] = expert_pdf[i].pca_transform(obs_n[i], act_n[i])

        feed_dict = dict()
        feed_dict.update(zip(self.expert_obs_phs_n, obs_n))
        feed_dict.update(zip(self.expert_act_phs_n, act_n))
        """
        feed_dict.update(zip(self.expert_s_a_phs_n, sa_n))
        """
        for i in range(self.n_agent):
            reward_n[i] = self.discriminators[i].get_expert_reward(feed_dict)
        return reward_n

    def store_dataset(self, obs_n, act_n):
        """
        Restore state-action dataset of learning agents.
        """
        return self.learning_dataset.push(obs_n, act_n)

    def clear_dataset(self):
        self.learning_dataset.clear()

    def train(self, expert_pdf, agent_pdf):
        loss = [0.0] * self.n_agent
        a_loss = [0.0] * self.n_agent
        e_loss = [0.0] * self.n_agent
        # train_epochs = len(self.learning_dataset) // self.batch_size
        train_epochs = 1
        # shuffle dataset
        self.expert_dataset.shuffle()
        self.learning_dataset.shuffle()
        # train
        for epoch in range(train_epochs):
            expert_batch_obs_n, expert_batch_act_n = self.expert_dataset.next()
            agent_batch_obs_n, agent_batch_act_n = self.learning_dataset.next() 

            expert_batch_sa_n = [None for _ in range(self.n_agent)]
            agent_batch_sa_n = [None for _ in range(self.n_agent)]
            for i in range(self.n_agent):
                expert_batch_sa_n[i] = expert_pdf[i].pca_transform(expert_batch_obs_n[i], expert_batch_act_n[i])
                agent_batch_sa_n[i] = agent_pdf[i].pca_transform(agent_batch_obs_n[i], agent_batch_act_n[i])

            feed_dict = dict()
            """
            feed_dict.update(zip(self.expert_s_a_phs_n, expert_batch_sa_n))
            feed_dict.update(zip(self.agent_s_a_phs_n, agent_batch_sa_n))
            """
            feed_dict.update(zip(self.agent_obs_phs_n, agent_batch_obs_n))
            feed_dict.update(zip(self.agent_act_phs_n, agent_batch_act_n))
            feed_dict.update(zip(self.expert_obs_phs_n, expert_batch_obs_n))
            feed_dict.update(zip(self.expert_act_phs_n, expert_batch_act_n))

            for i in range(self.n_agent):
                rho_1 = 1.0 * agent_pdf[i].prob(agent_batch_obs_n[i], agent_batch_act_n[i]) / expert_pdf[i].prob(agent_batch_obs_n[i], agent_batch_act_n[i])
                rho_2 = 1.0
                for j in range(self.n_agent):
                    rho_1 *= expert_pdf[j].prob(agent_batch_obs_n[j], agent_batch_act_n[j])
                    rho_2 *= agent_pdf[j].prob(agent_batch_obs_n[j], agent_batch_act_n[j])
                print("alpha mean:{} var:{}".format(np.mean(rho_1 / rho_2), np.var(rho_1 / rho_2)))
                alpha = self.lbd * np.clip(rho_1 / rho_2, 1e-1, 1)
                # alpha = self.lbd * np.minimum(rho_1 / rho_2, 1)
                # alpha = self.lbd * np.maximum(rho_1 / rho_2, 1e-1)
                # alpha = self.lbd * 1.0 * rho_1 / rho_2
                # print("rho_1:{} | rho_2:{} | alpha:{}".format(rho_1, rho_2, alpha))
                feed_dict.update({self.alpha_n[i]: alpha})
                l, le, la=self.discriminators[i].train(feed_dict)
                loss[i]+=l
                e_loss[i]+=le
                a_loss[i]+=la
            
        return loss, e_loss, a_loss

class MAIAIL(BaseAgent):
    def __init__(self, sess, env, scenario, name, n_agent, expert_dataset, batch_size=512, entcoeff=0.001, lr=1e-2, gamma=0.99, tau=0.01, memory_size=10**4, p_step=3, d_step=1, units=128, lbd=1, lower_dimension=None):
        super().__init__(env, name)
        # == Initialize ==
        self.sess = sess
        self.n_agent = n_agent
        self.batch_size = batch_size
        self.expert_dataset = expert_dataset
        self.p_step = p_step
        self.d_step = d_step
        self.lbd = lbd

        self.maddpg = None # agents 
        self.madcmt = None # discriminators

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            # Policy
            print("initialize policy agents ...")
            policy_name = "maddpg"
            self.maddpg = MADDPG(sess, env, policy_name, n_agent, batch_size, actor_lr=lr, critic_lr=lr, gamma=gamma, tau=tau, memory_size=memory_size)

            # Discriminator
            print("initialize discriminators ...")
            discri_name = "ma-discriminator"
            self.madcmt = MADiscriminator(self.sess, env, scenario=scenario, name=discri_name, n_agent=n_agent, expert_dataset=expert_dataset, batch_size=batch_size, entcoeff=entcoeff, lr=lr, memory_size=memory_size, units=units, lbd=lbd, lower_dimension=lower_dimension)

        self.learning_dataset = self.madcmt.learning_dataset

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs_set):
        """ Accept a observation list, return action list of all agents. """
        actions = self.maddpg.act(obs_set)

        return actions

    def save(self, dir_path, iteration, max_to_keep):
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

    def train(self, expert_pdf, agent_pdf):

        ## d step
        d_loss = [0.] * self.n_agent
        de_loss = [0.] * self.n_agent
        da_loss = [0.] * self.n_agent

        print("train discriminators for {} times".format(self.d_step))
        for _ in range(self.d_step): # train Discriminators 2 times
            loss, e_loss, a_loss = self.madcmt.train(expert_pdf, agent_pdf)
            d_loss = list(map(operator.add, loss, d_loss))
            de_loss = list(map(operator.add, loss, e_loss))
            da_loss = list(map(operator.add, loss, a_loss))

        d_loss = [_  / self.d_step for _ in d_loss]
        de_loss = [_  / self.d_step for _ in de_loss]
        da_loss = [_  / self.d_step for _ in da_loss]

        obs_n, act_n = self.learning_dataset.next(5)
        # print("agent-(s,a): ({},{})".format(obs_n, act_n))
        print("agent-reward-D: {}".format(self.madcmt.get_reward(obs_n, act_n, expert_pdf, agent_pdf)))
        obs_en, act_en = self.expert_dataset.next(5)
        # print("expert-(s,a): ({},{})".format(obs_en, act_en))
        print("expert-reward-D: {}".format(self.madcmt.get_expert_reward(obs_en, act_en, expert_pdf, agent_pdf)))

        """
        # add reward to maddpg's replay buffer
        buffer_data = self.maddpg.replay_buffer.get_data()
        buffer_obs_n = [None for _ in range(self.n_agent)]
        buffer_act_n = [None for _ in range(self.n_agent)]
        for i in range(self.n_agent):
            buffer_obs_n[i] = list(map(lambda x:x[0], buffer_data[i]))
            buffer_act_n[i] = list(map(lambda x:x[1], buffer_data[i]))
        reward_n = self.madcmt.get_reward(buffer_obs_n, buffer_act_n, expert_pdf, agent_pdf)

        for i in range(self.n_agent):
            for j in range(len(reward_n[i])):
                # print("raw-reward:{} | new-reward: {}".format(buffer_data[i][j][3], reward_n[i][j]))
                buffer_data[i][j] = Transition(buffer_data[i][j][0], buffer_data[i][j][1], buffer_data[i][j][2], reward_n[i][j], buffer_data[i][j][4])
            # tmp = zip(*zip(*buffer_data[i]))
            # buffer_data[i] = list(map(lambda x,y:Transition(x[0], x[1], x[2], y, x[4]), tmp, reward_n[i]))
        self.maddpg.replay_buffer.set_data(buffer_data)
        """

        ## p step
        # p_loss = [0.] * self.n_agent
        pa_loss = [0.0] * self.n_agent
        pc_loss = [0.0] * self.n_agent
        print("train policy for {} times".format(self.p_step))
        for _ in range(self.p_step): # train policy 6 times
            t_info = self.maddpg.train(reward_func=self.madcmt.get_reward, pdfs=[expert_pdf, agent_pdf])
            pa_loss = map(operator.add, pa_loss, t_info['a_loss'])
            pc_loss = map(operator.add, pc_loss, t_info['c_loss'])

        pa_loss = [_/self.p_step for _ in pa_loss]
        pc_loss = [_/self.p_step for _ in pc_loss]


        # clear buffer and dataset
        self.maddpg.clear_buffer()
        self.madcmt.clear_dataset()

        return {'d_loss': d_loss, 'de_loss': de_loss,'da_loss': da_loss, 'pa_loss': pa_loss, 'pc_loss': pc_loss} 

    def store_data(self, state_n, action_n, next_state_n, reward_n, done_n):
        # flag1 = self.maddpg.store_trans(state_n, action_n, next_state_n, np.zeros((self.n_agent,)), done_n)
        flag1 = self.maddpg.store_trans(state_n, action_n, next_state_n, reward_n, done_n)
        flag2 = self.madcmt.store_dataset(state_n, action_n)

        return (flag1 or flag2)
