import os
import random
import operator
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tc

from common.utils import BaseModel, BaseAgent


class Actor(BaseModel):
    def __init__(self, sess, name, agent_id, act_dim, obs_input, lr=0.01, units=64, discrete=True, trainable=True):
        super().__init__(name)

        self._lr = lr
        self.num_units = units
        self.sess = sess
        self.agent_id = agent_id
        self._act_dim = act_dim
        self.obs_input = obs_input
        self.trainable = trainable
        self.discrete = discrete

        self._loss = None
        self._train_op = None

        self._scope = tf.get_variable_scope().name
        self._act_probs = tf.nn.softmax(self._construct(act_dim))

        self.act_stochastic = tf.multinomial(
            tf.log(self.act_probs), num_samples=1)
        self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

        self.act_deterministic = tf.argmax(self.act_probs, axis=1)

    @property
    def variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)

    @property
    def act_probs(self):
        return self._act_probs

    def _construct(self, out_dim, norm=False):
        l1 = tf.layers.dense(self.obs_input, units=self.num_units,
                             activation=tf.nn.relu, name="l1", trainable=self.trainable)
        if norm:
            l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=self.num_units,
                             activation=tf.nn.relu, name="l2", trainable=self.trainable)
        if norm:
            l2 = tc.layers.layer_norm(l2)

        self.logits = tf.layers.dense(
            l2, units=out_dim, activation=None, name="out", trainable=self.trainable)
        out = self.logits

        return out

    def act(self, feed_dict, stochastic=True):
        if self.discrete:
            if stochastic:
                return self.sess.run(self.act_stochastic, feed_dict=feed_dict)
            else:
                return self.sess.run(self.act_deterministic, feed_dict=feed_dict)
        else:
            return self.sess.run(self._act_probs, feed_dict=feed_dict)


class Critic(BaseModel):
    def __init__(self, sess, name, agent_id, obs_phs, lr, units=64):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id
        self.num_units = units
        self._lr = lr
        self.obs_input = obs_phs

        self._scope = tf.get_variable_scope().name
        self._v = self._construct()

    @property
    def variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)

    @property
    def value(self):
        return self._v

    def get_value(self, feed_dict):
        return self.sess.run(self._v, feed_dict=feed_dict)

    def _construct(self, norm=False):
        l1 = tf.layers.dense(
            self.obs_input, units=self.num_units, activation=tf.nn.relu, name="l1")
        if norm:
            l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=self.num_units,
                             activation=tf.nn.relu, name="l2")
        if norm:
            l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=1, name="v")

        return out


class PPO(BaseAgent):
    def __init__(self, sess, env, name, agent_id, a_lr=0.01, c_lr=0.01, gamma=0.95, clip_value=0.2, ent_w=0.01, num_units=64, discrete=True):
        """
        :param clip_value:
        :param en_w: parameter for entropy bonus
        """
        super().__init__(env, name)
        # == Initialize ==
        self.sess = sess
        self.agent_id = agent_id
        self.gamma = gamma
        self.num_units = num_units
        self.discrete = discrete

        self.sta_dim = env.observation_space[agent_id].shape[0]
        self.act_dim = env.action_space[agent_id].n
        # self.sta_dim = env.observation_space.shape[0]
        # self.act_dim = env.action_space.n

        with tf.variable_scope("{}_{}".format(name, agent_id)):
            self._scope = tf.get_variable_scope().name

            self.obs_phs = tf.placeholder(tf.float32, shape=(
                None,) + env.observation_space[agent_id].shape, name="Obs")
            # self.obs_phs = tf.placeholder(tf.float32, shape=(None,) + env.observation_space.shape, name="Obs")
            self.tar_act = tf.placeholder(
                tf.float32, shape=(None, self.act_dim), name="Obs")

            # inputs for train_op
            with tf.variable_scope('train_inp'):
                self.actions = tf.placeholder(dtype=tf.float32, shape=[
                                              None, self.act_dim], name='actions')
                self.rewards = tf.placeholder(dtype=tf.float32, shape=[
                                              None, ], name='rewards')
                self.v_preds_next = tf.placeholder(
                    dtype=tf.float32, shape=[None, ], name='v_preds_next')
                self.gaes = tf.placeholder(
                    dtype=tf.float32, shape=[None, ], name='gaes')

            with tf.variable_scope("actor"):
                self.pi = Actor(sess, name, agent_id, self.act_dim, self.obs_phs,
                                a_lr, num_units, discrete=discrete, trainable=True)
            with tf.variable_scope("old_actor"):
                self.oldpi = Actor(sess, name, agent_id, self.act_dim, self.obs_phs,
                                   a_lr, num_units, discrete=discrete, trainable=False)

            with tf.variable_scope("critic"):
                self.critic = Critic(sess, name, agent_id,
                                     self.obs_phs, c_lr, num_units)

            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(
                    self.pi.variables, self.oldpi.variables)]

            self.act_probs = self.pi.act_probs
            self.logits = self.pi.logits
            act_probs_old = self.oldpi.act_probs

            # probabilities of actions which agent took with policy
            act_probs = self.act_probs * self.actions
            act_probs = tf.reduce_sum(act_probs, axis=1)

            # probabilities of actions which agent took with old policy
            act_probs_old = act_probs_old * self.actions
            act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

            with tf.variable_scope('bc_init'):
                if self.discrete:
                    self.bc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.tar_act, logits=self.logits))
                else:
                    self._bc_loss = tf.reduce_mean(
                        tf.square(self.tar_act - self.act_probs))
                bc_optim = tf.train.AdamOptimizer(a_lr)
                self._train_bc_op = bc_optim.minimize(self._bc_loss)

            with tf.variable_scope('optimization'):
                # construct computation graph for loss_clip
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.stop_gradient(tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0)) - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0))))
                clipped_ratios = tf.clip_by_value(
                    ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
                loss_clip = tf.minimum(tf.multiply(
                    self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                loss_clip = -tf.reduce_mean(loss_clip)

                # construct computation graph for loss of entropy bonus
                entropy = -tf.reduce_sum(self.act_probs *
                                         tf.log(tf.clip_by_value(self.act_probs, 1e-10, 1.0)), axis=1)
                # mean of entropy of pi(obs)
                entropy_loss = -tf.reduce_mean(entropy, axis=0)

                self.a_loss = loss_clip + ent_w * entropy_loss

                # construct computation graph for loss of value function
                v_preds = self.critic.value
                advantage = self.rewards + self.gamma * self.v_preds_next - v_preds
                loss_vf = tf.square(advantage)
                self.c_loss = tf.reduce_mean(loss_vf)

                optimizer_a = tf.train.AdamOptimizer(
                    learning_rate=a_lr, epsilon=1e-5)
                self.a_gradients = optimizer_a.compute_gradients(
                    self.a_loss, var_list=self.pi.variables)
                self.a_train_op = optimizer_a.minimize(self.a_loss)

                optimizer_c = tf.train.AdamOptimizer(
                    learning_rate=c_lr, epsilon=1e-5)
                self.c_gradients = optimizer_c.compute_gradients(
                    self.c_loss, var_list=self.critic.variables)
                self.c_train_op = optimizer_c.minimize(self.c_loss)

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def bc_init(self, obs, tar_act):
        feed_dict = {self.obs_phs: obs, self.tar_act: tar_act}
        bc_loss, _ = self.sess.run(
            [self._bc_loss, self._train_bc_op], feed_dict=feed_dict)
        return bc_loss

    def act(self, obs):
        feed_dict = {self.obs_phs: [obs]}
        act = self.pi.act(feed_dict)
        value = self.critic.get_value(feed_dict)
        value = np.asscalar(value)

        if self.discrete:
            act = np.eye(self.act_dim)[act[0]]
        else:
            act = act[0]

        return act, value

    def update_oldpi(self):
        self.sess.run(self.update_oldpi_op)

    def train(self, obs, actions, rewards, v_preds_next, gaes):
        # update actor
        _, a_loss = self.sess.run([self.a_train_op, self.a_loss], feed_dict={self.obs_phs: obs,
                                                                             self.actions: actions,
                                                                             self.rewards: rewards,
                                                                             self.v_preds_next: v_preds_next,
                                                                             self.gaes: gaes})

        # update critic
        _, c_loss = self.sess.run([self.c_train_op, self.c_loss], feed_dict={self.obs_phs: obs,
                                                                             self.rewards: rewards,
                                                                             self.v_preds_next: v_preds_next})

        return {'a_loss': a_loss, 'c_loss': c_loss}

    def get_grad(self, obs, actions, rewards, v_preds_next, gaes):
        return self.sess.run([self.a_gradient, self.c_gradients], feed_dict={self.obs: obs,
                                                                             self.actions: actions,
                                                                             self.rewards: rewards,
                                                                             self.v_preds_next: v_preds_next,
                                                                             self.gaes: gaes})

    def save(self, dir_path, epoch, max_to_keep=10):
        """ Save model
        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            tf.gfile.DeleteRecursively(dir_name)
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)
        # for i in model_vars:
        #    print(i)
        # print("sum {} vars".format(len(model_vars)))
        saver = tf.train.Saver(model_vars, max_to_keep=max_to_keep)
        save_path = saver.save(
            self.sess, dir_name + "/{}".format(self.name+'_'+str(self.agent_id)), global_step=epoch)
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, dir_path, epoch=None):
        """ Load model from local storage.

        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        file_path = None

        dir_name = os.path.join(dir_path, self.name)
        model_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)
        # for i in model_vars:
        #    print(i)
        # print("sum {} vars".format(len(model_vars)))
        saver = tf.train.Saver(model_vars)
        print("Loading [*] Model from {}".format(dir_name))
        if epoch is not None:
            file_path = os.path.join(
                dir_name, "{}-{}".format(self.name, epoch))
            saver.restore(self.sess, file_path)
        else:
            file_path = dir_name
            saver.restore(self.sess, tf.train.latest_checkpoint(file_path))
        print("Loaded [*] Model from file {}".format(file_path))
