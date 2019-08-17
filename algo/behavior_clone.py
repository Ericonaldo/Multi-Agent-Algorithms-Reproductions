import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel, BaseAgent
from common.buffer import Dataset


units = 64
class BCActor(BaseModel):
    def __init__(self, sess, state_space, act_space, lr=1e-2, name=None, agent_id=None, discrete=True, sample_action=True):
        super().__init__(name)

        self._lr = lr

        self.sess = sess
        self.agent_id = agent_id
        self.discrete = discrete
        self.sample_action = sample_action

        self._action_space = act_space
        self._observation_space = state_space

        self._loss = None
        self._train_op = None

        self.act_dim = flatten(self._action_space)

        self.obs_input = tf.placeholder(tf.float32, shape=(None,) + self._observation_space, name="Obs")
        self.target_act = tf.placeholder(tf.float32, shape=(None,) + self._action_space, name="TAct")

        self._scope = tf.get_variable_scope().name
        self._logits = self._construct(self.act_dim)
        self._act = tf.nn.softmax(self._logits)


        with tf.variable_scope("optimization"):
            if not discrete:
                self._loss = tf.reduce_mean(tf.square(self.target_act - self._act))
            else:
                self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_act, logits=self._logits))
            optimizer = tf.train.AdamOptimizer(self._lr)
            self._train_op = optimizer.minimize(self._loss)
            # grad_vars = optimizer.compute_gradients(self._loss, self.trainable_variables)
            # self._train_op = optimizer.apply_gradients(grad_vars)

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)

    @property
    def act_tensor(self):
        return self._act

    @property
    def tar_act_tensor(self):
        return self.target_act

    @property
    def obs_tensor(self):
        return self.obs_input

    def _construct(self, out_dim, norm=False):
        l1 = tf.layers.dense(self.obs_input, units=units, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=units, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=out_dim)
        # out = tf.nn.softmax(out, axis=1)

        return out

    def update(self):
        self.sess.run(self._update_op)

    def soft_update(self):
        self.sess.run(self._soft_update_op)

    def act(self, obs):
        action = self.sess.run(self._act, feed_dict={self.obs_input: [obs]})
        if self.discrete:
            label = np.argmax(action[0])
            if self.sample_action:
                label = np.random.choice(len(action[0]), 1, p=action[0])[0]
            action = [np.eye(len(action[0]))[label]]
        return action[0]

    def train(self, feed_dict):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=feed_dict)
        return loss

class MABehavioralCloning(BaseAgent):
    def __init__(self, sess, env, name, n_agent, batch_size=64, lr=1e-2, discrete=True, sample_action=True):
        super().__init__(env, name)
        self.sess = sess
        self.n_agent = n_agent

        self.actors = []
        self.actions_dims = []
        self.batch_size = batch_size

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize behavior actor for agent {} ...".format(i))
                with tf.variable_scope("policy_{}".format(i)):
                    obs_space, act_space = env.observation_space[i].shape, (env.action_space[i].n,)
                    self.actors.append(BCActor(self.sess, state_space=obs_space, act_space=act_space, lr=lr,
                                             name=name, agent_id=i, discrete=discrete, sample_action=sample_action))
                    self.actions_dims.append(self.env.action_space[i].n)

            # collect action outputs of all actors
            self.obs_phs = [actor.obs_tensor for actor in self.actors]
            self.tar_act_phs = [actor.tar_act_tensor for actor in self.actors]

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs_set):
        """ Accept a observation list, return action list of all agents. """
        actions = []
        for i, (obs, agent) in enumerate(zip(obs_set, self.actors)):
            policy = agent.act(obs)
            # noise = np.random.gumbel(size=np.shape(logits)) # gumbel softmax
            # policy += noise

            # n = self.actions_dims[i]
            # actions.append([np.random.choice(n, p=policy)])
            actions.append(policy)

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
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        saver = tf.train.Saver(model_vars)
        print("loading [*] Model from dir: {}".format(dir_name))
        if epoch is not None:
            file_path = os.path.join(dir_name, "{}-{}".format(self.name, epoch))
            saver.restore(self.sess, file_path)
        else:
            file_path = dir_name
            saver.restore(self.sess, tf.train.latest_checkpoint(file_path))

    def train(self, obs, tar_act):
        
        loss = [0.] * self.n_agent

        for i in range(self.n_agent):
            feed_dict = {self.obs_phs[i]: obs[i], self.tar_act_phs[i]: tar_act[i]}

            loss[i] = self.actors[i].train(feed_dict)

        return {'loss': loss} 
