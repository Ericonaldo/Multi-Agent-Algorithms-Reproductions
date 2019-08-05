import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel

class Dataset(object):
    def __init__(self, agent_num, capacity=65536):
        self.agent_num = agent_num
        self._capacity = capacity
        self._size = 0
        self._observations = [[] for _ in range(agent_num)]
        self._actions = [[] for _ in range(agent_num)]

    def __len__(self):
        return self._size

    def push(self, obs_n, act_n):
        if self._size<self._capacity:
            self._observations = map(lambda x, y: y + [x], obs_n, self._observations)
            self._actions = map(lambda x, y: y + [x], act_n, self._actions)
        
        self._size = min(self.size+1, self._capacity)
        
        return self._size<self._capacity

    def shuffle(self):
        indices = list(range(self._size)) 

    def next(self, batch_size):


class BCActor(BaseModel):
    def __init__(self, sess, state_space, act_space, lr=1e-4, name=None, agent_id=None):
        super().__init__(name)

        self._lr = lr

        self.sess = sess
        self.agent_id = agent_id

        self._action_space = act_space
        self._observation_space = state_space

        self._loss = None
        self._train_op = None

        self.act_dim = flatten(self._action_space)

        self.obs_input = tf.placeholder(tf.float32, shape=(None,) + self._observation_space, name="Obs")
        self.target_act = tf.placeholder(tf.float32, shape=(None,self._action_space), name="TAct")

        self._eval_scope = tf.get_variable_scope().name
        self.eval_net = self._construct(self.act_dim)
        self._act_prob = tf.nn.softmax(self.eval_net)

        self._act_tf = self._act_prob

        with tf.variable_scope("optimization"):
            self._loss = -tf.reduce_mean(tf.square(self.target_act - self._act_prob))
            optimizer = tf.train.AdamOptimizer(self._lr)
            grad_vars = optimizer.compute_gradients(self._loss, self.e_variables)
            self._train_op = optimizer.apply_gradients(grad_vars)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._eval_scope)

    @property
    def act_tensor(self):
        return self._act_tf

    @property
    def obs_tensor(self):
        return self.obs_input

    def _construct(self, out_dim, norm=True):
        l1 = tf.layers.dense(self.obs_input, units=100, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=100, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=out_dim)

        return out

    def update(self):
        self.sess.run(self._update_op)

    def soft_update(self):
        self.sess.run(self._soft_update_op)

    def act(self, obs):
        policy_logits = self.sess.run(self._act_tf, feed_dict={self.obs_input: [obs]})
        return policy_logits[0]

    def train(self, feed_dict):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=feed_dict)
        return loss

class MABehavioralCloning:
    def __init__(self, sess, env, name, n_agent, batch_size=64, lr=1e-4):
        self.name = name
        self.sess = sess
        self.env = env
        self.n_agent = n_agent

        self.action_dims = []
        self.batch_size = batch_size

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize behavior actor for agent {} ...".format(i))
                with tf.variable_scope("policy_{}_{}".format(name, i)):
                    obs_space, act_space = env.observation_space[i].shape, (env.action_space[i].n,)
                    self.actors.append(BCActor(self.sess, state_space=obs_space, act_space=act_space, lr=actor_lr,
                                             name=name, agent_id=i))

            # collect action outputs of all actors
            self.obs_phs = [actor.obs_tensor for actor in self.actors]
            self.tar_act_phs = [actor.tar_act_tensor for actor in self.actors]
            self.act_tfs = [actor.act_tensor for actor in self.actors]

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs_set):
        """ Accept a observation list, return action list of all agents. """
        actions = []
        for i, (obs, agent) in enumerate(zip(obs_set, self.actors)):
            n = self.actions_dims[i]

            logits = agent.act(obs)
            noise = np.random.gumbel(size=np.shape(logits)) # gumbel softmax
            logits += noise

            policy = softmax(logits)

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

    def train(self, obs, tar_act):
        
        loss = [0.] * self.n_agent

        for i in range(self.n_agent):
            feed_dict = dict()
            feed_dict.update(zip(self.obs_phs[i], obs[i]))
            feed_dict.update(zip(self.tar_act_phs[i], tar_act[i]))

            loss[i] = self.actors[i].train(feed_dict)

        return {'loss': loss} 
