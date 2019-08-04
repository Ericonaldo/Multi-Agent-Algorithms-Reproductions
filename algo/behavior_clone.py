import os
import random
import operator
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from common.utils import flatten, softmax
from common.utils import BaseModel

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

    def set_optimization(self, q_func):
        with tf.variable_scope("optimization"):
            self._loss = -tf.reduce_mean(tf.square(self.target_act - self._act_prob))
            optimizer = tf.train.AdamOptimizer(self._lr)
            grad_vars = optimizer.compute_gradients(self._loss, self.e_variables)
            self._train_op = optimizer.apply_gradients(grad_vars)

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

        self.tar_act_n_phs = [tf.placeholder(tf.int32, shape=[None,], name='target_act'+str(i) for i in range)
        
        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize behavior actor for agent {} ...".format(i))
                with tf.variable_scope("policy_{}_{}".format(name, i)):
                    obs_space, act_space = env.observation_space[i].shape, (env.action_space[i].n,)
                    self.actors.append(Actor(self.sess, state_space=obs_space, act_space=act_space, lr=actor_lr, tau=tau,
                                             name=name, agent_id=i))

            # collect action outputs of all actors
            self.obs_phs = [actor.obs_tensor for actor in self.actors]
            act_tfs = [actor.act_tensor for actor in self.actors]


        loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1)
        loss = - tf.reduce_mean(loss)
        tf.summary.scalar('loss/cross_entropy', loss)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(loss)

        self.merged = tf.summary.merge_all()

    def train(self, obs, act):


