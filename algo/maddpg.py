import os
import random
import operator
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tc

from common.utils import BunchBuffer, Transition
from common.utils import flatten, softmax
from common.utils import BaseModel

# This MADDPG assumes all actions are discret

num_units = 64
class Actor(BaseModel):
    def __init__(self, sess, state_space, act_space, lr=1e-2, tau=0.01, name=None, agent_id=None, grad_norm_clipping=None):
        super().__init__(name)

        self._lr = lr
        self._tau = tau

        self.sess = sess
        self.agent_id = agent_id
        self.grad_norm_clipping = grad_norm_clipping

        self._action_space = act_space
        self._observation_space = state_space

        self._loss = None
        self._train_op = None

        self.act_dim = flatten(self._action_space)

        self.obs_input = tf.placeholder(tf.float32, shape=(None,) + self._observation_space, name="Obs")

        with tf.variable_scope("eval"):
            self._eval_scope = tf.get_variable_scope().name
            self._eval_act = self._construct(self.act_dim)

        with tf.variable_scope("target"):
            self._target_scope = tf.get_variable_scope().name
            self._target_act = self._construct(self.act_dim)

        with tf.name_scope("Update"):  # smooth average update process
            self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
            self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]

    @property
    def t_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._target_scope)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._eval_scope)

    @property
    def act_tensor(self):
        return self._eval_act

    @property
    def obs_tensor(self):
        return self.obs_input

    def _construct(self, out_dim, norm=False):
        l1 = tf.layers.dense(self.obs_input, units=num_units, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=num_units, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=out_dim)

        u = tf.random_uniform(tf.shape(out))
        out = tf.nn.softmax(out - tf.log(-tf.log(u)), axis=-1) # gumbel softmax 

        return out

    def set_optimization(self, q_func):
        with tf.variable_scope("optimization"):
            q_loss = -tf.reduce_mean(q_func.value)
            reg_loss = tf.reduce_mean(tf.square(self._eval_act))
            self._loss = q_loss + reg_loss * 1e-3
            optimizer = tf.train.AdamOptimizer(self._lr)
            gradients = optimizer.compute_gradients(self._loss, self.e_variables)
            if self.grad_norm_clipping is not None:
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
            self._train_op = optimizer.apply_gradients(gradients)

    def update(self):
        self.sess.run(self._update_op)

    def soft_update(self):
        self.sess.run(self._soft_update_op)

    def act(self, obs):
        act = self.sess.run(self._eval_act, feed_dict={self.obs_input: obs})

        return act[0]

    def target_act(self, obs): # , one_hot=False):
        """ Return an action id -> integer """

        act = self.sess.run(self._target_act, feed_dict={self.obs_input: obs})
        
        return act
        # act = np.argmax(policy, axis=1)

        # if one_hot:
        #     return np.eye(*policy.shape)[act]
        # else:
        #     return act

    def train(self, feed_dict):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=feed_dict)
        self.soft_update()
        return loss


class Critic(BaseModel):
    def __init__(self, sess, multi_obs_phs, multi_act_phs, lr=1e-2, gamma=0.99, tau=0.01, name=None, agent_id=None, grad_norm_clipping=None):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id
        self.grad_norm_clipping = grad_norm_clipping

        # flatten observation shape
        self.mul_obs_dim = None
        # flatten action shape
        self.mul_act_dim = None

        self._lr = lr
        self._tau = tau
        self.L2 = 1e-3
        self.gamma = gamma

        self.multi_obs_phs = multi_obs_phs
        self.multi_act_phs = multi_act_phs

        obs_input = tf.concat(multi_obs_phs, axis=1, name="obs-clus-input")
        act_input = tf.concat(multi_act_phs, axis=1, name="act-clus-input")

        self.input = tf.concat([obs_input, act_input], axis=1, name="concat-input")
        self.target_input = tf.concat([obs_input, act_input], axis=1, name="target-concat-input")

        with tf.variable_scope("eval"):
            self._e_scope = tf.get_variable_scope().name
            self.e_q = self._construct(self.input)

        with tf.variable_scope("target"):
            self._t_scope = tf.get_variable_scope().name
            self.t_q = self._construct(self.target_input)

        with tf.name_scope("Update"):  # smooth average update process
            self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
            self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var
                                    in zip(self.t_variables, self.e_variables)]

        with tf.variable_scope("Optimization"):
            # weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.e_variables])

            self.t_q_input = tf.placeholder(tf.float32, shape=(None, 1), name="target-input")
            # self._loss = 0.5 * tf.reduce_mean(tf.square(self.t_q_input - self.e_q)) + weight_decay
            self._loss = tf.reduce_mean(tf.square(self.t_q_input - self.e_q)) 

            optimizer = tf.train.AdamOptimizer(self._lr)
            gradients = optimizer.compute_gradients(self._loss, self.e_variables)
            if self.grad_norm_clipping is not None:
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
            self._train_op = optimizer.apply_gradients(gradients)

    @property
    def t_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._t_scope)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._e_scope)

    @property
    def value(self):
        return self.e_q

    @property
    def obs_cluster_ph(self):
        return self.multi_obs_phs

    @property
    def act_cluster_ph(self):
        return self.multi_act_phs

    def _construct(self, input_ph, norm=False):
        l1 = tf.layers.dense(input_ph, units=num_units, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=num_units, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=1, name="Q")

        return out

    def update(self):
        self.sess.run(self._update_op)

    def soft_update(self):
        """ Soft update target net """

        self.sess.run(self._soft_update_op)

    def calculate_next_q(self, feed_dict):# next_obs_list, next_act_list):
        """ Return target Q value

        :param next_obs_list: list, a list of N agents' observations
        :param next_act_list: list, a list of N agents' actions
        :return: array-like with shape: (n_batch, 1)
        """

        q_values = self.sess.run(self.t_q, feed_dict=feed_dict)

        return q_values

    def train(self, target_q_values, feed_dict): # obs_list, act_list):
        """ Train critic network

        :param target_q_values: array-like with shape of (n_batch, 1)
        :param obs_list: list, list of N agents' observations
        :param act_list: list, list of N agents' actions
        :return: critic loss, float
        """

        feed_dict[self.t_q_input] = target_q_values
        _, loss = self.sess.run([self._train_op, self._loss], feed_dict=feed_dict)

        self.soft_update()

        return loss


class MADDPG(object):
    def __init__(self, sess, env, name, n_agent, batch_size=512, actor_lr=1e-2, critic_lr=1e-2, 
            gamma=0.99, tau=0.01, memory_size=10**6, grad_norm_clipping=None):
        # == Initialize ==
        self.name = name
        self.sess = sess
        self.env = env
        self.n_agent = n_agent
        self.gamma = gamma

        self.actors = []  # hold all Actors
        self.critics = []  # hold all Critics
        self.actions_dims = []  # record the action split for gradient apply

        self.replay_buffer = BunchBuffer(n_agent, memory_size, batch_size)
        self.batch_size = batch_size

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for i in range(self.env.n):
                print("initialize actor for agent {} ...".format(i))
                with tf.variable_scope("policy_{}_{}".format(self.name, i)):
                    obs_space, act_space = env.observation_space[i].shape, (env.action_space[i].n,)
                    self.actors.append(Actor(self.sess, state_space=obs_space, act_space=act_space, lr=actor_lr, tau=tau,
                                             name=name, agent_id=i, grad_norm_clipping=grad_norm_clipping))

            # collect action outputs of all actors
            self.obs_phs = [actor.obs_tensor for actor in self.actors]
            act_tfs = [actor.act_tensor for actor in self.actors]
            self.act_phs = act_tfs

            for i in range(self.env.n):
                print("initialize critic for agent {} ...".format(i))
                # mask_act_tfs = self._mask_other_act_phs(act_tfs, i)  # stop gradient
                with tf.variable_scope("critic_{}_{}".format(name, i)):
                    # self.critics.append(Critic(self.sess, self.obs_phs, self.act_phs, mask_act_tfs, lr=critic_lr, name=name, agent_id=i))
                    self.critics.append(Critic(self.sess, self.obs_phs, self.act_phs, lr=critic_lr, name=name, agent_id=i, grad_norm_clipping=grad_norm_clipping))
                    self.actions_dims.append(self.env.action_space[i].n)

            # set optimizer for actors
            for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
                with tf.variable_scope("optimizer_{}_{}".format(name, i)):
                    actor.set_optimization(critic)

    def init(self):
        self.sess.run(tf.global_variables_initializer())

        # hard sync
        for i in range(self.env.n):
            self.actors[i].update()
            self.critics[i].update()

    @staticmethod
    def _mask_other_act_phs(act_phs, agent_id):
        """ Mask action placeholders corresponding to other agents whose id != agent_id

        :param act_phs: list, action placeholder list
        :param agent_id: int, agent id
        :return: list of masked action placeholder
        """

        res = []
        for i, ph in enumerate(act_phs):
            # ph = tf.reshape(tf.dtypes.cast(ph, tf.float32), (-1, 1))
            if agent_id == i:
                res.append(ph)
            else:
                res.append(tf.stop_gradient(ph))

        return res

    def store_trans(self, state_n, action_n, next_state_n, reward_n, done_n):
        return self.replay_buffer.push(state_n, action_n, next_state_n, reward_n, done_n)

    def clear_buffer(self):
        self.replay_buffer.clear()

    def act(self, obs_set):
        """ Accept a observation list, return action list of all agents. """
        actions = []
        for i, (obs, agent) in enumerate(zip(obs_set, self.actors)):
            n = self.actions_dims[i]

            policy = agent.act([obs])

            # actions.append([np.random.choice(n, p=policy)])
            actions.append(policy)

        return actions

    def save(self, dir_path, epoch, max_to_keep):
        """ Save model
        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        # TODO(ming): store replay-buffer too

        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            tf.gfile.DeleteRecursively(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        #for i in model_vars:
        #    print(i)
        #print("sum {} vars".format(len(model_vars)))
        saver = tf.train.Saver(model_vars, max_to_keep=max_to_keep)
        save_path = saver.save(self.sess, dir_name + "/{}".format(self.name), global_step=epoch)
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, dir_path, epoch=0):
        """ Load model from local storage.

        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        # TODO(ming): load replay-buffer too

        file_path = None

        dir_name = os.path.join(dir_path, self.name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        #for i in model_vars:
        #    print(i)
        #print("sum {} vars".format(len(model_vars)))
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_name, "{}-{}".format(self.name, epoch))
        saver.restore(self.sess, file_path)
        # print("[!] Load model failed, please check {} exists".format(file_path))

    def train_step(self, batch_list):
        """ Train actor and critic step by step

        :param batch_list: list, list of batch data for N agents
        :return: a_loss, c_loss: two lists for N agents which are actor-loss and critic-loss respectively
        """
        a_loss, c_loss = [0.] * self.n_agent, [0.] * self.n_agent

        state_clus = [None for _ in range(self.n_agent)]
        next_state_clus = [None for _ in range(self.n_agent)]
        act_clus = [None for _ in range(self.n_agent)]

        # re-construct samples
        for i, batch in enumerate(batch_list):
            state_clus[i] = np.stack(batch.state)
            next_state_clus[i] = np.stack(batch.next_state)

            act_clus[i] = np.stack(batch.action)

            # convert to onehot
            # act_clus[i] = np.eye(len(act_clus[i]), self.env.action_space[i].n)[act_clus[i]]

        # get target action
        next_act_clus = [None for _ in range(self.n_agent)]
        for i, batch in enumerate(batch_list):
            next_act_clus[i] = self.actors[i].target_act(batch.next_state)
            # next_act_clus[i] = np.eye(len(next_act_clus[i]), self.env.action_space[i].n)[next_act_clus[i]]

        # train critic
        feed_dict1 = dict()
        feed_dict1.update(zip(self.obs_phs, next_state_clus))
        feed_dict1.update(zip(self.act_phs, next_act_clus))

        feed_dict2 = dict()
        feed_dict2.update(zip(self.obs_phs, state_clus))
        feed_dict2.update(zip(self.act_phs, act_clus))
        for i, batch in enumerate(batch_list):
            target_q = self.critics[i].calculate_next_q(feed_dict1).reshape((-1,))
            target_q = np.array(batch.reward) + (1. - np.array(batch.done)) * target_q * self.gamma

            c_loss[i] = self.critics[i].train(target_q.reshape((-1, 1)) ,feed_dict2) # state_clus, act_clus)

        # train actor
        for i, batch in enumerate(batch_list):
            act_phs_i = self.act_phs[:i]+self.act_phs[i+1:]
            act_clus_i = act_clus[:i]+act_clus[i+1:]
            feed_dict = dict()
            feed_dict.update(zip(act_phs_i, act_clus_i))
            feed_dict.update(zip(self.obs_phs, state_clus))

            a_loss[i] = self.actors[i].train(feed_dict)

        return a_loss, c_loss

    def train(self):
        """ Run multiple training steps

        :return: mean_a_loss, mean_c_loss: mean of actor loss and critic loss for N agents of n_batch
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        if len(self.replay_buffer) < self.replay_buffer.capacity:
            return None

        n_batch = 1
        mean_a_loss, mean_c_loss = [0.] * self.n_agent, [0.] * self.n_agent

        for i in range(n_batch):
            batch_list = self.replay_buffer.sample()
            a_loss, c_loss = self.train_step(batch_list)

            mean_a_loss = map(operator.add, mean_a_loss, a_loss)
            mean_c_loss = map(operator.add, mean_c_loss, c_loss)

        mean_a_loss = list(map(lambda x: x / n_batch, mean_a_loss))
        mean_c_loss = list(map(lambda x: x / n_batch, mean_c_loss))

        return {'a_loss': mean_a_loss, 'c_loss': mean_c_loss} # mean loss for n agent in n_batch
