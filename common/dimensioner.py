import numpy as np
import tensorflow as tf
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

class Dimensioner(object):
    def __init__(self, name, agent_id, input_dim, lower_dimension):
        self.name = name
        self.agent_id = agent_id
        self.lower_dimension = lower_dimension
        self.input_dim = input_dim

    def fit(self, **kwargs):
        raise NotImplementedError

    def transform(self, **kwargs):
        raise NotImplementedError

    def load(self):
        return 


class PCADimensioner(Dimensioner):
    def __init__(self, name, agent_id, s_dim, a_dim, lower_dimension=None):
        super().__init__(name, agent_id, s_dim+a_dim, lower_dimension)
        self.pca = None

    def fit(self, s, a):
        x = np.concatenate((s, a), axis=-1)

        if (self.lower_dimension is not None) and (np.shape(x)[-1]>self.lower_dimension):
            self.pca = PCA(n_components=self.lower_dimension, whiten=False)
            x = self.pca.fit_transform(x)
        
        return 

    def transform(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        if (self.lower_dimension is not None) and (self.pca is not None):
            x = self.pca.transform(x)
        return x

class AEDimensioner(Dimensioner):
    def __init__(self, sess, name, agent_id, s_dim, a_dim, lower_dimension=None, units=128, lr=0.01):
        super().__init__(name, agent_id, s_dim+a_dim, lower_dimension)
        self.sess = sess
        self.units=units
        self.ae = None
        self.lr = lr
        with tf.variable_scope('aedimensioner_{}'.format(agent_id)):
            self.input_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dim))
            out = tf.layers.dense(inputs=self.input_ph, units=self.units, activation=tf.nn.leaky_relu, name='l1')
            self.ae_out = tf.layers.dense(inputs=out, units=self.lower_dimension, activation=tf.nn.leaky_relu, name='l2')
            out = tf.layers.dense(inputs=self.ae, units=self.input_dim, activation=None, name='l3')

            self.loss = tf.reduce_mean(tf.square(self.input_ph-self.out))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

    def fit(self, s, a):
        x = np.concatenate((s, a), axis=-1)

        if (self.lower_dimension is not None) and (self.input_dim>self.lower_dimension):
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input_ph:x})
            self.ae = self.ae_out

        return

    def transform(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        if (self.lower_dimension is not None) and (self.ae is not None):
            x = sess.run(self.ae, feed_dict={self.input_ph:x})
        return x

class FWDimensioner(Dimensioner):
    def __init__(self, sess, name, agent_id, s_dim, a_dim, lower_dimension=None, units=128, lr=0.01):
        super().__init__(name, agent_id, s_dim+a_dim, lower_dimension)
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.sess = sess
        self.lr = lr
        self.units=units
        self.fw = None
        with tf.variable_scope('fwdimensioner_{}'.format(agent_id)):
            self.input_sa_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dim))
            self.target_s_ph = tf.placeholder(dtype=tf.float32, shape=(None, s_dim))
            out = tf.layers.dense(inputs=self.input_sa_ph, units=self.units, activation=tf.nn.leaky_relu, name='l1')
            out = tf.layers.dense(inputs=out, units=self.lower_dimension, activation=tf.nn.leaky_relu, name='l2')
            self.fw_out = tf.layers.dense(inputs=out, units=self.s_dim, activation=None, name='l2')

            self.loss = tf.reduce_mean(tf.square(self.target_s_ph-self.fw_out))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

    def fit(self, s, a):
        x = np.concatenate((s, a), axis=-1)

        if (self.lower_dimension is not None) and (self.input_dim>self.lower_dimension):
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input_ph:x})
            self.fw = self.fw_out
        
        return

    def transform(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        if (self.lower_dimension is not None) and (self.fw is not None):
            x = sess.run(self.fw, feed_dict={self.input_ph:x})
        return x
