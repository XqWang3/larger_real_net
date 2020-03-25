import tensorflow as tf
import numpy as np
from .noise import OUNoise
from . import tf_util as U

class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, s_dim_wave, s_dim_wait, a_bound=1, gamma=0.95, tau=0.01, lr_a=1e-4, lr_c=1e-3, memory_size=100000, batch_size=64, scope=""):
        self.memory = np.zeros((memory_size, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = sess
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_filled = 0
        self.exploration_noise = OUNoise(a_dim)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.obs_wave = s_dim_wave
        self.obs_wait = s_dim_wait
        self.Swave = tf.placeholder(tf.float32, (None,) + (str(self.obs_wave),), 's1')
        self.Swait = tf.placeholder(tf.float32, (None,) + (str(self.obs_wait),), 's2')
        #self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.a = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.S_wave = tf.placeholder(tf.float32, (None,) + (str(self.obs_wave),), 's1')
        self.S_wait = tf.placeholder(tf.float32, (None,) + (str(self.obs_wait),), 's2')
        #self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.float32, [None, 1], 'done')
        self.scope = scope

        with tf.variable_scope('Actor'):
            self.a, self.pre_a = self._build_a(self.Swave, self.Swait, scope='eval', trainable=True)
            a_, *_ = self._build_a(self.S_wave, self.S_wait, scope='target', trainable=False)
            #self.a, self.pre_a = self._build_a(self.S, scope='eval', trainable=True)
            #a_, *_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.Swave, self.Swait, self.a, scope='eval', trainable=True) #noise1  total re=-7.5
            #q = self._build_c(self.S, self.a, scope='eval', trainable=True)  #noise  total re=-10
            q_ = self._build_c(self.S_wave, self.S_wait, a_, scope='target', trainable=False)

        # networks parameters
        prefix = (self.scope + "/") if len(self.scope) > 0 else ""
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix+'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix+'Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix+'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix+'Critic/target')

        # target net replacement
        self.soft_replace = [
            tf.assign(t, (1 - tau) * t + tau * e)
            for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + (1.-self.D) * gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        # self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)
        optimizer = tf.train.AdamOptimizer(lr_c)
        self.ctrain = U.minimize_and_clip(optimizer, td_error, self.ce_params, .5)

        a_reg = tf.reduce_mean(tf.reduce_sum(tf.square(self.pre_a), axis=-1))
        a_loss = - tf.reduce_mean(q) + 1e-3 * a_reg   # maximize the q
        # self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)
        optimizer = tf.train.AdamOptimizer(lr_a)
        self.atrain = U.minimize_and_clip(optimizer, a_loss, self.ae_params, .5)
        #self.sess.run(tf.global_variables_initializer())

    def action(self, s):
        #print(s)    #[array([0., 1., 0., 5., 1., 0., 0., 0., 1., 0., 0., 0.])]
        s = s[0]
        Swave = s[:self.obs_wave]
        Swait = s[self.obs_wave:]
        feed_dict = {
            self.Swave: Swave[np.newaxis, :],
            self.Swait: Swait[np.newaxis, :],
        }
        #print(s)    #[0. 1. 0. 5. 1. 0. 0. 0. 1. 0. 0. 0.]
        #return self.sess.run([self.a[0],self.pre_a[0]], {self.S: s[np.newaxis, :]})
        #[array([0.9632396, 0.03676045], dtype=float32), array([-3.3891506, -6.0047126], dtype=float32)], [array([0.5186354, 0.4813646], dtype=float32), array([0.6577059, 0.19652545], dtype=float32)]
        #return self.sess.run(self.a[0], {self.S: s[np.newaxis, :]})
        return self.sess.run(self.a[0], feed_dict)

    def noise_action(self, s):
        s = s[0]
        Swave = s[:self.obs_wave]
        Swait = s[self.obs_wave:]
        #print(Swave[np.newaxis, :])
        feed_dict = {
            self.Swave: Swave[np.newaxis, :],
            self.Swait: Swait[np.newaxis, :],
        }
        #return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0] + self.exploration_noise.noise()
        return self.sess.run(self.a, feed_dict)[0] + self.exploration_noise.noise()

    def preupdate(self):
        pass

    def update(self, train_step):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_filled, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        #ba = bt[:, self.s_dim: self.s_dim + 1]
        br = bt[:, -self.s_dim - 2: -self.s_dim - 1]
        bs_ = bt[:, -self.s_dim - 1: -1]
        bd = bt[:, -1:]

        if self.pointer > 500:
            #self.sess.run(self.atrain, {self.S: bs})
            #self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.D: bd})
            self.sess.run(self.atrain, {self.Swave: bs[:, :self.obs_wave],self.Swait: bs[:, self.obs_wave:]})
            self.sess.run(self.ctrain, {self.Swave: bs[:, :self.obs_wave],self.Swait: bs[:, self.obs_wave:],
                                        self.a: ba, self.R: br,
                                        self.S_wave: bs_[:, :self.obs_wave],self.S_wait: bs_[:, self.obs_wave:],
                                        self.D: bd})

    def experience(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        self.memory_filled = min(self.memory_filled+1, self.memory_size)

    def _build_a(self, s_wave, s_wait, scope, trainable):
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s_wave, 64, activation=tf.nn.relu, name='Dense-Obs1', trainable=trainable)
            net2 = tf.layers.dense(s_wait, 64, activation=tf.nn.relu, name='Dense-Obs2', trainable=trainable)
            net = tf.concat([net1, net2], 1)
            #dense = tf.layers.dense(net, 256, activation=tf.nn.relu, name='Dense1', trainable=trainable)
            dense1 = tf.layers.dense(net, units=64, activation=tf.nn.relu, name="Dense-Out1")
            a = tf.layers.dense(dense1, self.a_dim, activation=None, name='a', trainable=trainable)
            #print('a',a)  #Tensor shape=(?, 2)
            u = tf.random_uniform(tf.shape(a))
            u = tf.nn.softmax(a - tf.log(-tf.log(u)), axis=-1)
            #print('u',u)   #Tensor shape=(?, 2)
            return u, a

    '''def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)
            #net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)  # add a layer for equal to Q-learning layer numbers.
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)'''
    def _build_c(self, s_wave, s_wait, a, scope, trainable):
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s_wave, 64, activation=tf.nn.relu, name='Dense-Obs3', trainable=trainable)
            net2 = tf.layers.dense(s_wait, 64, activation=tf.nn.relu, name='Dense-Obs4', trainable=trainable)
            net = tf.concat([net1, net2], 1)
            #dense = tf.layers.dense(net, 64, activation=tf.nn.relu, name='Dense2', trainable=trainable)
            net2 = tf.layers.dense(a, 64, activation=tf.nn.relu, name='Dense-a', trainable=trainable)
            net = tf.concat([net, net2], axis=1)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

            #w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            #w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            #b1_s = tf.get_variable('b1_s', [1, n_l1], trainable=trainable)
            #b1_a = tf.get_variable('b1_a', [1, n_l1], trainable=trainable)
            #net_a = tf.nn.relu(tf.matmul(a, w1_a) + b1_a)
            #net = tf.concat([net_s, net_a], axis=1)
            #net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)
            #net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)
            ##net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)
            #net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)  # add a layer for equal to Q-learning layer numbers.
            #return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

