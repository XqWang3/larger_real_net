import tensorflow as tf
import numpy as np
import maddpg.common.tf_util as U

class MADDPGEN(object):
    def __init__(self, nb_agent, **kwargs):
        self.nb_agent = nb_agent
        self.maddpgs = list()
        for i in range(self.nb_agent):
            scope_name = 'maddpg_{}'.format(i)
            with tf.variable_scope(scope_name) as sc:
                self.maddpgs.append(MADDPG(scope=scope_name, n_b_agent=nb_agent, **kwargs))

    def action(self, obs_n):
        return [maddpg.action([obs]) for obs, maddpg in zip(obs_n, self.maddpgs)]

    def experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal):
        for maddpg, rew, done in zip(self.maddpgs, rew_n, done_n):
            #print()
            maddpg.experience([np.hstack(obs_n)], [np.hstack(action_n)], [rew], [np.hstack(new_obs_n)], [done], terminal)

    def preupdate(self):
        pass

    def update(self, train_step):
        #i=0
        for i,maddpg in enumerate(self.maddpgs):
            bt = maddpg.sample()
            bs, single_bs, ba, br, bs_, single_bs_, bd = maddpg.update_1(i,bt,train_step)
            #i=i+1
            total_a_=[]
            #bs_ = bs_.tolist()
            for j,obs_n in enumerate(bs_):
                #obs_n = np.array(obs_n)
                obs_n = obs_n.reshape(self.nb_agent, maddpg.s_dim)
                total_a_.append(np.hstack([maddpg_.action_([obs]) for obs, maddpg_ in zip(obs_n, self.maddpgs)]))
            total_a_ = np.hstack(total_a_)
            total_a_ = total_a_.reshape(j+1, maddpg.a_dim*self.nb_agent)
            maddpg.update_2(bs, single_bs, ba, br, bs_, single_bs_, total_a_, bd, train_step)
            #ba_ = self.action(self, bs_)


class MADDPG(object):
    def __init__(self, n_b_agent, a_dim, s_dim, a_bound=1, gamma=0.95, tau=0.01, lr_a=1e-2, lr_c=1e-2, memory_size=100000, batch_size=64, scope=""):
        self.nb_agent = n_b_agent
        self.memory = np.zeros((memory_size, s_dim*2*self.nb_agent+a_dim*self.nb_agent+1+1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_filled = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.total_a = tf.placeholder(tf.float32, [None, self.a_dim*self.nb_agent], 'a')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.total_a_ = tf.placeholder(tf.float32, [None, self.a_dim*self.nb_agent], 'a_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.D = tf.placeholder(tf.float32, [None, 1], 'done')
        self.scope = scope

        with tf.variable_scope('Actor'):
            self.a, self.pre_a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_, *_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.total_a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, self.total_a_, scope='target', trainable=False)

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
        self.sess.run(tf.global_variables_initializer())

    def action(self, s):
        #print(s)
        if len(np.hstack(s).shape)==1:
            s = s[0]
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        else:
            return self.sess.run(self.a, {self.S: s})

    def action_(self,s_):
        return self.sess.run(self.a_, {self.S_: s_})

    def preupdate(self):
        pass

    def sample(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        indices = np.random.choice(self.memory_filled, size=self.batch_size)
        bt = self.memory[indices, :]
        return bt

    def update_1(self, i, batch,train_step):
        bt = batch
        bs = bt[:, :self.s_dim*self.nb_agent]
        single_bs = bt[:, self.s_dim*i:self.s_dim*(i+1)]
        ba = bt[:, self.s_dim*self.nb_agent: (self.s_dim+self.a_dim)*self.nb_agent]
        br = bt[:, -self.s_dim*self.nb_agent-2: -self.s_dim*self.nb_agent-1]
        bs_ = bt[:, -self.s_dim*self.nb_agent-1: -1]
        single_bs_ = bt[:, -self.s_dim*(16-i)-1: -self.s_dim*(15-i)-1]
        bd = bt[:, -1:]
        return bs, single_bs, ba, br, bs_, single_bs_, bd

    def update_2(self, bs, single_bs, ba, br, bs_, single_bs_, total_a_, bd, train_step):
        #print(single_bs.shape())
        self.sess.run(self.atrain, {self.S: single_bs, self.total_a: ba})
        self.sess.run(self.ctrain, {self.S: single_bs, self.total_a: ba, self.R: br, self.S_: single_bs_, self.total_a_: total_a_, self.D: bd})

    def experience(self, s, a, r, s_, done, terminal):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        self.memory_filled = min(self.memory_filled + 1, self.memory_size)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=None, name='a', trainable=trainable)
            u = tf.random_uniform(tf.shape(a))
            u = tf.nn.softmax(a - tf.log(-tf.log(u)), axis=-1)
            #print("{}".format(u),'yinggaishierweidecaidui')
            return u, a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim*self.nb_agent, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

