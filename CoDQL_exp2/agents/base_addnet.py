import tensorflow as tf
import numpy as np

class ValueNet:
    def __init__(self, nb_agent, a_dim, s_dim, obs_wave, obs_wait, use_mf=False, doubleQ=True, learning_rate=1e-4, tau=0.99, gamma=0.95):
        self.nb_agent = nb_agent
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.obs_wave = obs_wave
        #print('(obs_wave,):',self.obs_wave, type(self.obs_wave))
        self.obs_wait = obs_wait
        #print('(obs_wait,):', self.obs_wait, type(self.obs_wait))
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.use_mf = use_mf  # trigger of using mean field
        self.lr= learning_rate
        self.tau = tau
        self.gamma = gamma
        self.doubleQ = doubleQ

        with tf.variable_scope("ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            self.obs_input = tf.placeholder(tf.float32, (None,) + (str(obs_wave),), name="Obs-wave-Input")
            self.feat_input = tf.placeholder(tf.float32, (None,) + (str(obs_wait),), name="Feat-wait-Input")
            self.mask = tf.placeholder(tf.float32, shape=(None,), name='Terminate-Mask')

            if self.use_mf:
                self.act_prob_input = tf.placeholder(tf.float32, (None, self.a_dim), name="Act-Prob-Input")
                ##############mean_state
                #self.mean_wave_st = tf.placeholder(tf.float32, (None,) + (str(obs_wave),), name='Mean-state')
                #self.mean_wait_st = tf.placeholder(tf.float32, (None,) + (str(obs_wait),), name='Mean-state')


            # TODO: for calculating the Q-value, consider softmax usage
            self.act_input = tf.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(self.act_input, depth=self.a_dim, on_value=1.0, off_value=0.0)

            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                self.e_q = self._construct_net()
                self.e_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q = self._construct_net()
                self.t_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op = [tf.assign(self.t_variables[i],
                                            self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                    for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                self.target_q_input = tf.placeholder(tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.reduce_sum(tf.square(self.target_q_input - self.e_q_max) * self.mask) / tf.reduce_sum(self.mask)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def _construct_net(self, trainable=True, reuse=False):
        net1 = tf.layers.dense(self.obs_input, 128, activation=tf.nn.relu, name='Dense-Obs1', trainable=trainable)
        net2 = tf.layers.dense(self.feat_input, 128, activation=tf.nn.relu, name='Dense-Obs2', trainable=trainable)
        net = tf.concat([net1,net2],1)
        if self.use_mf:
            prob_emb = tf.layers.dense(self.act_prob_input, units=64, activation=tf.nn.relu, name='Prob-Emb')
            h_act_prob = tf.layers.dense(prob_emb, units=64, activation=tf.nn.relu, name="Dense-Act-Prob")
            ##############mean_state
            #mean_wave_st_emb = tf.layers.dense(self.mean_wave_st, units=64, activation=tf.nn.relu)
            #h_mean_wave_st = tf.layers.dense(mean_wave_st_emb, units=64, activation=tf.nn.relu)
            #mean_wait_st_emb = tf.layers.dense(self.mean_wait_st, units=64, activation=tf.nn.relu)
            #h_mean_wait_st = tf.layers.dense(mean_wait_st_emb, units=64, activation=tf.nn.relu)

            #net = tf.concat([net, h_act_prob, h_mean_wave_st, h_mean_wait_st], axis=1)
            net = tf.concat([net, h_act_prob], axis=1)
            #net = tf.concat([net, prob_emb, mean_wave_st_emb, mean_wait_st_emb], axis=1)

        dense2 = tf.layers.dense(net, 256, activation=tf.nn.relu, name='Dense2', trainable=trainable)
        #dense2 = tf.layers.dense(dense2, units=256, activation=tf.nn.relu, name="Dense-Out")  #默认trainable=True
        dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out1")
        #dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out2")

        q = tf.layers.dense(dense2, units=self.a_dim, name="Q-Value")
        return q


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        feed_dict = {
            self.obs_input: kwargs['obs'][:, :self.obs_wave],
            self.feat_input: kwargs['obs'][:, self.obs_wave:]
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']
            #feed_dict[self.mean_wave_st] = kwargs['mean_s'][:, :self.obs_wave]
            #feed_dict[self.mean_wait_st] = kwargs['mean_s'][:, self.obs_wave:]

        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        if self.doubleQ:
            act_idx = np.argmax(e_q, axis=1)
            q_values = t_q[np.arange(len(t_q)), act_idx]
        else:
            q_values = np.max(t_q, axis=1)  # the natural DQN

        target_q_value = kwargs['rewards'] + (1. - kwargs['dones']) * q_values.reshape(-1) * self.gamma

        return target_q_value

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    def act(self, **kwargs):
        feed_dict = {
            self.obs_input: kwargs['state'][:, :self.obs_wave],
            self.feat_input: kwargs['state'][:, self.obs_wave:],
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'])
            feed_dict[self.act_prob_input] = kwargs['prob']
            ##############mean_state
            #feed_dict[self.mean_wave_st] = kwargs['mean_s'][:, :self.obs_wave]
            #feed_dict[self.mean_wait_st] = kwargs['mean_s'][:, self.obs_wave:]

        q_values = self.sess.run(self.e_q, feed_dict=feed_dict)
        #print('q_values:\n',q_values)

        switch = np.random.uniform()  #默认0-1之间的随机数
        if switch < kwargs['eps']:   #[0, 700, 1400], [1, 0.2, 0.05]
            actions = np.random.choice(self.a_dim, len(kwargs['state'])).astype(np.int32)
        else:
            actions = np.argmax(q_values, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        feed_dict = {
            #self.obs_input: kwargs['state'],
            self.obs_input: kwargs['state'][:, :self.obs_wave],
            self.feat_input: kwargs['state'][:, self.obs_wave:],
            #self.feat_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
            self.mask: kwargs['masks']
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']
            #feed_dict[self.mean_wave_st] = kwargs['mean_s'][:, :self.obs_wave]
            #feed_dict[self.mean_wait_st] = kwargs['mean_s'][:, self.obs_wave:]

        feed_dict[self.act_input] = kwargs['acts']
        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
