import tensorflow as tf
import numpy as np

#from magent.gridworld import GridWorld


class ValueNet:
    def __init__(self, nb_agent, a_dim, s_dim, obs_space, update_every=5, use_mf=False, learning_rate=1e-4, tau=0.99, gamma=0.95):
        # assert isinstance(env, GridWorld)
        self.nb_agent = nb_agent
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.obs_space = obs_space
        self._saver = None
        self.sess = tf.Session()

        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 0.1

        self.lr= learning_rate
        self.tau = tau
        self.gamma = gamma

        with tf.variable_scope("M_ValueNet"):
            self.name_scope_M = tf.get_variable_scope().name
            self.obs_input_M = tf.placeholder(tf.float32, (None,) + self.obs_space, name="Obs-Input")
            #self.feat_input = tf.placeholder(tf.float32, (None,) + self.feature_space, name="Feat-Input")
            self.mask_M = tf.placeholder(tf.float32, shape=(None,), name='Terminate-Mask')

            if self.use_mf:
                self.act_prob_input = tf.placeholder(tf.float32, (None, self.a_dim), name="Act-Prob-Input")

            # TODO: for calculating the Q-value, consider softmax usage
            self.act_input_M = tf.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(self.act_input_M, depth=self.a_dim, on_value=1.0, off_value=0.0)

            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                self.e_q_M = self.M_construct_net()
                self.e_variables_M = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q_M = self.M_construct_net()
                self.t_variables_M = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op_M = [tf.assign(self.t_variables_M[i],
                                            self.tau * self.e_variables_M[i] + (1. - self.tau) * self.t_variables_M[i])
                                    for i in range(len(self.t_variables_M))]

            with tf.variable_scope("Optimization"):
                self.target_q_M_input = tf.placeholder(tf.float32, (None,), name="Q-Input")
                self.e_q_M_max = tf.reduce_sum(tf.multiply(self.act_one_hot, self.e_q_M), axis=1)
                self.loss_M = tf.reduce_sum(tf.square(self.target_q_M_input - self.e_q_M_max) * self.mask_M) / tf.reduce_sum(self.mask_M)
                self.train_op_M = tf.train.AdamOptimizer(self.lr).minimize(self.loss_M)

        with tf.variable_scope("ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            self.obs_input = tf.placeholder(tf.float32, (None,) + self.obs_space, name="Obs-Input")
            #self.feat_input = tf.placeholder(tf.float32, (None,) + self.feature_space, name="Feat-Input")
            self.mask = tf.placeholder(tf.float32, shape=(None,), name='Terminate-Mask')

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
                self.target_q_1_input = tf.placeholder(tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.reduce_sum(tf.square(self.target_q_1_input - self.e_q_max) * self.mask) / tf.reduce_sum(self.mask)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def M_construct_net(self, trainable=True, reuse=False):
        net = tf.layers.dense(self.obs_input_M, 64, activation=tf.nn.relu, name='Dense-Obs', trainable=trainable)
        if self.use_mf:
            prob_emb = tf.layers.dense(self.act_prob_input, units=64, activation=tf.nn.relu, name='Prob-Emb')
            h_act_prob = tf.layers.dense(prob_emb, units=64, activation=tf.nn.relu, name="Dense-Act-Prob")
            net = tf.concat([net, h_act_prob], axis=1)
        dense2 = tf.layers.dense(net, 64, activation=tf.nn.relu, name='Dense2', trainable=trainable)
        dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out")
        dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out1")
        #dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out2")
        q = tf.layers.dense(dense2, units=self.a_dim, name="Q-Value")
        return q

    def _construct_net(self, trainable=True, reuse=False):
        net = tf.layers.dense(self.obs_input, 64, activation=tf.nn.relu, name='Dense-Obs', trainable=trainable)
        dense2 = tf.layers.dense(net, 64, activation=tf.nn.relu, name='Dense2', trainable=trainable)
        dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out")
        dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out1")
        #dense2 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu, name="Dense-Out2")
        q = tf.layers.dense(dense2, units=self.a_dim, name="Q-Value")
        return q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'prob', 'dones', 'rewards'}
        """
        t_q_1, e_q_1 = self.sess.run([self.t_q, self.e_q], {self.obs_input: kwargs['obs']})

        feed_dict = {
            self.obs_input_M: kwargs['obs'],
        }
        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        t_q_M, e_q_M = self.sess.run([self.t_q_M, self.e_q_M], feed_dict=feed_dict)
        ##e_q = e_q_1 + e_q_M
        ##t_q = t_q_1 + t_q_M
        act_idx_1 = np.argmax(e_q_1, axis=1)
        act_idx_M = np.argmax(e_q_M, axis=1)
        q_values_1 = t_q_1[np.arange(len(t_q_1)), act_idx_1]
        q_values_M = t_q_M[np.arange(len(t_q_M)), act_idx_M]

        target_q_value_1 = kwargs['rewards'] + (1. - kwargs['dones']) * q_values_1.reshape(-1) * self.gamma
        target_q_value_M = kwargs['rewards'] + (1. - kwargs['dones']) * q_values_M.reshape(-1) * self.gamma

        return  target_q_value_1,target_q_value_M

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)
        self.sess.run(self.update_op_M)

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'prob', 'eps'}
        """
        q_values_1 = self.sess.run(self.e_q, {self.obs_input: kwargs['state']})
        feed_dict = {
            self.obs_input_M: kwargs['state']
        }
        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'])
            feed_dict[self.act_prob_input] = kwargs['prob']

        q_values_M = self.sess.run(self.e_q_M, feed_dict=feed_dict)
        #print('q_values_1:\n',q_values_1)
        #print('q_values_M:\n',q_values_M)
        q_values = q_values_1 + q_values_M
        switch = np.random.uniform()  #默认0-1之间的随机数

        if switch < kwargs['eps']:   #[0, 700, 1400], [1, 0.2, 0.05]
            actions = np.random.choice(self.a_dim, len(kwargs['state'])).astype(np.int32)
        else:
            actions = np.argmax(q_values, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': obs, 'target_q', 'prob', 'acts'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'],
            self.target_q_1_input: kwargs['target_q_1'],
            self.mask: kwargs['masks']
        }
        feed_dict[self.act_input] = kwargs['acts']
        _, loss_1, e_q_1 = self.sess.run([self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)

        feed_dict = {
            self.obs_input_M: kwargs['state'],
            self.target_q_M_input: kwargs['target_q_M'],
            self.mask_M: kwargs['masks']
        }
        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']
        feed_dict[self.act_input_M] = kwargs['acts']
        _, loss_M, e_q_M = self.sess.run([self.train_op_M, self.loss_M, self.e_q_M_max], feed_dict=feed_dict)
        return 0,0#loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
