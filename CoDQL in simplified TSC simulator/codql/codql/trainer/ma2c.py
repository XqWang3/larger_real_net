import os
from . import utils
import logging
import multiprocessing as mp
import numpy as np
import tensorflow as tf

class MA2C:
    def __init__(self, nb_agent, a_dim, s_dim, obs_space, seed=0):
        self.name = 'ma2c'
        self.agents = []
        self.n_agent = nb_agent
        self.a_dim = a_dim  #2
        self.s_dim = s_dim*5  #12*5
        self.obs_space = obs_space #(12,)
        self.n_step = 60
        n_f = a_dim*4
        # init tf
        #tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        #self.saver = tf.train.Saver(max_to_keep=5)
        self.nodes_finger = [np.array([0.5 for _ in range(a_dim)]) for _ in range(nb_agent)]
        #print('self.nodes_finger:',self.nodes_finger)
        self.neighbours_n = [[1, 4], [0, 2, 5], [1, 3, 6], [2, 7],
                        [0, 5, 8], [1, 4, 6, 9], [2, 5, 7, 10], [3, 6, 11],
                        [4, 9, 12], [5, 8, 10, 13], [6, 9, 11, 14], [7, 10, 15],
                        [8, 13], [9, 12, 14], [10, 13, 15], [11, 14]]
        self.policy_ls = []
        for i in range(nb_agent):
            self.policy_ls.append(self._init_policy(s_dim*5, a_dim, n_f, self.n_step, agent_name=str(i)))
        self._init_train()
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_f, n_step, agent_name=None):
        '''n_fw = 128
        n_ft = 64
        n_lstm = 64
        n_fp = 64'''
        policy = FPLstmACPolicy(n_s, n_a, n_f, n_step, n_fc_wave=128, n_fc_fp=64, n_lstm=64, name=agent_name)
        return policy

    def _init_train(self):
        # init loss
        v_coef = 0.5 #model_config.getfloat('value_coef')#value_coef = 0.5
        max_grad_norm = 40 #model_config.getfloat('max_grad_norm')#max_grad_norm = 40
        alpha = 0.99 #model_config.getfloat('rmsp_alpha')#rmsp_alpha = 0.99
        epsilon = 1e-5 #model_config.getfloat('rmsp_epsilon')#rmsp_epsilon = 1e-5
        gamma = 0.99 #model_config.getfloat('gamma')#gamma = 0.99
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
            self.trans_buffer_ls.append(utils.OnPolicyBuffer(gamma))

    def backward(self, R_ls):
        #cur_lr = self.lr_scheduler.get(self.n_step) #canstant
        cur_lr = 5e-4
        #cur_beta = self.beta_scheduler.get(self.n_step) #canstant
        cur_beta = 0.01
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)
    def only_clear_buffer(self):
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].reset()

    def forward(self, obs, done, out_type='pv'):
        if len(out_type) == 1:
            out = []
        elif len(out_type) == 2:
            out1, out2 = [], []
        for i in range(self.n_agent):
            cur_out = self.policy_ls[i].forward(self.sess, obs[i], done, out_type)
            if len(out_type) == 1:
                out.append(cur_out)
            else:
                out1.append(cur_out[0])
                out2.append(cur_out[1])
        if len(out_type) == 1:
            return out
        else:
            return out1, out2

    def add_transition_push(self, obs, actions, rewards, values, done):
        #print('obs:',obs)
        #print('obs[0]:',obs[0])
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i], rewards[i], values[i], done)

    def action(self, prev_ob):
        ob = prev_ob
        policy, value = self.forward(ob,done=False)
        #print('policy:',policy)
        #print('value:',value)
        self.update_fingerprint(policy)
        action = []
        for pi in policy:
            action.append(np.random.choice(np.arange(len(pi)), p=pi))
        return action, value
    def update_fingerprint(self, policy):
        for node, pi in enumerate(policy):
            #print('pi:', pi)
            #print('np.array(pi)', np.array(pi))
            #print('np.array(pi)[:-1]:',np.array(pi)[:-1])
            self.nodes_finger[node] = np.array(pi)

    def get_ma2c_obs_n(self, obs_n):
        obs_nn = obs_n
        ma2c_obs_n = []
        for i in range(self.n_agent):
            cur_state = [obs_nn[i]]
            j=0
            for nbs in self.neighbours_n[i]:
                j+=1
                cur_state.append(obs_nn[nbs] * 0.75)
            while j<4:   #ensure that 4 neighbours
                j+=1
                cur_state.append(obs_nn[0]*0)
            j=0
            for nbs in self.neighbours_n[i]:
                j+=1
                cur_state.append(self.nodes_finger[nbs])
            while j<4:
                j+=1
                cur_state.append(np.array([0,0])) #when the number of neighbour < 4, we set the virtual neighbours' finger to np.array([0,0])
            ma2c_obs_n.append(np.concatenate(cur_state))
        return ma2c_obs_n

    def get_ma2c_r_n(self, r_n):
        r_nn = r_n
        ma2c_r_n = []
        for i in range(self.n_agent):
            cur_r = r_nn[i]
            for nbs in self.neighbours_n[i]:
                cur_r += r_nn[nbs]*0.75
            ma2c_r_n.append(cur_r)
        return ma2c_r_n

    def explore(self, prev_ob):
        ob = prev_ob
        for _ in range(1024):
            policy, value = self.forward(ob)
                # need to update fingerprint before calling step
            #self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            next_ob, reward, done, global_reward = self.env.step(action)
            #rewards.append(global_reward)
            #global_step = self.global_counter.next()
            #self.cur_step += 1
            self.add_transition(ob, action, reward, value, done)

            if done:
                break
            ob = next_ob
        R = self.forward(ob, False, 'v')

        return ob, done, R, _ #rewards

class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None):

        self.env = env
        self.agent = self.env.agent   #ma2c
        self.model = model
        self.sess = self.model.sess

    '''def explore(self, prev_ob):
        ob = prev_ob
        #rewards = []
        for _ in range(1024):
            policy, value = self.model.forward(ob)
                # need to update fingerprint before calling step
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            next_ob, reward, done, global_reward = self.env.step(action)
            #rewards.append(global_reward)
            #global_step = self.global_counter.next()
            #self.cur_step += 1
            self.model.add_transition(ob, action, reward, value, done)

            if done:
                break
            ob = next_ob
        if done:
            R = 0 if self.agent == 'a2c' else [0] * self.model.n_agent
        else:
            R = self.model.forward(ob, False, 'v')

        return ob, done, R, _ #rewards'''
    '''def perform(self, test_ind):
        ob = self.env.reset(test_ind=test_ind)
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('a2c'):
                policy = self.model.forward(ob, False, 'p')
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        action.append(np.argmax(np.array(pi)))
            else:
                action, _ = self.model.forward(ob)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward'''

    def run(self):
        while not self.global_counter.should_stop():
            ob = self.env.reset()
            _, _, R, _ = self.explore(ob)
            self.model.backward(R)

class FPLstmACPolicy:
    def __init__(self, n_s, n_a, n_f, n_step, policy_name='fplstm', n_fc_wave=128, n_fc_fp=32, n_lstm=64, name=None):
        self.name = policy_name
        if name is not None:
            # for multi-agent system
            self.name += '_' + str(name)
        self.n_s = n_s
        self.n_a = n_a
        self.n_step = n_step   #60

        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_fp = n_fc_fp
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s+ n_f]) # forward 1-step
        #self.ob_fw = tf.placeholder(tf.float32, [1, n_s])  # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s+ n_f]) # backward n-step
        #self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s])  # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        #print('ob[0,:]:',ob[0,:])
        #print('ob[0,:self.n_s]:',ob[0,:self.n_s])
        h0 = utils.fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = utils.fc(ob[:, self.n_s:], out_type + '_fcf', self.n_fc_fp)
        h = tf.concat([h0, h1], 1)
        #h = h0
        h, new_states = utils.lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states
    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = utils.fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = utils.fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)
    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs
    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta):
        outs = sess.run(self._train,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))