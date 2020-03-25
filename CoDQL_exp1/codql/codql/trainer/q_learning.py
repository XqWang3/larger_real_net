import os
import tensorflow as tf
import numpy as np

from . import base
from . import tools
from . import M_base


class DQN(base.ValueNet):
    def __init__(self, nb_agent, a_dim, s_dim, obs_space, eps=1.0, update_every=5, memory_size=2**10, batch_size=1024):

        super().__init__(nb_agent, a_dim, s_dim, obs_space, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(obs_space, a_dim, memory_size, batch_size)
        self.update_every = update_every
        #self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        #batch_num = self.replay_buffer.get_batch_num()

        for i in range(60):
            obs, obs_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            #print('obs',obs)
            target_q = self.calc_target_q(obs=obs_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=obs, target_q=target_q, acts=actions, masks=masks)

            if i % self.update_every == 0:
                self.update()

            #if i % 50 == 0:
            #    pass
                #print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))


class MFQ(base.ValueNet):
    def __init__(self, nb_agent, a_dim, s_dim, obs_space, eps=1.0, update_every=5, memory_size=2**10, batch_size=1024):

        super().__init__(nb_agent, a_dim, s_dim, obs_space, use_mf=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': obs_space,
            #'feat_shape': self.feature_space,
            'act_n': a_dim,
            'use_mean': True
            #'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()#
        #batch_name = self.replay_buffer.get_batch_num()

        for i in range(60):
            #print(batch_name)#
            obs, me_state, acts, act_prob, obs_next, me_state_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, rewards=rewards, dones=dones, mean_s=me_state_next, prob=act_prob_next)
            loss, q = super().train(state=obs, target_q=target_q, mean_s=me_state, prob=act_prob, acts=acts, masks=masks)

            if i % self.update_every == 0:
                self.update()

        #if i % 50 == 0:
        #    pass
                #print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))

class M_MFQ(M_base.ValueNet):
    def __init__(self, nb_agent, a_dim, s_dim, obs_space, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):

        super().__init__(nb_agent, a_dim, s_dim, obs_space, use_mf=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': obs_space,
            #'feat_shape': self.feature_space,
            'act_n': a_dim,
            'use_mean': True
            #'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            #print(batch_name)
            obs, acts, act_prob, obs_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            #target_q = self.calc_target_q_M(obs=obs_next, rewards=rewards, dones=dones, prob=act_prob_next)
            target_q_value_1, target_q_value_M = self.calc_target_q(obs=obs_next, rewards=rewards, dones=dones, prob=act_prob_next)
            #target_q_value_1 = self.calc_target_q(obs=obs_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=obs, target_q_1=target_q_value_1, target_q_M=target_q_value_M, prob=act_prob, acts=acts, masks=masks)
            #loss, q = super().train(state=obs, target_q_1=target_q_value_1, prob=act_prob, acts=acts, masks=masks)
            if i % self.update_every == 0:
                self.update()