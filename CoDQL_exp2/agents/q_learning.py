import os
import tensorflow as tf
import numpy as np

#from . import base
from . import base_addnet
from . import tools

class DQN(base_addnet.ValueNet):
    #def __init__(self, nb_agent, a_dim, s_dim, obs_space, eps=1.0, update_every=5, memory_size=2**10, batch_size=1024):
    def __init__(self, nb_agent, a_dim, s_dim, s_dim_wave, s_dim_wait, config, doubleQ=True):
        #super().__init__(nb_agent, a_dim, s_dim, obs_space)
        super().__init__(nb_agent, a_dim, s_dim, s_dim_wave, s_dim_wait, doubleQ=True)

        memory_size = config.getint('memory_size')
        batch_size = config.getint('batch_size')
        self.reward_clip = config.getfloat('reward_clip')  # 2
        self.reward_norm = config.getfloat('reward_norm')  # 3000
        self.replay_buffer = tools.MemoryGroup((s_dim,), s_dim_wave, s_dim_wait, a_dim, memory_size, batch_size)
        self.update_every = config.getint('update_every')  # 1

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        #batch_num = self.replay_buffer.get_batch_num()
        for i in range(30):
            obs, obs_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            #print('obs',obs)
            target_q = self.calc_target_q(obs=obs_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=obs, target_q=target_q, acts=actions, masks=masks)
            if i % self.update_every == 0:
                self.update()

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=1000080):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))
        return True


#class MFQ(base.ValueNet):
class MFQ(base_addnet.ValueNet):
    def __init__(self, nb_agent, a_dim, s_dim, s_dim_wave, s_dim_wait, config):
        super().__init__(nb_agent, a_dim, s_dim, s_dim_wave, s_dim_wait, use_mf=True)
        
        memory_size = config.getint('memory_size')
        batch_size = config.getint('batch_size')
        memory_config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': (s_dim,),
            's_dim_wave': s_dim_wave,
            's_dim_wait': s_dim_wait,
            'act_n': a_dim,
            'use_mean': True
            #'sub_len': sub_len
        }

        #self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**memory_config)
        self.update_every = config.getint('update_every') #1
        self.reward_clip = config.getfloat('reward_clip') #2
        self.reward_norm = config.getfloat('reward_norm') #3000

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()#
        #batch_name = self.replay_buffer.get_batch_num()
        for i in range(30):
            #print(batch_name)#
            #obs, me_state, acts, act_prob, obs_next, me_state_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            #target_q = self.calc_target_q(obs=np.array(obs_next), rewards=rewards, dones=dones, mean_s=me_state_next, prob=act_prob_next)
            #loss, q = super().train(state=obs, target_q=target_q, mean_s=me_state, prob=act_prob, acts=acts, masks=masks)
            obs, acts, act_prob, obs_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=np.array(obs_next), rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=obs, target_q=target_q, prob=act_prob, acts=acts, masks=masks)

            if i % self.update_every == 0:
                self.update()

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "CoDQL_{}".format(step))
        saver.save(self.sess, file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=1000080):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "CoDQL_{}".format(step))
        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))
        return True
        