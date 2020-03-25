import os
import tensorflow as tf
from .ddpg import DDPG

class DDPGEN:
    def __init__(self, nb_agent, share_params=True, **kwargs):
        self.nb_agent = nb_agent #49
        self.ddpgs = list()
        self.sess = tf.Session()
        if not share_params:
            for i in range(self.nb_agent):
                scope_name = 'ddpg_{}'.format(i)
                with tf.variable_scope(scope_name) as sc:
                    self.ddpgs.append(DDPG(sess=self.sess, scope=scope_name, **kwargs))
        else:
            reuse=False
            for i in range(self.nb_agent):
                self.scope_name = 'ddpg'
                with tf.variable_scope(self.scope_name, reuse=reuse) as sc:
                    self.ddpgs.append(DDPG(sess=self.sess, scope=self.scope_name, **kwargs))
                    reuse = True
        self.sess.run(tf.global_variables_initializer())

    def action(self, obs_n):
        #print('a,u')
        return [ddpg.noise_action([obs]) for obs, ddpg in zip(obs_n, self.ddpgs)]

    def test_action(self, obs_n):
        #print('a,u')
        return [ddpg.action([obs]) for obs, ddpg in zip(obs_n, self.ddpgs)]

    def experience(self, obs_n, action_n, rew_n, new_obs_n, done_n):
        for ddpg, obs, action, rew, new_obs, done in zip(self.ddpgs, obs_n, action_n, rew_n, new_obs_n, done_n):
            ddpg.experience([obs], [action], [rew], [new_obs], [done])

    def preupdate(self):
        pass

    def update(self, train_step=1):
        for ddpg in self.ddpgs:
            ddpg.update(train_step)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "ddpg_{}".format(step))
        saver.save(self.sess, file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=1000080):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "ddpg_{}".format(step))
        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))
        return True
    
