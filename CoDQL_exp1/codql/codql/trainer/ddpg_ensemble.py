import tensorflow as tf
from maddpg.trainer.ddpg import DDPG

class DDPGEN:

    def __init__(self, nb_agent, share_params=True, **kwargs):
        
        self.nb_agent = nb_agent
        self.ddpgs = list()
        if not share_params:
            for i in range(self.nb_agent):
                scope_name = 'ddpg_{}'.format(i)
                with tf.variable_scope(scope_name) as sc:
                    self.ddpgs.append(DDPG(scope=scope_name, **kwargs))
        else:
            reuse=False
            for i in range(self.nb_agent):
                scope_name = 'ddpg'
                with tf.variable_scope(scope_name, reuse=reuse) as sc:
                    self.ddpgs.append(DDPG(scope=scope_name, **kwargs))
                    reuse = True

    def action(self, obs_n):
        #print('a,u')
        return [ddpg.noise_action([obs]) for obs, ddpg in zip(obs_n, self.ddpgs)]

    def experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal):
        for ddpg, obs, action, rew, new_obs, done in zip(self.ddpgs, obs_n, action_n, rew_n, new_obs_n, done_n):
            ddpg.experience([obs], [action], [rew], [new_obs], [done], terminal)

    def preupdate(self):
        pass

    def update(self, train_step):
        for ddpg in self.ddpgs:
            ddpg.update(train_step)
    
