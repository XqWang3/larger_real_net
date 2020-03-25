import sys, os
import os.path as osp
this_abs_path = osp.abspath(__file__)
experiment_dir = osp.dirname(this_abs_path)
maddpg_dir = osp.dirname(experiment_dir)
root_dir = osp.dirname(maddpg_dir)
multiagent_dir = osp.join(root_dir, "multiagent")
models_dir = osp.join(root_dir, "models")
logs_dir = osp.join(root_dir, "flow22_doublering")

sys.path.extend([
    experiment_dir,
    maddpg_dir,
    root_dir,
    multiagent_dir
])

import numpy as np
import tensorflow as tf
import time

from logger import Logger
import maddpg.common.tf_util as U
import maddpg.common.tf_module as M
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.ddpg_ensemble import DDPGEN
from maddpg.trainer.maddpg_self import MADDPGEN
from traffic_visualization.traffic_4t import TrafficEnv
from maddpg.trainer.q_learning import MFQ, DQN, M_MFQ
from maddpg.trainer.Double_q_learning import MFDoubleQ, DoubleDQN
from maddpg.trainer.ma2c import MA2C
from maddpg.trainer.ac import MFAC, ActorCritic
from lib.configs import load_cfg, cfg
import lib.configs as lc

import serialization_utils as sut

def make_env(scenario_name):
    if scenario_name == 'group_traffic':
        env = TrafficEnv(add_newCar=5,flag_traffic_flow=2)   #randomflow:5,2; doublering:4,1; fourring:3,3.
        return env
    else:
        print('Error, no environment')
        exit(1)

def get_trainers(env, num_adversaries, obs_shape_n, arglist, env_n):
    model_q = M.mlp_model_Q
    model_p = M.mlp_model   #网络层数改了
    def get_space_vector_len(space):
        from gym import spaces
        if type(space) is spaces.Discrete:
            return space.n
        elif type(space) is spaces.Box:
            assert len(space.low.shape) == 1
            return space.low.shape[0]
        else:
            print(type(space))
            raise NotImplementedError
    s_dim = get_space_vector_len(env.observation_space[0])
    print(s_dim)  #traffic 12
    obs_space = env.observation_space[0].low.shape
    print(obs_space)
    a_dim = get_space_vector_len(env.action_space[0])
    print(a_dim) #traffic 2
    h_dim = lc.cfg.latent_dims
    print(h_dim)  #traffic 8
    num_agents = len(env.observation_space)
    print(num_agents) #traffic 16
    if arglist.simple_ddpg:   #ddpg
        trainers = DDPGEN(nb_agent=num_agents, share_params=arglist.simple_ddpg_share, a_dim=a_dim, s_dim=s_dim)
    elif arglist.maddpg_self:   #maddpg, self-code
        trainers = MADDPGEN(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim)
    elif arglist.use_ma2c:
        trainers = MA2C(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim, obs_space=obs_space)
    elif arglist.algo == 'mfq':
        print("this is mfq algorithm.")
        trainers = MFQ(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim, obs_space=obs_space, update_every=1, memory_size=100000)
    elif arglist.algo == 'dqn':
        print("this is dqn algorithm.")
        trainers = DQN(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim, obs_space=obs_space, update_every=1, memory_size=100000)
    elif arglist.algo == 'm_mfq':
        print("this is M_mfq algorithm.")
        trainers = M_MFQ(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim, obs_space=obs_space, update_every=1,
                       memory_size=100000)
    elif arglist.algo == 'mf_Double_q':
        print("this is mf_Double_q algorithm.")
        trainers = MFDoubleQ(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim, obs_space=obs_space, update_every=1, memory_size=100000)
    elif arglist.algo == 'Double_dqn':
        print("this is Double_dqn algorithm.")
        trainers = DoubleDQN(nb_agent=num_agents, a_dim=a_dim, s_dim=s_dim, obs_space=obs_space, update_every=1, memory_size=100000)
    elif arglist.algo == 'mfac':
        sess = tf.Session()
        print("this is mfac algorithm.")
        trainers = MFAC(sess, arglist.algo+'_net', a_dim=a_dim, s_dim=s_dim, obs_space=obs_space)
        sess.run(tf.global_variables_initializer())
    elif arglist.algo == 'ac':
        sess = tf.Session()
        print("this is ac algorithm.")
        trainers = ActorCritic(sess, arglist.algo+'_net', a_dim=a_dim, s_dim=s_dim, obs_space=obs_space)
        sess.run(tf.global_variables_initializer())
    else:
        print("this is MADDPG algorithm.")  #openai-ddpg-maddpg
        trainers = []
        trainer = MADDPGAgentTrainer
        for i in range(num_adversaries):
            trainers.append(trainer(
                "agent_%d" % i, model_q, model_p, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.adv_policy=='ddpg')))
        for i in range(num_adversaries, env_n):
            trainers.append(trainer(
                "agent_%d" % i, model_q, model_p, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
mkdir(models_dir)
mkdir(logs_dir)

def train(arglist):

    import tensorflow as tf
    tf.reset_default_graph()
    data_collector = dict()

    def upload(key, val):
        if key not in data_collector:
            data_collector[key] = list()
        data_collector[key].append(val)

    env = make_env(arglist.scenario)
    mkdir(arglist.models_dir)
    exp_dir = osp.join(arglist.models_dir, arglist.exp_name)
    mkdir(exp_dir)

    if arglist.replay >= 0:

        video_filename = 'episode_{}.npz'.format(arglist.replay)
        video_serialization = sut.npz_to_serialization(osp.join(exp_dir, video_filename))
        env.reset()

        idx = 0
        len_episode = sut.len_serialization(video_serialization)
        # from IPython import embed; embed()
        while True:
            frame = sut.index_serialization(video_serialization, idx)
            idx = (idx + 1) % len_episode
            env.deserialize(frame)
            time.sleep(0.1)
            env.render(mode="inhuman")

    else:
        with U.single_threaded_session():
            
            if arglist.tf_log:
                mkdir(arglist.tf_dir)
                logdir = osp.join(arglist.tf_dir, arglist.exp_name)
                mkdir(logdir)
                logstep = 120
                logger = Logger(logdir)

            # Create agent trainers
            env_n = env.n**2 if type(env) == TrafficEnv else env.n
            obs_shape_n = [env.observation_space[i].shape for i in range(env_n)]
            #print(obs_shape_n)  [(12,), (12,), (12,), ******]
            num_adversaries = min(env_n, arglist.num_adversaries) #0
            trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist, env_n)
            print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

            # Initialize
            U.initialize()

            # Load previous results, if necessary
            if arglist.load_dir == '':
                arglist.load_dir = osp.join(exp_dir, 'model')
            if arglist.restore:
                print('Loading previous state...')
                # print(arglist.load_dir)
                U.load_state(arglist.load_dir)

            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env_n)]  # individual agent reward

            train_step = 0
            t_start = time.time()

            print('Starting iterations...')
            # from IPython import embed; embed()

            env.reset()
            video_log_proto = sut.create_serialization(env.serialize())
            from copy import deepcopy

            train_step = 0

            import matplotlib.pyplot as plt

            #total_re = 0
            from tqdm import tqdm
            for episode_id in tqdm(range(arglist.num_episodes)):

                obs_n = env.reset()
                #print(type(obs_n[0]))   #Output: <class 'numpy.ndarray'>
                video_episode = arglist.video and (episode_id + 1) % arglist.video_rate == 0
                video_log = deepcopy(video_log_proto)

                if arglist.algo in ['mfq', 'dqn', 'ac', 'mfac', 'm_mfq', 'mf_Double_q', 'Double_dqn']:
                    former_act_prob = np.zeros((1, 2)) #,np.zeros((1, 2))]  #a_dim=2
                    #print(former_act_prob)
                    ##############mean_state
                    mean_state = [sum(state)/len(state) for state in zip(*obs_n)]
                    #print(mean_state)

                if arglist.use_ma2c:
                    obs_n = trainers.get_ma2c_obs_n(obs_n)
                    #print('ma2c_obs_n',len(obs_n)) #16

                for step_id in range(arglist.max_episode_len):
                    # eps = magent.utility.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05])
                    #eps = tf.train.piecewise_constant(step_id, [20, 40], [1.0, 0.2, 0.05])
                    #eps = U.piecewise_decay(step_id, [0, 20, 40], [1, 0.2, 0.05])
                    eps = 0.05
                    #print(step_id, '\n',eps)
                    # get action
                    if arglist.simple_ddpg:
                        #print(obs_n)
                        action_n = trainers.action(obs_n)
                        #print(action_n)#[array([0.28251922, 0.3178802 ]), array([-0.44342542,  0.32045864]), ......
                    elif arglist.maddpg_self:
                        action_n = trainers.action(obs_n)
                        #print(action_n)
                    elif arglist.use_ma2c:
                        action_n, value = trainers.action(obs_n)
                        #print('use_ma2c_action_n:',action_n)
                    elif arglist.algo in ['mfq', 'dqn', 'ac', 'mfac', 'm_mfq', 'mf_Double_q', 'Double_dqn']:
                        former_act_prob = np.tile(former_act_prob, (len(obs_n), 1))
                        ##############mean_state
                        mean_state = np.tile(mean_state, (len(obs_n), 1))
                        #print(former_act_prob)
                        #action_n = trainers.act(state=obs_n, prob=former_act_prob, eps=eps)
                        #print(action_n)  #[0 1 1 0 0 1 1 1 1 0 0 1 0 1 0 0]
                        ##############mean_state
                        action_n = trainers.act(state=obs_n, mean_s=mean_state, prob=former_act_prob, eps=eps)
                    else:
                        action_n = [agent.noise_action(obs) for agent, obs in zip(trainers, obs_n)]
                        #print('action_n:\n',action_n) #[array([9.9988675e-01, 1.1327704e-04], dtype=float32), array([0.7924716 , 0.20752843], dtype=float32),***]
                    _action_n = deepcopy(action_n)

                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)

                    if arglist.use_ma2c:
                        for i, rew in enumerate(rew_n):
                            episode_rewards[-1] += rew
                            agent_rewards[i][-1] += rew
                            upload("reward_{}".format(i), rew)
                        new_obs_n = trainers.get_ma2c_obs_n(new_obs_n)
                        rew_n = trainers.get_ma2c_r_n(rew_n)

                    #==============================================================
                    if arglist.algo=='mfq':
                        #print(rew_n)
                        weight_rew_n = []
                        sum_rew = np.sum(rew_n)
                        for i, rew in enumerate(rew_n):
                            episode_rewards[-1] += rew
                            agent_rewards[i][-1] += rew
                            upload("reward_{}".format(i), rew)
                            # begin calculate weight rew_n
                            weight_rew_n.append((sum_rew - rew)/15 + rew)
                            # end
                        #print('rew_n:{}'.format(rew_n))
                        #print('weight_rew_n:{}'.format(weight_rew_n))

                    #==============================================================
                    action_n = _action_n
                    if not hasattr(done_n, '__iter__'):
                        done_n = [done_n] * len(rew_n)
                    done = all(done_n)

                    terminal = (step_id + 1 == arglist.max_episode_len)
                    # collect experience
                    if arglist.simple_ddpg or arglist.maddpg_self:
                        trainers.experience(obs_n, action_n, rew_n, new_obs_n, done_n, terminal)
                    elif arglist.use_ma2c:
                        #print('obs_n:',len(obs_n),'action_n:',len(action_n),'rew_n:',len(rew_n),'value:',len(value))
                        trainers.add_transition_push(obs_n, action_n, rew_n, value, done)
                    elif arglist.algo in ['mfq', 'dqn', 'ac', 'mfac', 'm_mfq', 'mf_Double_q', 'Double_dqn']:
                        buffer = {
                            'state': obs_n, 'acts': action_n, 'rewards': weight_rew_n, 'dones': done_n}
                        buffer['prob'] = former_act_prob
                        ##############mean_state
                        buffer['mean_sta'] = mean_state
                        mean_state = [sum(state) / len(state) for state in zip(*new_obs_n)]
                        former_act_prob = np.mean(list(map(lambda x: np.eye(2)[x], action_n)), axis=0, keepdims=True)
                        trainers.flush_buffer(**buffer)
                        #print('flush_buffer')
                    else:
                        for i, agent in enumerate(trainers):
                            agent.experience(obs_n[i], action_n[i].astype(np.float64), rew_n[i], new_obs_n[i], done_n[i], terminal)
                    obs_n = new_obs_n

                    '''for i, rew in enumerate(rew_n):
                        episode_rewards[-1] += rew
                        agent_rewards[i][-1] += rew
                        upload("reward_{}".format(i), rew)'''
                    if arglist.use_ma2c:
                        R_n = trainers.forward(obs_n, False, 'v')
                    if done or terminal:
                        if arglist.use_ma2c:
                            R_n = trainers.forward(obs_n, False, 'v')
                        obs_n = env.reset()
                        #episode_rewards[-1]=episode_rewards[-1]/100
                        episode_rewards.append(0)
                        #total_re = total_re+episode_rewards[-2]
                        for a in agent_rewards:
                            a.append(0)

                    # increment global step counter
                    train_step += 1
                    if arglist.tf_log and train_step % 60 == 0:
                        logger.scalar_summary('episode_rewards', np.mean(episode_rewards[-10:]), train_step/60)

                    if video_episode:
                        sut.append_serialization(video_log, env.serialize())

                    # for displaying learned policies
                    if arglist.display:
                        time.sleep(0.1)
                        env.render(mode="inhuman")

                    # update all trainers, if not in display or benchmark mode
                    if arglist.maddpg_self:
                        trainers.preupdate()
                        loss = trainers.update(train_step)
                    if arglist.use_ma2c:
                        pass
                    elif arglist.simple_ddpg or (arglist.algo in ['mfq', 'dqn', 'ac', 'mfac', 'm_mfq', 'mf_Double_q', 'Double_dqn']):
                        #trainers.train()
                        pass
                    else:
                        loss = None
                        for agent in trainers:
                            agent.preupdate()
                        for agent in trainers:
                            loss = agent.update(trainers, train_step)

                if arglist.algo in ['mfq', 'dqn', 'ac', 'mfac', 'm_mfq', 'mf_Double_q', 'Double_dqn']:
                    trainers.train()
                elif arglist.simple_ddpg:
                    for _ in range(60):
                        trainers.preupdate()
                        loss = trainers.update(train_step)
                elif arglist.use_ma2c:
                    trainers.backward(R_n)
                    '''if train_step > 60*10:  #
                        trainers.backward(R_n)  #TODO: The backward will clear the buffer automotially.
                    else:
                        trainers.only_clear_buffer()'''
                else:
                    pass

                if video_episode:
                    sut.serialization_to_npz(video_log, osp.join(exp_dir, 'episode_{}'.format(episode_id + 1)))


            discrete_episode_rewards=[]
            for i,j in enumerate(episode_rewards):
                if i!=0 and i%10==0:
                    discrete_episode_rewards.append(np.mean(episode_rewards[i-10:i]))
            plt.plot(discrete_episode_rewards)
            plt.show()

    env.close()
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
            "Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--cfg", type=str, default='configs/traffic.yaml')
    parser.add_argument("--replay", type=int, default=-1, help="replay episode id. negative id to get into training mode")

    from lib.collections import AttrDict
    parsed = AttrDict(vars(parser.parse_args()))

    args = load_cfg(parsed.cfg)
    args.models_dir = models_dir
    args.tf_dir = logs_dir

    from lib.configs import merge
    merge(parsed, args)

    return args

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)