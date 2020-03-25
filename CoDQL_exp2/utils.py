import itertools
import logging
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd
import subprocess


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        # self.init_test = True

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        # if self.init_test:
        #     self.init_test = False
        #     return True
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    # def update_test(self, reward):
    #     if self.prev_reward is not None:
    #         if abs(self.prev_reward - reward) <= self.delta_reward:
    #             self.stop = True
    #     self.prev_reward = reward

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent #codql;dqn
        self.model = model #MFQ; DQN
        self.sess = self.model.sess
        if self.agent in ['codql', 'dqn', 'ddpg']:
            pass
        else:
            self.n_step = self.model.n_step #batch_size=120
            assert self.env.T % self.n_step == 0
        self.summary_writer = summary_writer
        self.run_test = run_test #False
        self.data = []
        self.output_path = output_path #'./large_network/codql/data'
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        rewards = []
        for _ in range(self.n_step): #120
            if self.agent.endswith('a2c'):
                policy, value = self.model.forward(ob, done)
                # need to update fingerprint before calling step
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    action = np.random.choice(np.arange(len(policy)), p=policy)
                else:
                    action = []
                    for pi in policy:
                        action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action, policy = self.model.forward(ob, mode='explore')
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            if self.agent.endswith('a2c'):
                self.model.add_transition(ob, action, reward, value, done)
            else:
                self.model.add_transition(ob, action, reward, next_ob, done)
            # logging
            if self.global_counter.should_log(): #1W
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            if done:
                break
            ob = next_ob
        if self.agent.endswith('a2c'):
            if done:
                R = 0 if self.agent == 'a2c' else [0] * self.model.n_agent
            else:
                R = self.model.forward(ob, False, 'v')
        else:
            R = 0
        return ob, done, R, rewards

    def perform(self, test_ind, demo=False):
        ob = self.env.reset(gui=demo, test_ind=test_ind)
        # note this done is pre-decision to reset LSTM states!
        done = True
        if self.agent in ['codql', 'dqn', 'greedy']:
            eps = 0 #max
            a_dim = self.env.n_a_ls[0]
            former_act_prob = np.zeros((1, a_dim))
            mean_state = np.mean(ob, axis=0, keepdims=True)[0]
        elif self.agent=='ddpg':
            pass
        else:
            self.model.reset()
        rewards = []
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            elif self.agent.endswith('a2c'):
                policy = self.model.forward(ob, done, 'p')
                print('ma2c_policy:',policy)
                if self.agent == 'ma2c':
                    self.env.update_fingerprint(policy)
                if self.agent == 'a2c':
                    action = np.argmax(np.array(policy))
                else:
                    action = []
                    for pi in policy:
                        action.append(np.argmax(np.array(pi)))
                    print('ma2c_action:',action)
            elif self.agent in ['codql', 'dqn']:
                former_act_prob = np.tile(former_act_prob, (len(ob), 1))
                mean_state = np.tile(mean_state, (len(ob), 1))
                action = self.model.act(state=np.array(ob), mean_s=mean_state, prob=former_act_prob, eps=eps)
            elif self.agent in ['ddpg']:
                action = self.model.test_action(ob)
            else:
                action, _ = self.model.forward(ob)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if self.agent in ['codql', 'dqn']:
                mean_state = np.mean(next_ob, axis=0, keepdims=True)[0]
                former_act_prob = np.mean(list(map(lambda x: np.eye(self.env.n_a_ls[0])[x], action)), axis=0, keepdims=True)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run_thread(self, coord):
        '''Multi-threading is disabled'''
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward = self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            if self.agent.endswith('a2c'):
                self.model.backward(R, self.summary_writer, global_step)
            else:
                self.model.backward(self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                self.env.terminate()
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return

    def run(self):
        while not self.global_counter.should_stop():
            # test
            if self.run_test and self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    mean_reward, std_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(mean_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'avg_reward': mean_reward,
                           'std_reward': std_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
            # train
            self.env.train_mode = True
            ob = self.env.reset() #[agent0=[np.concatenate([node.wave_state, node.wait_state])], agent1, ..., agentn]
            #obs = np.array(ob)
            #wave_st = obs[:][:6]
            #print('================ob:',ob)
            # note this done is pre-decision to reset LSTM states!
            done = True
            if self.agent in ['codql', 'dqn', 'ddpg']:
                pass
            else:
                self.model.reset()
            self.cur_step = 0
            rewards = []
            if self.agent in ['codql', 'dqn']:
                print('Start training %s ...' % self.agent)
                eps = 0.05
                a_dim = self.env.n_a_ls[0]
                former_act_prob = np.zeros((1, a_dim))  # ,np.zeros((1, 2))]  #a_dim=2
                ##############mean_state
                #mean_state1 = [sum(state) / len(state) for state in zip(*ob)]
                #print('mean_state1:', mean_state1) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                mean_state = np.mean(ob, axis=0, keepdims=True)[0]
                #print('mean_state:',mean_state) #[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                while True: #episode
                    former_act_prob = np.tile(former_act_prob, (len(ob), 1))
                    #print('former_act_prob;',former_act_prob)
                    ##############mean_state
                    mean_state = np.tile(mean_state, (len(ob), 1))
                    #print('mean_state:',mean_state)
                    #print(mean_state[:,:6])
                    # print(former_act_prob)
                    # action_n = trainers.act(state=obs_n, prob=former_act_prob, eps=eps)
                    # print(action_n)  #[0 1 1 0 0 1 1 1 1 0 0 1 0 1 0 0]
                    ##############mean_state
                    action = self.model.act(state=np.array(ob), mean_s=mean_state, prob=former_act_prob, eps=eps)
                    #print('action:',action)
                    #new_obs_n, rew_n, done_n, info_n = env.step(action)
                    next_ob, reward, done, global_reward = self.env.step(action)
                    #print('reward:',reward)
                    if (self.model.reward_norm):  # 3000.0
                        reward /= self.model.reward_norm
                    #if self.model.reward_clip:  # 2
                    #    reward = np.clip(reward, -self.model.reward_clip, self.model.reward_clip)
                    #print('reward:', reward)
                    rewards.append(global_reward) #global_reward is the sum reward of all agent at a step
                    self.global_counter.next()
                    self.cur_step += 1 #episode step
                    # print(rew_n)
                    if not hasattr(done, '__iter__'):
                        done_n = [done] * len(reward)
                    buffer = {
                        'state': ob, 'acts': action, 'rewards': reward, 'dones': done_n}
                    buffer['prob'] = former_act_prob
                    ##############mean_state
                    buffer['mean_sta'] = mean_state
                    #mean_state = [sum(state) / len(state) for state in zip(*next_ob)]
                    mean_state = np.mean(next_ob, axis=0, keepdims=True)[0]
                    former_act_prob = np.mean(list(map(lambda x: np.eye(self.env.n_a_ls[0])[x], action)), axis=0, keepdims=True)
                    #print('former_act_prob:',former_act_prob)
                    self.model.flush_buffer(**buffer)
                    ob = next_ob
                    global_step = self.global_counter.cur_step
                    # episode # termination
                    if done:
                        self.env.terminate()
                        break
                self.model.train()

            elif self.agent == 'ddpg':
                while True:  # episode
                    action_n = self.model.action(ob)
                    actions = [np.argmax(action) if hasattr(action, '__iter__') else action for action in action_n]
                    next_ob, reward, done, global_reward = self.env.step(actions)
                    if (True):  # 3000.0
                        reward_norm = 3000
                        reward /= reward_norm
                    rewards.append(global_reward)  # global_reward is the sum reward of all agent at a step
                    self.global_counter.next()
                    self.cur_step += 1
                    if not hasattr(done, '__iter__'):
                        done_n = [done] * len(reward)
                    self.model.experience(ob, action_n, reward, next_ob, done_n)
                    ob = next_ob
                    global_step = self.global_counter.cur_step
                    # episode # termination
                    if done:
                        self.env.terminate()
                        break
                for _ in range(60):
                    self.model.preupdate()
                    self.model.update()
            else:
                while True:
                    ob, done, R, cur_rewards = self.explore(ob, done)
                    rewards += cur_rewards
                    global_step = self.global_counter.cur_step
                    if self.agent.endswith('a2c'):
                        self.model.backward(R, self.summary_writer, global_step)
                    else:
                        self.model.backward(self.summary_writer, global_step)
                    # termination
                    if done:
                        self.env.terminate()
                        break

            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            self.summary_writer.flush()
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path #dirs['eva_data']
        self.demo = demo

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, demo=self.demo)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
