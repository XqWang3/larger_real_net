"""
Particular class of large traffic grid
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from large_grid_larger.data.build_file import gen_rou_file

sns.set_color_codes()


STATE_NAMES = ['wave', 'wait']
PHASE_NUM = 5
# map from ild order (alphabeta) to signal order (clockwise from north)
# STATE_PHASE_MAP = {'nt1': [2, 3, 1, 0], 'nt2': [2, 3, 1, 0],
#                    'nt3': [2, 3, 1, 0], 'nt4': [2, 3, 1, 0],
#                    'nt5': [2, 1, 0, 3], 'nt6': [3, 2, 0, 1],
#                    'nt7': [0, 2, 3, 1], 'nt8': [0, 2, 3, 1],
#                    'nt9': [1, 0, 2, 3], 'nt10': [1, 0, 2, 3],
#                    'nt11': [3, 1, 0, 2], 'nt12': [3, 1, 0, 2],
#                    'nt13': [3, 1, 0, 2], 'nt14': [3, 1, 0, 2],
#                    'nt15': [1, 2, 3, 0], 'nt16': [3, 2, 1, 0],
#                    'nt17': [2, 3, 1, 0], 'nt18': [2, 3, 1, 0],
#                    'nt19': [2, 3, 1, 0], 'nt20': [1, 2, 3, 0],
#                    'nt21': [0, 3, 2, 1], 'nt22': [0, 2, 3, 1],
#                    'nt23': [0, 2, 3, 1], 'nt24': [0, 2, 3, 1],
#                    'nt25': [1, 0, 2, 3]}
# MAX_CAR_NUM = 30

class LargeGridPhase(PhaseMap):
    def __init__(self):
        phases = ['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr',
                  'rrrGGGrrrrrr', 'rrrrrrrrrGGG']
        self.phases = {PHASE_NUM: PhaseSet(phases)}

class LargeGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))

class LargeGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1') #1100
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return PHASE_NUM

    def _init_large_neighbor_map(self):
        neighbor_map = {}
        # corner nodes
        neighbor_map['nt1'] = ['nt8', 'nt2']
        neighbor_map['nt7'] = ['nt14', 'nt6']
        neighbor_map['nt43'] = ['nt44', 'nt36']
        neighbor_map['nt49'] = ['nt42', 'nt48']
        # edge nodes
        neighbor_map['nt2'] = ['nt9', 'nt3', 'nt1']
        neighbor_map['nt3'] = ['nt10', 'nt4', 'nt2']
        neighbor_map['nt4'] = ['nt11', 'nt5', 'nt3']
        neighbor_map['nt5'] = ['nt12', 'nt6', 'nt4']
        neighbor_map['nt6'] = ['nt13', 'nt7', 'nt5']
        neighbor_map['nt44'] = ['nt45', 'nt37', 'nt43']
        neighbor_map['nt45'] = ['nt46', 'nt38', 'nt44']
        neighbor_map['nt46'] = ['nt47', 'nt39', 'nt45']
        neighbor_map['nt47'] = ['nt48', 'nt40', 'nt46']
        neighbor_map['nt48'] = ['nt49', 'nt41', 'nt47']
        neighbor_map['nt14'] = ['nt21', 'nt7', 'nt13']
        neighbor_map['nt21'] = ['nt28', 'nt14', 'nt20']
        neighbor_map['nt28'] = ['nt35', 'nt21', 'nt27']
        neighbor_map['nt35'] = ['nt42', 'nt28', 'nt34']
        neighbor_map['nt42'] = ['nt49', 'nt35', 'nt41']
        neighbor_map['nt8'] = ['nt15', 'nt9', 'nt1']
        neighbor_map['nt15'] = ['nt22', 'nt16', 'nt8']
        neighbor_map['nt22'] = ['nt29', 'nt23', 'nt15']
        neighbor_map['nt29'] = ['nt36', 'nt30', 'nt22']
        neighbor_map['nt36'] = ['nt43', 'nt37', 'nt29']
        # internal nodes
        #for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
        for i in [9, 10, 11, 12, 13,
                  16, 17, 18, 19, 20,
                  23, 24, 25, 26, 27,
                  30, 31, 32, 33, 34,
                  37, 38, 39, 40, 41]:
            n_node = 'nt' + str(i + 7)
            s_node = 'nt' + str(i - 7)
            w_node = 'nt' + str(i - 1)
            e_node = 'nt' + str(i + 1)
            cur_node = 'nt' + str(i)
            neighbor_map[cur_node] = [n_node, e_node, s_node, w_node]
        return neighbor_map

    '''def _init_large_distance_map(self):
        distance_map = {}
        # corner nodes
        distance_map['nt1'] = {'nt3':2, 'nt7':2, 'nt11':2,
                               'nt4':3, 'nt8':3, 'nt12':3, 'nt16':3,
                               'nt5':4, 'nt9':4, 'nt13':4, 'nt17':4, 'nt21':4,
                               'nt10':5, 'nt14':5, 'nt18':5, 'nt22':5}
        distance_map['nt5'] = {'nt3':2, 'nt9':2, 'nt15':2,
                               'nt2':3, 'nt8':3, 'nt14':3, 'nt20':3,
                               'nt1':4, 'nt7':4, 'nt13':4, 'nt19':4, 'nt25':4,
                               'nt6':5, 'nt12':5, 'nt18':5, 'nt24':5}
        distance_map['nt21'] = {'nt11':2, 'nt17':2, 'nt23':2,
                                'nt6':3, 'nt12':3, 'nt18':3, 'nt24':3,
                                'nt1':4, 'nt7':4, 'nt13':4, 'nt19':4, 'nt25':4,
                                'nt2':5, 'nt8':5, 'nt14':5, 'nt20':5}
        distance_map['nt25'] = {'nt15':2, 'nt19':2, 'nt23':2,
                                'nt10':3, 'nt14':3, 'nt18':3, 'nt22':3,
                                'nt5':4, 'nt9':4, 'nt13':4, 'nt17':4, 'nt21':4,
                                'nt4':5, 'nt8':5, 'nt12':5, 'nt16':5}
        # edge nodes
        distance_map['nt2'] = {'nt4':2, 'nt6':2, 'nt8':2, 'nt12':2,
                               'nt5':3, 'nt9':3, 'nt11':3, 'nt13':3, 'nt17':3,
                               'nt10':4, 'nt14':4, 'nt16':4, 'nt18':4, 'nt22':4,
                               'nt15':5, 'nt19':5, 'nt21':5, 'nt23':5}
        distance_map['nt3'] = {'nt1':2, 'nt5':2, 'nt7':2, 'nt9':2, 'nt13':2,
                               'nt6':3, 'nt10':3, 'nt12':3, 'nt14':3, 'nt18':3,
                               'nt11':4, 'nt15':4, 'nt17':4, 'nt19':4, 'nt23':4,
                               'nt16':5, 'nt20':5, 'nt22':5, 'nt24':5}
        distance_map['nt4'] = {'nt2':2, 'nt8':2, 'nt10':2, 'nt14':2,
                               'nt1':3, 'nt7':3, 'nt13':3, 'nt15':3, 'nt19':3,
                               'nt6':4, 'nt12':4, 'nt18':4, 'nt20':4, 'nt24':4,
                               'nt11':5, 'nt17':5, 'nt23':5, 'nt25':5}
        distance_map['nt22'] = {'nt12':2, 'nt16':2, 'nt18':2, 'nt24':2,
                                'nt7':3, 'nt11':3, 'nt13':3, 'nt19':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt8':4, 'nt14':4, 'nt20':4,
                                'nt1':5, 'nt3':5, 'nt9':5, 'nt15':5}
        distance_map['nt23'] = {'nt13':2, 'nt17':2, 'nt19':2, 'nt21':2, 'nt25':2,
                                'nt8':3, 'nt12':3, 'nt14':3, 'nt16':3, 'nt20':3,
                                'nt3':4, 'nt7':4, 'nt9':4, 'nt11':4, 'nt15':4,
                                'nt2':5, 'nt4':5, 'nt6':5, 'nt10':5}
        distance_map['nt24'] = {'nt14':2, 'nt18':2, 'nt20':2, 'nt22':2,
                                'nt9':3, 'nt13':3, 'nt15':3, 'nt17':3, 'nt21':3,
                                'nt4':4, 'nt8':4, 'nt10':4, 'nt12':4, 'nt16':4,
                                'nt3':5, 'nt5':5, 'nt7':5, 'nt11':5}
        distance_map['nt10'] = {'nt4':2, 'nt8':2, 'nt14':2, 'nt20':2,
                                'nt3':3, 'nt7':3, 'nt13':3, 'nt19':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt12':4, 'nt18':4, 'nt24':4,
                                'nt1':5, 'nt11':5, 'nt17':5, 'nt23':5}
        distance_map['nt15'] = {'nt5':2, 'nt9':2, 'nt13':2, 'nt19':2, 'nt25':2,
                                'nt4':3, 'nt8':3, 'nt12':3, 'nt18':3, 'nt24':3,
                                'nt3':4, 'nt7':4, 'nt11':4, 'nt13':4, 'nt23':4,
                                'nt2':5, 'nt6':5, 'nt16':5, 'nt22':5}
        distance_map['nt20'] = {'nt10':2, 'nt14':2, 'nt18':2, 'nt24':2,
                                'nt5':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt23':3,
                                'nt4':4, 'nt8':4, 'nt12':4, 'nt16':4, 'nt22':4,
                                'nt3':5, 'nt7':5, 'nt11':5, 'nt21':5}
        distance_map['nt6'] = {'nt2':2, 'nt8':2, 'nt12':2, 'nt16':2,
                               'nt3':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt21':3,
                               'nt4':4, 'nt10':4, 'nt14':4, 'nt18':4, 'nt22':4,
                               'nt5':5, 'nt15':5, 'nt19':5, 'nt23':5}
        distance_map['nt11'] = {'nt1':2, 'nt7':2, 'nt13':2, 'nt17':2, 'nt21':2,
                                'nt2':3, 'nt8':3, 'nt14':3, 'nt18':3, 'nt22':3,
                                'nt3':4, 'nt9':4, 'nt15':4, 'nt19':4, 'nt23':4,
                                'nt4':5, 'nt10':5, 'nt20':5, 'nt24':5}
        distance_map['nt16'] = {'nt2':2, 'nt8':2, 'nt12':2, 'nt16':2,
                                'nt3':3, 'nt9':3, 'nt13':3, 'nt17':3, 'nt21':3,
                                'nt4':4, 'nt10':4, 'nt14':4, 'nt18':4, 'nt22':4,
                                'nt5':5, 'nt15':5, 'nt19':5, 'nt23':5}
        # internal nodes
        distance_map['nt7'] = {'nt1':2, 'nt3':2, 'nt9':2, 'nt11':2, 'nt13':2, 'nt17':2,
                               'nt4':3, 'nt10':3, 'nt14':3, 'nt16':3, 'nt18':3, 'nt22':3,
                               'nt5':4, 'nt15':4, 'nt19':4, 'nt21':4, 'nt23':4,
                               'nt20':5, 'nt24':5}
        distance_map['nt8'] = {'nt2':2, 'nt4':2, 'nt6':2, 'nt10':2, 'nt12':2, 'nt14':2, 'nt18':2,
                               'nt1':3, 'nt5':3, 'nt11':3, 'nt15':3, 'nt17':3, 'nt19':3, 'nt23':3,
                               'nt16':4, 'nt20':4, 'nt22':4, 'nt24':4,
                               'nt21':5, 'nt25':5}
        distance_map['nt9'] = {'nt3':2, 'nt5':2, 'nt7':2, 'nt13':2, 'nt15':2, 'nt19':2,
                               'nt2':3, 'nt6':3, 'nt12':3, 'nt18':3, 'nt20':3, 'nt24':3,
                               'nt1':4, 'nt11':4, 'nt17':4, 'nt23':4, 'nt25':4,
                               'nt16':5, 'nt22':5}
        distance_map['nt12'] = {'nt2':2, 'nt6':2, 'nt8':2, 'nt14':2, 'nt16':2, 'nt18':2, 'nt22':2,
                                'nt1':3, 'nt3':3, 'nt9':3, 'nt15':3, 'nt19':3, 'nt21':3, 'nt23':3,
                                'nt4':4, 'nt10':4, 'nt20':4, 'nt24':4,
                                'nt5':5, 'nt25':5}
        distance_map['nt13'] = {'nt3':2, 'nt7':2, 'nt9':2, 'nt11':2, 'nt15':2, 'nt17':2, 'nt19':2, 'nt23':2,
                                'nt2':3, 'nt4':3, 'nt6':3, 'nt10':3, 'nt16':3, 'nt20':3, 'nt22':3, 'nt24':3,
                                'nt1':4, 'nt5':4, 'nt21':4, 'nt25':4}
        distance_map['nt14'] = {'nt4':2, 'nt8':2, 'nt10':2, 'nt12':2, 'nt18':2, 'nt20':2, 'nt24':2,
                                'nt3':3, 'nt5':3, 'nt7':3, 'nt11':3, 'nt17':3, 'nt23':3, 'nt25':3,
                                'nt2':4, 'nt6':4, 'nt16':4, 'nt22':4,
                                'nt1':5, 'nt21':5}
        distance_map['nt17'] = {'nt7':2, 'nt11':2, 'nt13':2, 'nt19':2, 'nt21':2, 'nt23':2,
                                'nt2':3, 'nt6':3, 'nt8':3, 'nt14':3, 'nt20':3, 'nt24':3,
                                'nt1':4, 'nt3':4, 'nt9':4, 'nt15':4, 'nt25':4,
                                'nt4':5, 'nt10':5}
        distance_map['nt18'] = {'nt8':2, 'nt12':2, 'nt14':2, 'nt16':2, 'nt20':2, 'nt22':2, 'nt24':2,
                                'nt3':3, 'nt7':3, 'nt9':3, 'nt11':3, 'nt15':3, 'nt21':3, 'nt25':3,
                                'nt2':4, 'nt4':4, 'nt6':4, 'nt10':4,
                                'nt1':5, 'nt5':5}
        distance_map['nt19'] = {'nt9':2, 'nt13':2, 'nt15':2, 'nt17':2, 'nt23':2, 'nt25':2,
                                'nt4':3, 'nt8':3, 'nt10':3, 'nt12':3, 'nt16':3, 'nt22':3,
                                'nt3':4, 'nt5':4, 'nt7':4, 'nt11':4, 'nt21':4,
                                'nt2':5, 'nt6':5}
        return distance_map'''

    def _init_map(self):
        self.neighbor_map = self._init_large_neighbor_map()
        # for spatial discount
        #self.distance_map = self._init_large_distance_map()
        #self.max_distance = 6
        self.phase_map = LargeGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        return gen_rou_file(self.data_path,
                            self.peak_flow1,
                            self.peak_flow2,
                            self.init_density,
                            seed=seed,
                            thread=self.sim_thread)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_large.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = LargeGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(2)
    ob = env.reset()
    controller = LargeGridController(env.node_names)
    rewards = []
    while True:
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
