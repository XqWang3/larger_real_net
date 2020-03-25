import sys, os;

sys.path.append(os.getcwd())
import os.path as osp

this_abs_path = osp.abspath(__file__)
experiment_dir = osp.dirname(this_abs_path)
maddpg_dir = osp.dirname(experiment_dir)
root_dir = osp.dirname(maddpg_dir)
seed_dir = osp.join(root_dir, "maddpg/traffic_visualization/env_reset_seed")

sys.path.extend([
    experiment_dir,
    maddpg_dir,
    root_dir,
])

import numpy as np
import random
import experiments.serialization_utils as sut
import gym
from gym import spaces
import pyglet

class TrafficEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self, n=4, m=5, n_vih=100, add_newCar=5, max_keep_time=4, min_length=2, max_length=20,
                 flag_traffic_flow=1):
        '''
         m=6  车在某段路上行驶的最长时间是 m
         n_vih=64   初始化路网车辆数
         add_newCar=2  每个step 新生成车量
         min_time=3 至少每隔3个step，s_light才可改变一次
         max_time=5 至多每隔5个step，s_light才可改变一次
         max_keep_time=5 每隔5+1个step，s_light才可改变一次
         min_length=2
         max_length=20  车的路线长度最大值
         n=4  则路网为3*3田字格 or 4*4 or ····  up to you
          road_net=[0——1——2
                    |  |  |
                    3——4——5
                    |  |  |
                    6——7——8]
              ·········· 3*3
          road_net=[0——1——2——3
                    |  |  |  |
                    4——5——6——7
                    |  |  |  |
                    8——9—10—11
                    |  |  |  |
                   12—13-14-15]
              ·········4*4
        '''
        #self.count = 0  # step的次数
        #self.total_re = 0
        #self.step_total_re = 0
        self.keep_time = 0
        self.n_vih_now = 0
        #self.delete_flag = [0 for _ in range(self.n_vih_now)]
        self.n = n
        self.done_n = [False for _ in range(16)]
        self.nodes = [i for i in range(self.n * self.n)]
        self.transition = [0 for _ in range(self.n * self.n)]  # gobal stotistic transition
        self.small_transition = [0 for _ in range(self.n * self.n)]  # four small ring transition
        self.outer_transition = [0 for _ in range(self.n * self.n)]  # outer ring state transition
        self.inner_transition = [0 for _ in range(self.n * self.n)]  # inner ring state transition
        self.span_transition = [0 for _ in range(self.n * self.n)]  # span ring state transition
        self.outer = []  # 外环中的状态
        self.inner = []  # 内环中的状态   eg.[5,6,9,10]
        for i in range(self.n * self.n):
            if i == 0:
                self.transition[i] = [i + 1, i + self.n]
                self.small_transition[i] = [i + 1, i + self.n]
                self.outer_transition[i] = [i + 1, i + self.n]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = []
            elif i == self.n - 1:
                self.transition[i] = [i - 1, i + self.n]
                self.small_transition[i] = [i - 1, i + self.n]
                self.outer_transition[i] = [i - 1, i + self.n]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = []
            elif 0 < i < self.n - 1:
                self.transition[i] = [i - 1, i + 1, i + self.n]
                if i == 1:
                    self.small_transition[i] = [i - 1, i + self.n]
                else:
                    self.small_transition[i] = [i + 1, i + self.n]
                self.outer_transition[i] = [i - 1, i + 1]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = [i + self.n]
            elif i == self.n * (self.n - 1):
                self.transition[i] = [i - self.n, i + 1]
                self.small_transition[i] = [i - self.n, i + 1]
                self.outer_transition[i] = [i - self.n, i + 1]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = []
            elif i == self.n * self.n - 1:
                self.transition[i] = [i - 1, i - self.n]
                self.small_transition[i] = [i - 1, i - self.n]
                self.outer_transition[i] = [i - 1, i - self.n]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = []
            elif self.n * (self.n - 1) < i < self.n * self.n - 1:
                self.transition[i] = [i - 1, i + 1, i - self.n]
                if i == 13:
                    self.small_transition[i] = [i - 1, i - self.n]
                else:
                    self.small_transition[i] = [i + 1, i - self.n]
                self.outer_transition[i] = [i - 1, i + 1]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = [i - self.n]
            elif i % self.n == 0 and i != 0 and i != self.n * (self.n - 1):
                self.transition[i] = [i - self.n, i + 1, i + self.n]
                if i == 4:
                    self.small_transition[i] = [i - self.n, i + 1]
                else:
                    self.small_transition[i] = [i + 1, i + self.n]
                self.outer_transition[i] = [i - self.n, i + self.n]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = [i + 1]
            elif (i + 1 - self.n) % self.n == 0 and (i + 1 - self.n) != 0 and (i + 1 - self.n) != self.n * (self.n - 1):
                self.transition[i] = [i - self.n, i - 1, i + self.n]
                if i == 7:
                    self.small_transition[i] = [i - self.n, i - 1]
                else:
                    self.small_transition[i] = [i - 1, i + self.n]
                self.outer_transition[i] = [i - self.n, i + self.n]
                self.outer.append(i)
                self.inner_transition[i] = []
                self.span_transition[i] = [i - 1]
            else:
                self.transition[i] = [i - 1, i - self.n, i + 1, i + self.n]
                self.outer_transition[i] = []
                self.inner.append(i)
                if i == self.n + 1:  # 5
                    self.small_transition[i] = [i - 1, i - self.n]
                    self.inner_transition[i] = [i + 1, i + self.n]
                    self.span_transition[i] = [i - 1, i - self.n]
                elif i == self.n + 2:  # 6
                    self.small_transition[i] = [i - self.n, i + 1]
                    self.inner_transition[i] = [i - 1, i + self.n]
                    self.span_transition[i] = [i + 1, i - self.n]
                elif i == self.n + 5:  # 9
                    self.small_transition[i] = [i - 1, i + self.n]
                    self.inner_transition[i] = [i + 1, i - self.n]
                    self.span_transition[i] = [i - 1, i + self.n]
                else:  # i==10
                    self.small_transition[i] = [i + 1, i + self.n]
                    self.inner_transition[i] = [i - 1, i - self.n]
                    self.span_transition[i] = [i + 1, i + self.n]

        self.m = m
        self.n_vih = n_vih
        # self.min_time=min_time
        # self.max_time=max_time
        self.min_length = min_length
        self.max_length = max_length
        self.add_newCar = add_newCar
        self.max_keep_time = max_keep_time # 4 times
        self.flag_traffic_flow = flag_traffic_flow  # (default)1 or 2 or 3
        self.seed_name = 'traffic_flow{}'.format(self.flag_traffic_flow)

        self.label = []
        #self.nod_rewards = []
        self.decode_lines = []
        self.lines = []
        for i in range(4):
            self.label.append([])
            #self.nod_rewards.append([])
            self.decode_lines.append([])
            self.lines.append([])
            for j in range(4):
                self.label[i].append(0)
                #self.nod_rewards[i].append(0)
                self.decode_lines[i].append([])
                self.lines[i].append([])
                for k in range(4):
                    self.decode_lines[i][j].append([])
                    self.lines[i][j].append([])
                    for l in range(4):
                        self.decode_lines[i][j][k].append([1, 1, 1])
                        self.lines[i][j][k].append(None)
        self.action_space = []
        self.observation_space = []
        for i in range(self.n * self.n):
            self.action_space.append(spaces.Discrete(2))
            self.low = np.array([0,0,0,0, 0,0,0,0, 0,0,0,0])
            self.high = np.array([np.inf,np.inf,np.inf,np.inf, 1,1,1,1, 1,1,1,1])
            self.observation_space.append(spaces.Box(self.low, self.high))
        self.viewer = None

    '''def inf_line_clear(self):
        for i in range(4):
            for j in range(4):
                self.nod_rewards[i][j] = 0
                for k in range(4):
                    for l in range(4):
                        self.decode_lines[i][j][k][l][1] = 0
                        self.decode_lines[i][j][k][l][2] = 0
        return self.decode_lines'''

    def reset(self):
        seed_load_dir = osp.join(seed_dir, self.seed_name)
        self.s_car = []
        self.s_light = []
        self.states = []
        obs_n = []
        for i in range(16):
            self.states.append([])
            #self.s_light.append([])
            for j in range(16):
                self.states[i].append(0)
                #self.s_light[i].append(0)
        car_state_filename = 'step_{}.npz'.format(random.randint(2990, 3000))
        video_serialization = sut.npz_to_serialization(osp.join(seed_load_dir, car_state_filename))
        idx = 0
        frame = sut.index_serialization(video_serialization, idx)
        self.deserialize_seed(frame)
        self.s_car = self.s_car.tolist()
        #print(self.s_light)

        self.n_vih_now = len(self.s_car)  # 当前路网中的车辆数
        #self.delete_flag = [0 for _ in range(self.n_vih_now)]
        for j in range(self.n_vih_now):
            t = self.s_car[j][0]
            if t == 0:
                self.states[self.s_car[j][1]][self.s_car[j][2]] += 1
        for i in range(16):
            obs_n.append(self._get_obs(i))

        '''for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        self.decode_lines[i][j][k][l] = [0, 0, 0]'''

        return obs_n

    # get observation for a particular agent
    def _get_obs(self,i):
        obs = np.zeros(4 + 8)
        if i == 0:
            obs[1] = self.states[i + 4][i]
            self.states[i + 4][i] = 0
            obs[3] = self.states[i + 1][i]
            self.states[i + 1][i] = 0
            obs[4] = 1  # 4,5,6,7
            obs[i + 8] = 1  # 8,9,10,11
        elif i in [1, 2]:
            obs[1] = self.states[i + 4][i]
            self.states[i + 4][i] = 0
            obs[2] = self.states[i - 1][i]
            self.states[i - 1][i] = 0
            obs[3] = self.states[i + 1][i]
            self.states[i + 1][i] = 0
            obs[4] = 1  # 4,5,6,7
            obs[i + 8] = 1  # 8,9,10,11
        elif i == 3:
            obs[1] = self.states[i + 4][i]
            self.states[i + 4][i] = 0
            obs[2] = self.states[i - 1][i]
            self.states[i - 1][i] = 0
            obs[4] = 1  # 4,5,6,7
            obs[i + 8] = 1  # 8,9,10,11
        elif i in [4, 8]:
            obs[0] = self.states[i - 4][i]
            self.states[i - 4][i] = 0
            obs[1] = self.states[i + 4][i]
            self.states[i + 4][i] = 0
            obs[3] = self.states[i + 1][i]
            self.states[i + 1][i] = 0
            obs[i // 4 + 4] = 1  # 4,5,6,7
            obs[8] = 1  # 8,9,10,11
        elif i in [7, 11]:
            obs[0] = self.states[i - 4][i]
            self.states[i - 4][i] = 0
            obs[1] = self.states[i + 4][i]
            self.states[i + 4][i] = 0
            obs[2] = self.states[i - 1][i]
            self.states[i - 1][i] = 0
            obs[(i - 3) // 4 + 4] = 1  # 4,5,6,7
            obs[11] = 1  # 8,9,10,11
        elif i in [5, 6, 9, 10]:
            obs[0] = self.states[i - 4][i]
            self.states[i - 4][i] = 0
            obs[1] = self.states[i + 4][i]
            self.states[i + 4][i] = 0
            obs[2] = self.states[i - 1][i]
            self.states[i - 1][i] = 0
            obs[3] = self.states[i + 1][i]
            self.states[i + 1][i] = 0
            if i in [5, 6]:
                obs[5] = 1  # 4,5,6,7
                obs[i + 4] = 1  # 4,5,6,7
            else:
                obs[6] = 1  # 4,5,6,7
                obs[i] = 1  # 8,9,10,11
        elif i == 12:
            obs[0] = self.states[i - 4][i]
            self.states[i - 4][i] = 0
            obs[3] = self.states[i + 1][i]
            self.states[i + 1][i] = 0
            obs[7] = 1  # 4,5,6,7
            obs[8] = 1  # 8,9,10,11
        elif i in [13, 14]:
            obs[0] = self.states[i - 4][i]
            self.states[i - 4][i] = 0
            obs[2] = self.states[i - 1][i]
            self.states[i - 1][i] = 0
            obs[3] = self.states[i + 1][i]
            self.states[i + 1][i] = 0
            obs[7] = 1  # 4,5,6,7
            obs[i-4] = 1  # 8,9,10,11
        elif i == 15:
            obs[0] = self.states[i - 4][i]
            self.states[i - 4][i] = 0
            obs[2] = self.states[i - 1][i]
            self.states[i - 1][i] = 0
            obs[7] = 1  # 4,5,6,7
            obs[11] = 1  # 8,9,10,11
        '''for j in range(4):
            obs[j] = obs[j]/20'''
        return obs

    '''def getNewLine(self, x1, y1, x2, y2):
        from multiagent import rendering
        x11 = 100 * x1
        y11 = 100 * y1
        x22 = 100 * x2
        y22 = 100 * y2
        m = 20  # 表示每边截掉的长度
        if x11 == x22:
            if y11 > y22:
                long_line = rendering.Line((x11 - 5, y11 - m), (x22 - 5, y22 + m))  # move left,  r=6
                circle = rendering.make_circle(radius=5, res=30)
                circle_transform = rendering.Transform(translation=(x11-5+50,y11-m+50))
                #circle_transform = rendering.Transform(translation=(x11-5,y11-m)
            elif y11 < y22:
                long_line = rendering.Line((x11 + 5, y11 + m), (x22 + 5, y22 - m))  # move right
                circle = rendering.make_circle(radius=5, res=30)
                circle_transform = rendering.Transform(translation=(x11+5+50,y11+m+50))
                #short_line = rendering.Line((x11 + 5 - n_vih1, y11 + m), (x11 + 5 + n_vih1, y11 + m))
            else:
                print('1,It is a point rather than a line')
        elif y11 == y22:
            if x11 > x22:
                long_line = rendering.Line((x11-m, y11+5), (x22+m, y22+5))  # move up
                circle = rendering.make_circle(radius=5, res=30)
                circle_transform = rendering.Transform(translation=(x11-m+50,y11+5+50))
                #short_line = rendering.Line((x11 - m, y11 + 5 - n_vih1), (x11 - m, y11 + 5 + n_vih1))
            elif x11 < x22:
                long_line = rendering.Line((x11+m, y11-5), (x22-m, y22-5))  # move down
                circle = rendering.make_circle(radius=5, res=30)
                circle_transform = rendering.Transform(translation=(x11+m+50, y11-5+50))
                #short_line = rendering.Line((x11 + m, y11 - 5 - n_vih1), (x11 + m, y11 - 5 + n_vih1))
            else:
                print('2,It is a point rather than a line')
        else:
            print('3,It is not a line')

        self._transform = rendering.Transform(translation=(50, 50))
        long_line.add_attr(self._transform)
        long_line.set_linewidth(4)
        self.viewer.add_geom(long_line)

       # short_line.set_color(0, 0, 0)
        circle.add_attr(circle_transform)
        self.viewer.add_geom(circle)

        newline = []
        newline.append(circle)
        newline.append(long_line)

        return newline'''

    '''def render(self):
        # 每条路长road_length=80，同条路上的两车道相距bi_dis=5，两条路之间相距distance=100
        from multiagent import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 600)  # 600x600 是画板的长和框
            for i in range(4):
                for j in range(4):
                    rectangle = rendering.make_polyline([(-15,15),(15,15),(15,-15),(-15,-15),(-15,15)])
                    rect_transform = rendering.Transform(translation=(100 * i + 50, 100 * j + 50))
                    rectangle.add_attr(rect_transform)
                    self.viewer.add_geom(rectangle)

                    score_label = pyglet.text.Label('0', font_size=8,
                                                    x=100 * i + 38, y=100 * j + 50, anchor_x='left', anchor_y='center',
                                                    color=(25, 25, 25, 255))
                    self.label[i][j] = score_label
            self.step_count_label = pyglet.text.Label('step_count:000',
                                                      font_name='Times New Roman', font_size=12,
                                                      x=39, y=100 * 5 - 20, anchor_x='left', anchor_y='center',
                                                      color=(25, 25, 25, 255))
            self.step_total_re_label = pyglet.text.Label('step_total_reward:000',
                                                         font_name='Times New Roman', font_size=12,
                                                         x=39, y=100 * 5 + 5, anchor_x='left', anchor_y='center',
                                                         color=(25, 25, 25, 255))
            self.total_re_label = pyglet.text.Label('total_reward:000',
                                                    font_name='Times New Roman', font_size=12,
                                                    x=39, y=100 * 5 + 30, anchor_x='left', anchor_y='center',
                                                    color=(25, 25, 25, 255))
            for x1 in range(4):
                for y1 in range(3):
                    x2 = x1
                    y2 = y1 + 1
                    self.lines[x1][y1][x2][y2] = self.getNewLine(x1, y1, x2, y2)
                    self.lines[x2][y2][x1][y1] = self.getNewLine(x2, y2, x1, y1)
            for x1 in range(3):
                for y1 in range(4):
                    x2 = x1 + 1
                    y2 = y1
                    self.lines[x1][y1][x2][y2] = self.getNewLine(x1, y1, x2, y2)
                    self.lines[x2][y2][x1][y1] = self.getNewLine(x2, y2, x1, y1)

        for x1 in range(4):
            for y1 in range(3):
                x2 = x1
                y2 = y1 + 1
                self.updateLine(x1, y1, x2, y2)
                self.updateLine(x2, y2, x1, y1)
        for x1 in range(3):
            for y1 in range(4):
                x2 = x1 + 1
                y2 = y1
                self.updateLine(x1, y1, x2, y2)
                self.updateLine(x2, y2, x1, y1)
        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()
        for geom in self.viewer.geoms:
            geom.render()
        for i in range(4):
            for j in range(4):
                self.label[i][j].text = "{}".format(self.nod_rewards[i][j])
                self.label[i][j].draw()
        self.step_count_label.text = "step_count:{}".format(self.count)
        self.step_count_label.draw()
        self.step_total_re_label.text = "step_total_reward:{}".format(self.step_total_re)
        self.step_total_re_label.draw()
        self.total_re_label.text = "total_reward:{}".format(self.total_re)
        self.total_re_label.draw()
        win.flip()
        return arr'''

    '''def updateLine(self, x1, y1, x2, y2):
        red_g = self.decode_lines[x1][y1][x2][y2][0]   #1 is green
        n_vih2 = self.decode_lines[x1][y1][x2][y2][2]  # 其值来自于step

        long_line = self.lines[x1][y1][x2][y2][1]
        if self.flag_traffic_flow in [1,3]:
            if n_vih2 <= 15:
                r = n_vih2/15
                g = 1
                b = 1-r
                long_line.set_color(r, g, b)
            else:
                r = 1
                g = 1-b
                b = (n_vih2-15)/15
                long_line.set_color(r, g, b)
        else:
            if n_vih2 <= 10:
                r = n_vih2/10
                g = 1
                b = 1-r
                long_line.set_color(r, g, b)
            else:
                r = 1
                b = (n_vih2-10)/10
                g = 1 - b
                long_line.set_color(r, g, b)
        long_line.set_color(r, g, b)

        circle = self.lines[x1][y1][x2][y2][0]
        #circle.set_color(0,1,0)
        circle.set_color(1-red_g, red_g, 0)'''

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def deserialization_seed(self, s_car, s_light, keep_time):
        self.s_car = s_car
        self.s_light = s_light
        self.keep_time = keep_time
    def deserialize_seed(self, dicts):
        for key in dicts:
            self.deserialization_seed(**dicts[key])
    def serialize(self):
        results = dict()
        entity = {
            'decode_lines': self.decode_lines,
            #'nod_rewards': self.nod_rewards,
            #'count': self.count,
            #'step_total_re': self.step_total_re,
            #'total_re': self.total_re
        }
        results['entity_{}'.format(0)] = entity
        return results
    def deserialization(self, decode_lines, nod_rewards, count, step_total_re, total_re):
        self.decode_lines = decode_lines
        #self.nod_rewards = nod_rewards
        #self.count = count
        #self.step_total_re = step_total_re
        #self.total_re = total_re
    def deserialize(self, dicts):
        for key in dicts:
            self.deserialization(**dicts[key])

    # def create_one_car(self):
    def create_car_kind_1(self):  # span inner or outer circle
        s_car_i = []
        outer_inner_flag = random.randint(0, 100)  # 随机数决定车走内环或外环或跨环
        if outer_inner_flag < 55:  # 外环45%
            t = random.randint(0, self.m)  # 随机初始车在某段路上还剩时间t到达路口
            s0 = random.choice(self.outer)
            s1 = random.choice(self.outer_transition[s0])  # 第一个需要到达的路口节点
            length = random.randint(self.min_length, self.max_length)  # 随机初始车的路线长度
            s_car_i = [t, s0, s1]
            for _ in range(length):
                s_temp = random.choice(self.outer_transition[s_car_i[-1]])
                s_car_i.append(s_temp)
            return s_car_i

        elif outer_inner_flag > 65:  # 内环45%
            t = random.randint(0, self.m)  # 随机初始车在某段路上还剩时间t到达路口
            s0 = random.choice(self.inner)
            s1 = random.choice(self.inner_transition[s0])  # 第一个需要到达的路口节点
            length = random.randint(self.min_length, self.max_length)  # 随机初始车的路线长度
            s_car_i = [t, s0, s1]
            for _ in range(length):
                s_temp = random.choice(self.inner_transition[s_car_i[-1]])
                s_car_i.append(s_temp)
            return s_car_i

        elif 55 <= outer_inner_flag < 60:  # 先外环-再跨环-再内环5%
            t = random.randint(0, self.m)  # 随机初始车在某段路上还剩时间t到达路口
            s0 = random.choice(self.outer)
            s1 = random.choice(self.outer_transition[s0])  # 第一个需要到达的路口节点
            length1 = random.randint(self.min_length / 2, self.max_length / 2)  # 随机初始车的路线长度
            length2 = random.randint(self.min_length / 2, self.max_length / 2)
            s_car_i = [t, s0, s1]
            for _ in range(length1):
                s_temp = random.choice(self.outer_transition[s_car_i[-1]])
                s_car_i.append(s_temp)
            if s_car_i[-1] in [0, self.n - 1, self.n * (self.n - 1), self.n * self.n - 1]:
                del s_car_i[-1]
            s_length1 = random.choice(self.span_transition[s_car_i[-1]])
            s_car_i.append(s_length1)
            for _ in range(length2):
                s_temp = random.choice(self.inner_transition[s_car_i[-1]])
                s_car_i.append(s_temp)
            return s_car_i

        else:  # 先内环-再跨环-再外环5%
            t = random.randint(0, self.m)  # 随机初始车在某段路上还剩时间t到达路口
            s0 = random.choice(self.inner)
            s1 = random.choice(self.inner_transition[s0])  # 第一个需要到达的路口节点
            length1 = random.randint(self.min_length / 2, self.max_length / 2)  # 随机初始车的路线长度
            length2 = random.randint(self.min_length / 2, self.max_length / 2)
            s_car_i = [t, s0, s1]
            for _ in range(length1):
                s_temp = random.choice(self.inner_transition[s_car_i[-1]])
                s_car_i.append(s_temp)
            s_length1 = random.choice(self.span_transition[s_car_i[-1]])
            s_car_i.append(s_length1)
            for _ in range(length2):
                s_temp = random.choice(self.outer_transition[s_car_i[-1]])
                s_car_i.append(s_temp)
            return s_car_i

    def create_car_kind_2(self):  # uniform spread
        s_car_i = []
        t = random.randint(0, self.m)  # 随机初始车在某段路上还剩时间t到达路口
        s0 = random.randint(0, self.n * self.n - 1)
        s1 = random.choice(self.transition[s0])  # 第一个需要到达的路口节点
        length = random.randint(self.min_length, self.max_length)  # 随机初始车的路线长度
        s_car_i = [t, s0, s1]
        for _ in range(length):
            s_temp = random.choice(self.transition[s_car_i[-1]])
            s_car_i.append(s_temp)
        return s_car_i

    def create_car_kind_3(self):  # non-intersection four circle
        s_car_i = []
        t = random.randint(0, self.m)  # 随机初始车在某段路上还剩时间t到达路口
        s0 = random.randint(0, self.n * self.n - 1)
        s1 = random.choice(self.small_transition[s0])  # 第一个需要到达的路口节点
        length = random.randint(self.min_length, self.max_length)  # 随机初始车的路线长度
        s_car_i = [t, s0, s1]
        for _ in range(length):
            s_temp = random.choice(self.small_transition[s_car_i[-1]])
            s_car_i.append(s_temp)
        return s_car_i

    def step(self, a):
        #self.inf_line_clear()

        rewards = [0 for _ in range(self.n * self.n)]  # 每个交通灯的奖励清零
        #self.step_total_re = 0
        actions = a

        # transform probabilities to discrete actions, if it is the case
        actions = [np.argmax(action) if hasattr(action, '__iter__') else action for action in actions]
        #print(actions)
        if self.keep_time == 0: #or self.keep_time > self.max_keep_time:  # >3,namely the light keep green or red 4s
            #print(self.count)
            for i in range(self.n * self.n):
                if i in [0, 1, 2, 3]:
                    self.s_light[i][i + 4] = actions[i]
                    #self.decode_lines[i // 4][i % 4][(i + 4) // 4][(i + 4) % 4][0] = self.s_light[i][i + 4]  # 红绿灯颜色
                elif i in [12, 13, 14, 15]:
                    self.s_light[i][i - 4] = actions[i]
                    #self.decode_lines[i // 4][i % 4][(i - 4) // 4][(i - 4) % 4][0] = self.s_light[i][i - 4]  # 红绿灯颜色
                else:
                    self.s_light[i][i - 4] = actions[i]
                    #self.decode_lines[i // 4][i % 4][(i - 4) // 4][(i - 4) % 4][0] = self.s_light[i][i - 4]  # 红绿灯颜色
                    self.s_light[i][i + 4] = actions[i]
                    #self.decode_lines[i // 4][i % 4][(i + 4) // 4][(i + 4) % 4][0] = self.s_light[i][i + 4]  # 红绿灯颜色
                if i in [0, 4, 8, 12]:
                    self.s_light[i][i + 1] = 1 - actions[i]
                    #self.decode_lines[i // 4][i % 4][(i + 1) // 4][(i + 1) % 4][0] = self.s_light[i][i + 1]  # 红绿灯颜色
                elif i in [3, 7, 11, 15]:
                    self.s_light[i][i - 1] = 1 - actions[i]
                    #self.decode_lines[i // 4][i % 4][(i - 1) // 4][(i - 1) % 4][0] = self.s_light[i][i - 1]  # 红绿灯颜色
                else:
                    self.s_light[i][i - 1] = 1 - actions[i]
                    #self.decode_lines[i // 4][i % 4][(i - 1) // 4][(i - 1) % 4][0] = self.s_light[i][i - 1]  # 红绿灯颜色
                    self.s_light[i][i + 1] = 1 - actions[i]
                    #self.decode_lines[i // 4][i % 4][(i + 1) // 4][(i + 1) % 4][0] = self.s_light[i][i + 1]  # 红绿灯颜色
            #self.keep_time = 0

        for m in range(self.keep_time,self.max_keep_time):
            #self.count += 1
            delete_count = 0
            delete_flag = []
            for _ in range(self.add_newCar):  # 每个step,新生成车
                if self.flag_traffic_flow == 1:  # span ring
                    s_car_i = self.create_car_kind_1()
                if self.flag_traffic_flow == 2:  # gobal uniform
                    s_car_i = self.create_car_kind_2()
                if self.flag_traffic_flow == 3:  # four small ring
                    s_car_i = self.create_car_kind_3()
                self.s_car.append(s_car_i)
            #self.n_vih_now = len(self.s_car)  # 当前路网中的车辆数
            self.n_vih_now = self.n_vih_now + self.add_newCar
            '''try:
                n_vih_now == 0
            except:
                print("even no one car driving in the road-net,please initial more cars.")'''
            #self.delete_flag = [0 for _ in range(self.n_vih_now)]  # 准备删除的车辆 flag=1
            '''for _ in range(self.add_newCar):
                self.delete_flag.append(0)'''

            for i in range(self.n_vih_now):
                rest_nods = len(self.s_car[i]) - 2  # rest_nods=1时表示该车到达de下一路口即wei终点
                t = self.s_car[i][0]
                if t != 0:
                    t -= 1
                    self.s_car[i][0] = t
                    #self.decode_lines[self.s_car[i][1] // 4][self.s_car[i][1] % 4][self.s_car[i][2] // 4][self.s_car[i][2] % 4][2] += 1
                elif rest_nods == 1:
                    #self.delete_flag[i] = 1  # 准备删除这个车辆的状态信息
                    delete_flag.append(i)
                else:
                    #delete_s_flag = 0
                    if self.s_light[self.s_car[i][2]][self.s_car[i][3]] == 1:  # 绿灯
                        #delete_s_flag = 1
                        t = self.m
                        self.s_car[i][0] = t
                        del self.s_car[i][1]
                        #self.decode_lines[self.s_car[i][2] // 4][self.s_car[i][2] % 4][self.s_car[i][3] // 4][self.s_car[i][3] % 4][2] += 1
                    else:
                        rewards[self.s_car[i][2]] -= 1  # 红灯，则此时该路口奖励-1
                        if m == self.max_keep_time-1:
                            self.states[self.s_car[i][1]][self.s_car[i][2]] += 1
                        #self.decode_lines[self.s_car[i][2] // 4][self.s_car[i][2] % 4][self.s_car[i][3] // 4][self.s_car[i][3] % 4][1] += 1
                        #self.nod_rewards[self.s_car[i][2] // 4][self.s_car[i][2] % 4] -= 1
                    #if delete_s_flag == 1:
                    #    del self.s_car[i][1]
            '''for i in range(self.n_vih_now):
                if self.delete_flag[i] == 1:
                    self.delete_flag[i] = 0
                    del self.s_car[i - delete_count]
                    delete_count += 1
            self.n_vih_now = self.n_vih_now - delete_count
            for i in range(delete_count):
                del self.delete_flag[i]'''
            for i in delete_flag:
                del self.s_car[i - delete_count]
                delete_count += 1
                self.n_vih_now = self.n_vih_now - 1

        self.keep_time = 0
        '''self.step_total_re = sum(rewards)
        self.total_re += self.step_total_re'''

        '''n_vih_now = len(self.s_car)  # 当前路网中的车辆数
        done_n = [False for _ in range(16)]
        if n_vih_now <= 10:
            for i in range(16):
                done_n[i] = True'''

        #n_vih_now = len(self.s_car)

        '''for i in range(16):
            states.append([])
            for j in range(16):
                states[i].append(0)
        for k in range(self.n_vih_now):
            t = self.s_car[k][0]
            if t == 0:
                states[self.s_car[k][1]][self.s_car[k][2]] += 1'''

        obs_n = []
        for l in range(16):
            obs_n.append(self._get_obs(l))
        for i in range(16):
            rewards[i]=rewards[i]/(100*4*16)
            #rewards[i] = 1/(-rewards[i]+1)
        return obs_n, rewards, self.done_n, {}  # self.decode_lines
