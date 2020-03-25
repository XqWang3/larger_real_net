import sys, os; sys.path.append(os.getcwd())
import os.path as osp
this_abs_path = osp.abspath(__file__)
experiment_dir = osp.dirname(this_abs_path)
maddpg_dir = osp.dirname(experiment_dir)
root_dir = osp.dirname(maddpg_dir)
models_dir = osp.join(root_dir, "models")

sys.path.extend([
    experiment_dir,
    maddpg_dir,
    root_dir,
])

import time
import matplotlib
#matplotlib.rcParams['backend'] = 'PDF'
import matplotlib.pyplot as plt
import numpy as np
#from logger import Logger
from traffic import TrafficEnv
import experiments.serialization_utils as sut
def test():

	data_collector = dict()
	def upload(key, val):
		if key not in data_collector:
			data_collector[key] = list()
		data_collector[key].append(val)

	def mkdir(directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	#mkdir(logs_dir)
	i=0
	video_filename= 'episode_2.npz'
	#exp_name = 'dedededebug'
	#====================
	each_epi_step = 200
	exp_name = 'traffic_flow2'
	video_replay = False  #True  or False
	save_episode = False	#True  or False
	save_fig = False		#True  or False
	env = TrafficEnv(flag_traffic_flow=2)   #默认为第一种交通流（回字形）;=2（车流类型为全局随机）;=3（四个口字型车流）
	#====================
	#mkdir(models_dir)
	load_dir = ''
	exp_dir = osp.join(models_dir, exp_name)
	if save_episode == True:
		pass
		#mkdir(exp_dir)

	episode_rewards = [0.0]  # sum of rewards for all agents
	agent_rewards = [[0.0] for _ in range(16)]  # individual agent reward
	final_ep_rewards = []  # sum of rewards for training curve
	final_ep_ag_rewards = []  # agent rewards for training curve
	agent_info = [[[]]]  # placeholder for benchmarking info
	#saver = tf.train.Saver()

	t_start = time.time()

	if video_replay == True:
		video_serialization = sut.npz_to_serialization(osp.join(exp_dir, video_filename))
		env.reset()
		idx = 0
		len_episode = sut.len_serialization(video_serialization)
		while True:
			frame = sut.index_serialization(video_serialization, idx)
			idx = (idx + 1) % len_episode
			env.deserialize(frame)
			time.sleep(0.1)
			env.render(mode="inhuman")

	else:
		arr = []
		env.reset()
		video_log_proto = sut.create_serialization(env.serialize())
		from copy import deepcopy

		train_step = 0

		for episode_id in range(5):
			env.reset()
			video_episode = ((episode_id + 1) % 2 == 0)
			video_log = deepcopy(video_log_proto)
			rewards = []
			total_re = 0
			mean_re = 0
			for step_id in range(each_epi_step):
				# get action
				action_n = [space.sample() for space in env.action_space]
				new_obs_n, rew_n, done_n, info_n = env.step(action_n)

				arr = env.render(return_rgb_array=True)

				re = sum(rew_n)
				total_re += re
				mean_re = total_re/(step_id+1)
				#rewards.append(re)
				rewards.append(mean_re)

				if not hasattr(done_n, '__iter__'):
					done_n = [done_n] * len(rew_n)
				done = all(done_n)
				terminal = (step_id + 1 == each_epi_step)

				for i, rew in enumerate(rew_n):
					episode_rewards[-1] += rew
					agent_rewards[i][-1] += rew
					upload("reward_{}".format(i), rew)

				if done or terminal:
					#obs_n = env.reset()
					episode_rewards.append(0)
					for a in agent_rewards:
						a.append(0)
					agent_info.append([[]])

				# increment global step counter
				train_step += 1

				if video_episode:
					sut.append_serialization(video_log, env.serialize())

				# save model, display training output
				if terminal and ((len(episode_rewards) + 1) % 5 == 0):  # each 5th episode saving
					print('saving model...')
					print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
						train_step, len(episode_rewards)-1, np.mean(episode_rewards[-5:]),
						round(time.time() - t_start, 3)))
					t_start = time.time()
					# Keep track of final episode reward
					final_ep_rewards.append(np.mean(episode_rewards[-5:]))
					for rew in agent_rewards:
						final_ep_ag_rewards.append(np.mean(rew[-5:]))
				#plt.imshow(arr)
				#plt.savefig('{}_{}.pdf'.format(exp_name,train_step),format='pdf')
				plt.imsave('{}_{}.pdf'.format(exp_name,train_step),arr)

			if episode_id == 0 and save_fig:
				x = [i for i in range(len(rewards))]
				y = rewards
				plt.figure()
				plt.title('mean reward of each step in a episode({}_{}_step)'.format(exp_name,each_epi_step))
				plt.plot(x,y)
				plt.xlabel("step_count")
				plt.ylabel("mean_re/step")
				#plt.savefig("{}_{}_step.eps".format(exp_name,each_epi_step))
				plt.show()

			if video_episode and save_episode:
				sut.serialization_to_npz(video_log, osp.join(exp_dir, 'episode_{}'.format(episode_id + 1)))

if __name__=="__main__":
	test()
