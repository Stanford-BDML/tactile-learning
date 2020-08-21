from collections import deque
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from algorithm.REINFORCEAgent import REINFORCEAgent
#from ur_reaching.algorithm.REINFORCEAgent import REINFORCEAgent

# import our training environment
import gym
from env.ur_reaching_env import URSimReaching
#from ur_reaching.env.ur_reaching_env import URSimReaching

import rospy
import rospkg

seed = 0
obs_dim = 15 # env.observation_space.shape[0]
n_act = 6 #config: act_dim #env.action_space.n
agent = REINFORCEAgent(obs_dim, n_act, epochs=5, hdim=32, lr=3e-4,seed=seed)


'''
PPO Agent with Gaussian policy
'''

def run_episode(env, agent): # use 'run_episode' function for 1 episode and collect(state, action, reward) (URSimReaching-v0 , REINFORCEAgent)
    obs = env.reset()						# reset the env and get the initial observation ①現在状態(observation_t)（A→B)
    observes, actions, rewards, infos = [], [], [], []      	# make the lists for observations, actions, rewards, and infos
    done = False						# set the done flag as False

    for update in range(1000):                              # 1エピソードのループ(1000steps)
        action = agent.get_action([obs])                    # ②③行動(action_t)（B→A→E), REINFORCEAgent.py
        
        next_obs, reward, done, info = env.step(action)     # ④次の状態、報酬、終了条件(observation_t+1、Done)（E→A)
        
        observes.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)

        obs = next_obs                                      # 現在の状態obsを次の状態next_obsに更新

        if done is True:
            break

    return np.asarray(observes), np.asarray(actions), np.asarray(rewards), infos

def run_policy(env, agent, episodes): # collect trajectories. if 'evaluation' is ture, then only mean value of policy distribution is used without sampling.
                                      # 所定のエピソード数までのループ（このプログラムでは１エピソード分のみ）
    total_steps = 0                # initialized the step number
    trajectories = []              # make the box for trajectories
    for e in range(episodes):      # run the episode loop
        observes, actions, rewards, infos = run_episode(env, agent)  # run the 'run_episode' function（１エピソードの計算：①現在の状態〜④次の状態）
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos}
        trajectories.append(trajectory)  # メモリに経験を追加　⑤次の状態(observation_t+1)、行動(action_t)、報酬(reward_t+1)の更新
    return trajectories

def build_train_set(trajectories):       # observes, actions, returnsの箱準備
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])

    return observes, actions, returns

def compute_returns(trajectories, gamma=0.995): # Add value estimation for each trajectories（状態価値の計算？）
    for trajectory in trajectories:
        rewards = trajectory['rewards']         # trajectoryからrewardsを抽出
        returns = np.zeros_like(rewards)        # return array of 0(rewards用の箱準備)
        g = 0
        for t in reversed(range(len(rewards))): # 状態価値の計算？
            g = rewards[t] + gamma*g		# update the total discounted reward
            returns[t] = g
        trajectory['returns'] = returns


def main():
    # Can check log msgs according to log_level {rospy.DEBUG, rospy.INFO, rospy.WARN, rospy.ERROR} 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.INFO)		   # Define ros node "ur_gym"
    # Define Parameters and variables for sim models and RL
    env = gym.make('URSimReaching-v0')						   # call ur_reaching_env.py (make new invironment in gym application)
    env._max_episode_steps = 10000						   # define max episode steps
    np.random.seed(seed)							   # makes the random numbers in np(Ex.numpy.random.seed(0) ; numpy.random.rand(4)-> array([ 0.55,  0.72,  0.6 ,  0.54]))
    tf.set_random_seed(seed)							   # makes the random numbers in tf
    env.seed(seed=seed)								   # define random param "seed" in env

    avg_return_list = deque(maxlen=1000)					   # define return_list and loss_list as "deque" type which is easy to access the first and last element in the list
    avg_loss_list = deque(maxlen=1000)

    episode_size = 1
    batch_size = 16
    nupdates = 100000

    # repeat the main loop
    for update in range(nupdates+1):                                               # 100001エピソード分のループ
        #print ('update: ', update)
        trajectories = run_policy(env, agent, episodes=episode_size)               # run the 'run_policy' function：１エピソードのみを実行する
        compute_returns(trajectories)                                              # calculate rewards by 'compute_returns'
        observes, actions, returns = build_train_set(trajectories)                 # 'build_train_set'を使ってtrajectoriesからobservers, actions, retrunsを抽出して分割

        pol_loss = agent.update(observes, actions, returns, batch_size=batch_size) # ⑥テーブルの更新
        avg_loss_list.append(pol_loss)                                             # appendでテーブルをリストに追加
        avg_return_list.append([np.sum(t['rewards']) for t in trajectories])	   # calculate sum of rewards and append to the list
        
        if (update%1)==0:
            print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update, nupdates, np.mean(avg_loss_list), np.mean(avg_return_list)))
            
        if (np.mean(avg_return_list) > 10000) and np.shape(np.mean(avg_loss_list)) == np.shape(np.mean(avg_return_list)): # Threshold return to success cartpole
            print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update, nupdates, np.mean(avg_loss_list), np.mean(avg_return_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break
	
    #env.close()

if __name__ == '__main__':
    main()
