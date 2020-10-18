# Python
import copy
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

# Tensorflow
import tensorflow as tf

# ROS
import rospy
import rospkg

# import our training environment
import gym
from env.ur_door_opening_env import URSimDoorOpening

# import our training algorithms
from algorithm.ppo_gae import PPOGAEAgent

import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

seed = rospy.get_param("/ML/seed")
obs_dim = rospy.get_param("/ML/obs_dim")
n_act = rospy.get_param("/ML/n_act")
epochs = rospy.get_param("/ML/epochs")
hdim = rospy.get_param("/ML/hdim")
policy_lr = rospy.get_param("/ML/policy_lr")
value_lr = rospy.get_param("/ML/value_lr")
max_std = rospy.get_param("/ML/max_std")
clip_range = rospy.get_param("/ML/clip_range")
n_step = rospy.get_param("/ML/n_step")

gamma = rospy.get_param("/ML/gamma")
lam = rospy.get_param("/ML/lam")
episode_size = rospy.get_param("/ML/episode_size")
batch_size = rospy.get_param("/ML/batch_size")
nupdates = rospy.get_param("/ML/nupdates")

agent = PPOGAEAgent(obs_dim, n_act, epochs, hdim, policy_lr, value_lr, max_std, clip_range, seed)
#agent = PPOGAEAgent(obs_dim, n_act, epochs=10, hdim=obs_dim, policy_lr=3e-3, value_lr=1e-3, max_std=1.0, clip_range=0.2, seed=seed)

'''
PPO Agent with Gaussian policy
'''
def run_episode(env, animate=False): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False

    for update in range(n_step):
        obs = np.array(obs)
        obs = obs.astype(np.float32).reshape((1, -1)) # numpy.ndarray (1, num_obs)
        observes.append(obs)
        
        action = agent.get_action(obs) # List
        actions.append(action)
        obs, reward, done, info = env.step(action, update)
        
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward) # List
        infos.append(info)

        if done is True:
            break
        
    return (np.concatenate(observes), np.array(actions), np.array(rewards, dtype=np.float32), infos)

def run_policy(env, episodes): # collect trajectories
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, infos = run_episode(env) # numpy.ndarray
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos} 
        
#        print ("######################run_policy######################")
#        print ("observes: ", observes.shape, type(observes)) 		#(n_step, 21), <type 'numpy.ndarray'>
#        print ("actions: ", actions.shape, type(actions))  		#(n_step,  6), <type 'numpy.ndarray'>
#        print ("rewards: ", rewards.shape, type(rewards))  		#(n_step,   ), <type 'numpy.ndarray'>
#        print ("trajectory: ", len(trajectory), type(trajectory)) 	#(      ,  4), <type 'dict'>
#        print ("#####################run_policy#######################")
        
        trajectories.append(trajectory)
    return trajectories
        
def add_value(trajectories, val_func): # Add value estimation for each trajectories
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.get_value(observes)
        trajectory['values'] = values

def add_gae(trajectories, gamma, lam): # generalized advantage estimation (for training stability)
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        # temporal differences
        
#        print ("###############################add_gae###########################")
#        print ("rewards: ", rewards.shape, type(rewards))  	# (n_step, ), <type 'numpy.ndarray'>
#        print ("values): ", values.shape, type(values))  	# (n_step, ), <type 'numpy.ndarray'>
#        print ("###############################add_gae###########################")
        
        tds = rewards + np.append(values[1:], 0) * gamma - values
        advantages = np.zeros_like(tds)
        advantage = 0
        for t in reversed(range(len(tds))):
            advantage = tds[t] + lam*gamma*advantage
            advantages[t] = advantage
        trajectory['advantages'] = advantages

def add_rets(trajectories, gamma): # compute the returns
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        
        returns = np.zeros_like(rewards)
        ret = 0
        for t in reversed(range(len(rewards))):
            ret = rewards[t] + gamma*ret
            returns[t] = ret            
        trajectory['returns'] = returns

def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])

    # Normalization of advantages 
    # In baselines, which is a github repo including implementation of PPO coded by OpenAI, 
    # all policy gradient methods use advantage normalization trick as belows.
    # The insight under this trick is that it tries to move policy parameter towards locally maximum point.
    # Sometimes, this trick doesnot work.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, returns
    
def main():
    # Can check log msgs according to log_level {rospy.DEBUG, rospy.INFO, rospy.WARN, rospy.ERROR} 
    rospy.init_node('ur_gym', anonymous=True, log_level=rospy.DEBUG)
    
    env = gym.make('URSimDoorOpening-v0')
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed=seed)

    avg_return_list = deque(maxlen=1) # 10
    avg_pol_loss_list = deque(maxlen=1) # 10
    avg_val_loss_list = deque(maxlen=1) # 10
    entropy_list = deque(maxlen=1) # 10

    max_return = 0
    min_return = 0
    max_val_loss = 0
    min_val_loss = 0
    max_pol_loss = 0
    min_pol_loss = 0
    max_entropy = 0
    min_entropy = 0

    # save fig
    x_data = []
    y_data = []
    x_data_v = []
    y_data_v = []
    x_data_p = []
    y_data_p = []
    x_data_e = []
    y_data_e = []
    fig = plt.figure(figsize=(20, 10))
    
    env.first_reset()

    for update in range(nupdates+1):
        trajectories = run_policy(env, episodes=episode_size)
        add_value(trajectories, agent)
        add_gae(trajectories, gamma, lam)
        add_rets(trajectories, gamma)
        observes, actions, advantages, returns = build_train_set(trajectories)
        
#        print ("----------------------------------------------------")
#        print ("update: ", update)
#        print ("updates: ", nupdates)
#        print ("observes: ", observes.shape, type(observes)) 		# ('observes: ',   (n_step, 21), <type 'numpy.ndarray'>)
#        print ("advantages: ", advantages.shape, type(advantages))	# ('advantages: ', (n_step,),    <type 'numpy.ndarray'>)
#        print ("returns: ", returns.shape, type(returns)) 		# ('returns: ',    (n_step,),    <type 'numpy.ndarray'>)
#        print ("actions: ", actions.shape, type(actions)) 		# ('actions: ',    (n_step, 6),  <type 'numpy.ndarray'>)
#        print ("----------------------------------------------------")

        pol_loss, val_loss, kl, entropy = agent.update(observes, actions, advantages, returns, batch_size=batch_size)

        avg_pol_loss_list.append(pol_loss)
        avg_val_loss_list.append(val_loss)
        avg_return_list.append([np.sum(t['rewards']) for t in trajectories])
        entropy_list.append(entropy)

        if (update%1) == 0:
            print('[{}/{}] n_step : {},return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}'.format(
                update, nupdates, observes.shape, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list), kl, entropy))
        if max_return < np.mean(avg_return_list):
                max_return = np.mean(avg_return_list)
        if min_return > np.mean(avg_return_list):
                min_return = np.mean(avg_return_list)
        if max_val_loss < np.mean(avg_val_loss_list):
                max_val_loss = np.mean(avg_val_loss_list)
        if min_val_loss > np.mean(avg_val_loss_list):
                min_val_loss = np.mean(avg_val_loss_list)
        if max_pol_loss < np.mean(avg_pol_loss_list):
                max_pol_loss = np.mean(avg_pol_loss_list)
        if min_pol_loss > np.mean(avg_pol_loss_list):
                min_pol_loss = np.mean(avg_pol_loss_list)
        if max_entropy < entropy:
                max_entropy = entropy
        if min_entropy > entropy:
                min_entropy = entropy

        x_data.append(update)
        y_data.append(np.mean(avg_return_list))
        x_data_v.append(update)
        y_data_v.append(np.mean(avg_val_loss_list))
        x_data_p.append(update)
        y_data_p.append(np.mean(avg_pol_loss_list))
        x_data_e.append(update)
        y_data_e.append(entropy)


        if (update%10) == 0:
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(x_data, y_data, 'r-')
            ax1.set_xlabel("episodes")
            ax1.set_ylabel("ave_return")
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(x_data_v, y_data_v, 'b-')
            ax2.set_xlabel("episodes")
            ax2.set_ylabel("ave_val_loss")
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(x_data_p, y_data_p, 'g-')
            ax3.set_xlabel("episodes")
            ax3.set_ylabel("ave_pol_loss")
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.plot(x_data_e, y_data_e, 'c-')
            ax4.set_xlabel("episodes")
            ax4.set_ylabel("entropy")

            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            plt.draw()
            plt.pause(1e-17)
            plt.savefig("./results/ppo_with_gae_list.png")

        if (np.mean(avg_return_list) > 3140): # Threshold return to success 
            print('[{}/{}] return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}'.format(update,nupdates, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break

        #env.close() # rospy.wait_for_service('/pause_physics') -> raise ROSInterruptException("rospy shutdown")

if __name__ == '__main__':
    main()
