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

    maxlen_num = 10
    avg_return_list = deque(maxlen=maxlen_num) # 10
    avg_pol_loss_list = deque(maxlen=maxlen_num) # 10
    avg_val_loss_list = deque(maxlen=maxlen_num) # 10
    avg_entropy_list = deque(maxlen=maxlen_num) # 10
    avg_max_knob_rotation_list = deque(maxlen=maxlen_num) # 10
    avg_max_door_rotation_list = deque(maxlen=maxlen_num) # 10
    max_wrist3_list = deque(maxlen=maxlen_num) # 10
    min_wrist3_list = deque(maxlen=maxlen_num) # 10
    max_wrist2_list = deque(maxlen=maxlen_num) # 10
    min_wrist2_list = deque(maxlen=maxlen_num) # 10
    max_wrist1_list = deque(maxlen=maxlen_num) # 10
    min_wrist1_list = deque(maxlen=maxlen_num) # 10
    max_elb_list = deque(maxlen=maxlen_num) # 10
    min_elb_list = deque(maxlen=maxlen_num) # 10
    max_shl_list = deque(maxlen=maxlen_num) # 10
    min_shl_list = deque(maxlen=maxlen_num) # 10
    max_shp_list = deque(maxlen=maxlen_num) # 10
    min_shp_list = deque(maxlen=maxlen_num) # 10
    max_force_x_list = deque(maxlen=maxlen_num) # 10
    min_force_x_list = deque(maxlen=maxlen_num) # 10
    max_force_y_list = deque(maxlen=maxlen_num) # 10
    min_force_y_list = deque(maxlen=maxlen_num) # 10
    max_force_z_list = deque(maxlen=maxlen_num) # 10
    min_force_z_list = deque(maxlen=maxlen_num) # 10
    max_torque_x_list = deque(maxlen=maxlen_num) # 10
    min_torque_x_list = deque(maxlen=maxlen_num) # 10
    max_torque_y_list = deque(maxlen=maxlen_num) # 10
    min_torque_y_list = deque(maxlen=maxlen_num) # 10
    max_torque_z_list = deque(maxlen=maxlen_num) # 10
    min_torque_z_list = deque(maxlen=maxlen_num) # 10

    # save fig
    x_data = []
    y_data = []
    x_data_v = []
    y_data_v = []
    x_data_p = []
    y_data_p = []
    x_data_e = []
    y_data_e = []
    x_data_k = []
    y_data_k = []
    x_data_d = []
    y_data_d = []
    x_data_a = []
    y_data_max_wrist3 = []
    y_data_min_wrist3 = []
    y_data_max_wrist2 = []
    y_data_min_wrist2 = []
    y_data_max_wrist1 = []
    y_data_min_wrist1 = []
    y_data_max_elb = []
    y_data_min_elb = []
    y_data_max_shl = []
    y_data_min_shl = []
    y_data_max_shp = []
    y_data_min_shp = []
    x_data_f = []
    y_data_max_force_x = []
    y_data_min_force_x = []
    y_data_max_force_y = []
    y_data_min_force_y = []
    y_data_max_force_z = []
    y_data_min_force_z = []
    y_data_max_torque_x = []
    y_data_min_torque_x = []
    y_data_max_torque_y = []
    y_data_min_torque_y = []
    y_data_max_torque_z = []
    y_data_min_torque_z = []
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
        avg_entropy_list.append(entropy)
        avg_max_knob_rotation_list.append(env.max_knob_rotation)
        avg_max_door_rotation_list.append(env.max_door_rotation)
        max_wrist3_list.append(env.max_wirst3)
        min_wrist3_list.append(env.min_wirst3)
        max_wrist2_list.append(env.max_wirst2)
        min_wrist2_list.append(env.min_wirst2)
        max_wrist1_list.append(env.max_wirst1)
        min_wrist1_list.append(env.min_wirst1)
        max_elb_list.append(env.max_elb)
        min_elb_list.append(env.min_elb)
        max_shl_list.append(env.max_shl)
        min_shl_list.append(env.min_shl)
        max_shp_list.append(env.max_shp)
        min_shp_list.append(env.min_shp)
        max_force_x_list.append(env.max_force_x)
        min_force_x_list.append(env.min_force_x)
        max_force_y_list.append(env.max_force_y)
        min_force_y_list.append(env.min_force_y)
        max_force_z_list.append(env.max_force_z)
        min_force_z_list.append(env.min_force_z)
        max_torque_x_list.append(env.max_torque_x)
        min_torque_x_list.append(env.min_torque_x)
        max_torque_y_list.append(env.max_torque_y)
        min_torque_y_list.append(env.min_torque_y)
        max_torque_z_list.append(env.max_torque_z)
        min_torque_z_list.append(env.min_torque_z)

        if (update%1) == 0:
            print('[{}/{}] n_step : {}, return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}'.format(
                update, nupdates, returns.shape, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list), kl, entropy))

        x_data.append(update)
        y_data.append(np.mean(avg_return_list))
        x_data_v.append(update)
        y_data_v.append(np.mean(avg_val_loss_list))
        x_data_p.append(update)
        y_data_p.append(np.mean(avg_pol_loss_list))
        x_data_e.append(update)
        y_data_e.append(np.mean(avg_entropy_list))
        x_data_k.append(update)
        y_data_k.append(np.mean(avg_max_knob_rotation_list))
        x_data_d.append(update)
        y_data_d.append(np.mean(avg_max_door_rotation_list))
        x_data_a.append(update)
        y_data_max_wrist3.append(np.mean(max_wrist3_list))
        y_data_min_wrist3.append(np.mean(min_wrist3_list))
        y_data_max_wrist2.append(np.mean(max_wrist2_list))
        y_data_min_wrist2.append(np.mean(min_wrist2_list))
        y_data_max_wrist1.append(np.mean(max_wrist1_list))
        y_data_min_wrist1.append(np.mean(min_wrist1_list))
        y_data_max_elb.append(np.mean(max_elb_list))
        y_data_min_elb.append(np.mean(min_elb_list))
        y_data_max_shl.append(np.mean(max_shl_list))
        y_data_min_shl.append(np.mean(min_shl_list))
        y_data_max_shp.append(np.mean(max_shp_list))
        y_data_min_shp.append(np.mean(min_shp_list))
        x_data_f.append(update)
        y_data_max_force_x.append(np.mean(max_force_x_list))
        y_data_min_force_x.append(np.mean(min_force_x_list))
        y_data_max_force_y.append(np.mean(max_force_y_list))
        y_data_min_force_y.append(np.mean(min_force_y_list))
        y_data_max_force_z.append(np.mean(max_force_z_list))
        y_data_min_force_z.append(np.mean(min_force_z_list))
        y_data_max_torque_x.append(np.mean(max_torque_x_list))
        y_data_min_torque_x.append(np.mean(min_torque_x_list))
        y_data_max_torque_y.append(np.mean(max_torque_y_list))
        y_data_min_torque_y.append(np.mean(min_torque_y_list))
        y_data_max_torque_z.append(np.mean(max_torque_z_list))
        y_data_min_torque_z.append(np.mean(min_torque_z_list))

        if (update%1) == 0:
            ax1 = fig.add_subplot(2, 4, 1)
            ax1.plot(x_data, y_data, 'r-')
            ax1.set_xlabel("episodes")
            ax1.set_ylabel("ave_return")
            ax2 = fig.add_subplot(2, 4, 2)
            ax2.plot(x_data_v, y_data_v, 'b-')
            ax2.set_xlabel("episodes")
            ax2.set_ylabel("ave_val_loss")
            ax3 = fig.add_subplot(2, 4, 3)
            ax3.plot(x_data_p, y_data_p, 'g-')
            ax3.set_xlabel("episodes")
            ax3.set_ylabel("ave_pol_loss")
            ax4 = fig.add_subplot(2, 4, 4)
            ax4.plot(x_data_e, y_data_e, 'c-')
            ax4.set_xlabel("episodes")
            ax4.set_ylabel("entropy")
            ax5 = fig.add_subplot(2, 4, 5)
            ax5.plot(x_data_k, y_data_k, 'r-')
            ax5.plot(x_data_k, y_data_d, 'b-')
            ax5.set_xlabel("episodes")
            ax5.set_ylabel("max_knob&door_rotation")
            ax6 = fig.add_subplot(2, 4, 6)
            ax6.plot(x_data_a, y_data_max_wrist3, 'r-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_wrist3, 'r-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_wrist2, 'b-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_wrist2, 'b-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_wrist1, 'g-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_wrist1, 'g-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_elb, 'c-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_elb, 'c-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_shl, 'm-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_shl, 'm-', linestyle="dashed")
            ax6.plot(x_data_a, y_data_max_shp, 'k-', linestyle="solid")
            ax6.plot(x_data_a, y_data_min_shp, 'k-', linestyle="dashed")
            ax6.set_xlabel("episodes")
            ax6.set_ylabel("max&min_action")
            ax7 = fig.add_subplot(2, 4, 7)
            ax7.plot(x_data_f, y_data_max_force_x, 'r-', linestyle="solid")
            ax7.plot(x_data_f, y_data_min_force_x, 'r-', linestyle="dashed")
            ax7.plot(x_data_f, y_data_max_force_y, 'b-', linestyle="solid")
            ax7.plot(x_data_f, y_data_min_force_y, 'b-', linestyle="dashed")
            ax7.plot(x_data_f, y_data_max_force_z, 'g-', linestyle="solid")
            ax7.plot(x_data_f, y_data_min_force_z, 'g-', linestyle="dashed")
            ax7.set_xlabel("episodes")
            ax7.set_ylabel("max&min_force")
            ax8 = fig.add_subplot(2, 4, 8)
            ax8.plot(x_data_f, y_data_max_torque_x, 'r-', linestyle="solid")
            ax8.plot(x_data_f, y_data_min_torque_x, 'r-', linestyle="dashed")
            ax8.plot(x_data_f, y_data_max_torque_y, 'b-', linestyle="solid")
            ax8.plot(x_data_f, y_data_min_torque_y, 'b-', linestyle="dashed")
            ax8.plot(x_data_f, y_data_max_torque_z, 'g-', linestyle="solid")
            ax8.plot(x_data_f, y_data_min_torque_z, 'g-', linestyle="dashed")
            ax8.set_xlabel("episodes")
            ax8.set_ylabel("max&min_torque")

            fig.subplots_adjust(hspace=0.3, wspace=0.4)
            plt.draw()
            plt.pause(1e-17)
            plt.savefig("./results/ppo_with_gae_list.png")

        if (np.mean(avg_return_list) > 12800): # Threshold return to success 
            print('[{}/{}] return : {:.3f}, value loss : {:.3f}, policy loss : {:.3f}'.format(update,nupdates, np.mean(avg_return_list), np.mean(avg_val_loss_list), np.mean(avg_pol_loss_list)))
            print('The problem is solved with {} episodes'.format(update*episode_size))
            break

        #env.close() # rospy.wait_for_service('/pause_physics') -> raise ROSInterruptException("rospy shutdown")

if __name__ == '__main__':
    main()
