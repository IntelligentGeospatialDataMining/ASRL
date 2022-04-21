"""
Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
To run
------
python tutorial_DDPG.py --train/test
"""
import pandas as pd
import argparse
import os
import time
from DDPG import DDPG
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'env_name-v1'  # environment name
RANDOMSEED = 1  # random seed

MEMORY_CAPACITY = 20000  # size of replay buffer
MAX_EPOCHS=50            # number of epochs
MAX_EPISODES = 10 # total number of episodes for training
MAX_EP_STEPS = 100000 # total number of steps for each episode
TEST_PER_EPISODES = 1  # test the model per episodes
VAR = 5  # control exploration


run_log='reward-irl'   #experiment label




if __name__ == '__main__':
    # 初始化环境
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    #env.get_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    # define state space and action space
    s_dim = env.state.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    print('s_dim', s_dim)
    print('a_dim', a_dim)


    ddpg = DDPG(a_dim, s_dim, a_bound)

    if args.train:  # train
        actions=[]
        reward_buffer = []
        t0 = time.time()  # time start
        for e in range(MAX_EPOCHS):
            epoch_reward = 0  # record reward of current ep

            for i in range(MAX_EPISODES):
                t1 = time.time()
                s,dir= env.reset()
                ep_reward=0

                for j in range(MAX_EP_STEPS):

                    a = ddpg.choose_action(s)
                    # Add exploration noise
                    a = np.clip(np.random.normal(a, VAR), 1, a_bound)
                    a=int(a)
                    actions.append(a)

                    # interact with env
                    s_, r, done, info = env.step(a)
                    if len(info)>1:
                        pe = pd.DataFrame(actions)
                        path = 'results/' + run_log + '/' + dir
                        folder = os.path.exists(path)
                        if not folder:  # new a folder
                            os.makedirs(path)
                        folder = os.path.exists(path + '/' + 'ddpg-periods.txt')
                        pe.to_csv('results/'+run_log+'/' + dir + '/' + 'ddpg-periods.txt', index=False)
                        actions.clear()
                        dir=info[1]

                    if not done:
                       ddpg.store_transition(s, a, r/10 , s_)


                    # start to learn
                    if ddpg.pointer > MEMORY_CAPACITY:
                        ddpg.learn()


                    s = s_
                    ep_reward += r
                    if done:           #print time and reward at last step
                        print(
                            '\rTrain:Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                e*10+i, MAX_EPISODES, ep_reward,
                                time.time() - t1
                            ), end=''
                        )
                        epoch_reward+=ep_reward
                        reward_buffer.append(ep_reward)

                        actions.clear()
                        break
                    plt.show()


            '''
                # test
                if i and not i % TEST_PER_EPISODES:           #test in a while
                    t1 = time.time()
                    periods=[]
                    s,dir,txt = env.reset()
                    ep_reward = 0
                    for j in range(MAX_EP_STEPS):
    
                        a = ddpg.choose_action(s)  # we don't need to add noise during test
                        a = abs(a)
                        a = int(a)
                        periods.append(a)  # 保留每一步的动作值
                        s_, r, done, info = env.step(a)
                        #print("s,s_,test reward,action:",s,s_,r, a)
                        s = s_
                        ep_reward += r
                        if done:
                            print(
                                '\rTest:Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                    i, MAX_EPISODES, ep_reward,
                                    time.time() - t1
                                )
                            )
                            reward_buffer.append(ep_reward)           
                            pe = pd.DataFrame(periods)
                            pe.to_csv('results/'+reward_type+'/' + dir + '/' + 'ddpg-periods.txt', index=False)
                            break
                '''
            if reward_buffer:
                plt.ion()
                plt.cla()
                plt.title('DDPG')
                plt.plot(np.array(range(len(reward_buffer))) , reward_buffer)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.ylim(-300000,300000)
                plt.show()
                plt.pause(0.1)


        plt.ioff()
        plt.show()
        print('\nRunning time: ', time.time() - t0)
        pe = pd.DataFrame(reward_buffer)
        pe.to_csv(run_log+'.txt', index=False)
        ddpg.save_ckpt(run_log)


