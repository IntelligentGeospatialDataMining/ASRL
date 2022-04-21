"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/
Environment
-----------
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
from model.DDPG import DDPG
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
MAX_EPOCHS=50            # 训练次数
MAX_EPISODES = 10 # total number of episodes for training
MAX_EP_STEPS = 100000 # total number of steps for each episode
TEST_PER_EPISODES = 1  # test the model per episodes
VAR = 5  # control exploration


reward_type='reward-irl'   #experiment label




if __name__ == '__main__':
    # 初始化环境
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    #env.get_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    # 定义状态空间，动作空间，动作幅度范围
    s_dim = env.state.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    print('s_dim', s_dim)
    print('a_dim', a_dim)


    ddpg = DDPG(a_dim, s_dim, a_bound)

    if args.train:  # train
        actions=[]
        reward_buffer = []
        t0 = time.time()  # 统计时间
        for e in range(MAX_EPOCHS):
            epoch_reward = 0  # 记录当前EP的reward

            for i in range(MAX_EPISODES):
                t1 = time.time()
                s,dir= env.reset()
                ep_reward=0

                for j in range(MAX_EP_STEPS):
                    # Add exploration noise
                    a = ddpg.choose_action(s)  # 这里很简单，直接用actor估算出a动作



                    # 为了能保持开发，这里用了另外一种方式增加探索。
                    # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                    # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                    # 然后进行裁剪
                    a = np.clip(np.random.normal(a, VAR), 1, a_bound)
                    a=int(a)
                    actions.append(a)

                    # 与环境进行互动
                    s_, r, done, info = env.step(a)
                    if len(info)>1:
                        pe = pd.DataFrame(actions)
                        path = 'results/' + reward_type + '/' + dir
                        folder = os.path.exists(path)
                        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                            os.makedirs(path)
                        folder = os.path.exists(path + '/' + 'ddpg-periods.txt')
                        pe.to_csv('results/'+reward_type+'/' + dir + '/' + 'ddpg-periods.txt', index=False)
                        actions.clear()
                        dir=info[1]

                    if not done:
                       ddpg.store_transition(s, a, r/10 , s_)


                    # 第一次数据满了，就可以开始学习
                    if ddpg.pointer > MEMORY_CAPACITY:
                        ddpg.learn()

                    # 输出数据记录
                    s = s_
                    ep_reward += r  # 记录当前EP的总reward
                    if done:           #最后一步前输出时间和奖励等
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
                if i and not i % TEST_PER_EPISODES:           #每隔一定次数测试一次
                    t1 = time.time()
                    periods=[]
                    s,dir,txt = env.reset()
                    ep_reward = 0
                    for j in range(MAX_EP_STEPS):
    
                        a = ddpg.choose_action(s)  # 注意，在测试的时候，我们就不需要用正态分布了，直接一个a就可以了。
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
                            reward_buffer.append(ep_reward)           #保留测试的奖励结果绘制成图像
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
        pe.to_csv(reward_type+'.txt', index=False)
        ddpg.save_ckpt(reward_type)


