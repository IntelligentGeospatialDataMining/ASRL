import pandas as pd
import os
import time
from model.DDPG import DDPG
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#####################  hyper parameters  ####################

ENV_NAME = 'env_name-v1'  # environment name      env_name-v0
RANDOMSEED = 1  # random seed
MEMORY_CAPACITY = 20000  # size of replay buffer
MAX_EPOCHS=1            # 训练次数
MAX_EPISODES = 1 # total number of episodes for training
MAX_EP_STEPS = 100000  # total number of steps for each episode
TEST_PER_EPISODES = 1  # test the model per episodes
VAR = 5  # control exploration


reward_type='reward-irl'   #train model type
result_type='reward-irl-test'   #store test result



# 初始化环境
env = gym.make(ENV_NAME)
env = env.unwrapped

np.random.seed(RANDOMSEED)
tf.random.set_seed(RANDOMSEED)
# 定义状态空间，动作空间，动作幅度范围
s_dim = env.state.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

print('s_dim', s_dim)
print('a_dim', a_dim)

ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg.load_ckpt(reward_type)
actions = []
state_action=[]
reward_buffer = []  # 用于记录每个EP的reward，统计变化
t0 = time.time()  # 统计时间
for e in range(MAX_EPOCHS):
    epoch_reward = 0  # 记录当前EP的reward
    for i in range(MAX_EPISODES):
        t1 = time.time()
        s, dir = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = ddpg.choose_action(s)  # 这里很简单，直接用actor估算出a动作

            a = int(a)
            if(a==0): a=1
            actions.append(a)
            # 与环境进行互动
            s_, r, done, info = env.step(a)
            if len(info) > 1:
                pe = pd.DataFrame(actions)
                path = 'results/' + result_type + '/' + dir
                folder = os.path.exists(path)
                if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(path)
                folder = os.path.exists(path + '/' + 'ddpg-periods.txt')

                if folder:
                    os.remove(path + '/' + 'ddpg-periods.txt')
                    open(path + '/' + 'ddpg-periods.txt','w')

                pe.to_csv('results/' + result_type + '/' + dir + '/' + 'ddpg-periods.txt', index=False)
                actions.clear()
                dir = info[1]
            state_action.append([s[0],s[1],s[2],s[3],a])

            s = s_
            ep_reward += r  # 记录当前EP的总reward
            if done:  # 最后一步前输出时间和奖励等
                epoch_reward += ep_reward
                reward_buffer.append(ep_reward)

                actions.clear()
                break
            plt.show()

print('\nRunning time: ', time.time() - t0)
pe = pd.DataFrame(reward_buffer)
# ress=pd.DataFrame(state_action)    #store state and action
# ress.to_csv('state_and_action.txt', index=False,sep='\t')
pe.to_csv('reward.txt', index=False)  #store reward
