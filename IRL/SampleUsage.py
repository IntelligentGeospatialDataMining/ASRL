
from IRL.RelEntIRL import RelEntIRL
import pandas as pd
import numpy as np
import os

def ReadExpertTraj():
    path='./resource/ExpertTrajectories'
    dir =os.listdir(r'F:\global-map-matching-dataset')
    res=[]
    states=pd.read_table('./resource/state_detail.txt')
    for each in dir:
        one=[]
        traj=pd.read_table(path+'/'+each+'/expert_traj.txt')
        for i in range(len(traj)):
            state=states.iloc[traj.iloc[i,0],:]
            action=traj.iloc[i,1]
            # one.append([float(state[0]),float(state[1]),float(state[2]),float(state[3])])
            one.append([float(state[0]), float(state[1]), float(state[2]), float(state[3]),float(action)])
            # one.append([float(state[0]), float(state[1]), float(state[2])])
        res.append(one)
    return np.array(res)

def ReadRandomTraj(i):
    path = './resource/RandomTrajectories'
    dir = os.listdir(path)
    dir=dir[0:(i+1)*64]    #choose the trajectory num,one directory stands for one trajectory
    res = []
    states = pd.read_table('./resource/state_detail.txt')
    for each in dir:
        one = []
        traj = pd.read_table(path + '/' + each + '/random_traj.txt')
        for i in range(len(traj)):
            state = states.iloc[traj.iloc[i, 0], :]
            action = traj.iloc[i, 1]
            # one.append([float(state[0]),float(state[1]),float(state[2]),float(state[3])])
            one.append([float(state[0]), float(state[1]), float(state[2]), float(state[3]), float(action)])
            # one.append([float(state[0]), float(state[1]), float(state[2])])
        res.append(one)
    return np.array(res)


if __name__ == '__main__':
    expert_demo=ReadExpertTraj()
    for i in range(0,1):
        random_demo = ReadRandomTraj(i)
        relent = RelEntIRL(expert_demo, random_demo)
        relent.train()
        print(relent.weights)






