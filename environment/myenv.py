from gym import spaces, core
import numpy as np
import os
import pandas as pd
import geohash

from scipy.stats import norm
import geopandas
import math
from pyproj import Transformer
from geopy.distance import geodesic


prar='Train'

class MyEnv(core.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=1, high=80,shape=(1,1))  # action space
        self.state=spaces.Box(low=0,high=5,shape=(4,1))         # state space
        self.points=[]
        self.current_index=0
        self.point_length=0
        self.done=False
        self.seed=0
        self.speeds=[]
        self.data_path = 'F:/global-map-matching-dataset/'+prar+'/'           #read points data to train model ,if test,then read test data
        self.directory=os.listdir(self.data_path)
        self.record=[]
        self.geohash_road=self.ReadRoadComplexity('../resource/geohash/road_geohash.txt')             #read density_geohash file
        self.geohash_noise=self.ReadNoise('../resource/geohash/noise_geohash.txt')            #read noise_geohash file


        self.radius=100
        self.directory_num=0

    def reset(self):

        direc=self.Choose_Tra(self.directory_num)
        self.points = self.points.values.tolist()
        self.current_index = 0
        self.done = False
        self.point_length = len(self.points)

        self.speeds=self.ReadSpeed(direc)
        speed=self.Speed_cau(self.speeds[self.current_index][0])
        road_complex = self.RoadComplexity_cau(self.points[0][0], self.points[0][1])
        noise = self.Noise_cau(self.points[0][0], self.points[0][1])
        angle = 0

        self.state=[road_complex, noise, angle,speed]

        return np.array(self.state),direc


    def Choose_Tra(self,direc):

        self.points = pd.read_table(self.data_path + self.directory[direc] + "/track.txt",header=None)
        return  self.directory[direc]


    def step(self, action):
        old_state = self.state
        old_index=self.current_index
        # print(self.current_index)
        self.current_index+=action

        #print(self.current_index)
        if(self.current_index<=self.point_length-1):

            road_complex = self.RoadComplexity_cau(self.points[self.current_index][0], self.points[self.current_index][1])
            noise=self.Noise_cau(self.points[self.current_index][0],self.points[self.current_index][1])
            angle=self.Angle_cau2(self.points[self.current_index],self.points[self.current_index-action])
            speed=self.Speed_cau(self.speeds[self.current_index][0])

            self.state = [road_complex, noise, angle,speed]



            reward=self.reward_irl(old_state,action)

            info = []  # It is used to record the environmental information during the training process, which is convenient to observe the training status.
        else:                                              #If the current index reaches or exceeds the trajectory length, the state of the last point is taken
            self.current_index=self.point_length-1

            if (self.directory_num == len(self.directory) - 1):  # all tarj done
                self.done = True

                reward = self.reward_irl(old_state, action)

                self.directory_num=0
                info = [1,1]

            else:
                  # one directory done
                self.directory_num += 1

                reward = self.reward_irl(old_state, action)
                s, direc= self.reset()  # next traj
                info = [0, direc]





        return np.array(self.state),reward,self.done,info





    def reward_irl(self,old,action):

        old.append(action)
        reward = np.array(old).dot(
            np.array([0.02368745,  0.35580747 , 0.6752002  , 0.63197829 ,-0.13246919]))  *action*(10**-1)


        return reward





    def Speed_cau(self,speed):  # Calculate speed status

        status = round(speed/10)
        return status

    def cos_sim(self,a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a, b) / (a_norm * b_norm)
        #print(cos)
        return cos

    def seed(self,seed):
        self.seed=seed

    def RoadComplexity_cau(self,longitude,latitude):

        hashcode = geohash.encode(latitude, longitude, precision=7)  # decode about 153*153m
        #print(self.directory[self.directory_num],latitude,longitude)
        if(hashcode in self.geohash_road):
            num=self.geohash_road[hashcode]               #find
        else:
            num=1
        status=int (num/3)+1
        return status

    def Noise_cau(self,longitude,latitude):
        keys=[]
        num=0
        hashcode = geohash.encode(latitude, longitude, precision=7)  # decode about 153*153m
        if(hashcode in self.geohash_noise):
            num=self.geohash_noise[hashcode]
        else:
            hashcode=hashcode[0:6]
            for elem in self.geohash_noise.keys():
                if(hashcode==elem[0:6]):
                    keys.append(self.geohash_noise[elem])
            if(not keys):
                num=0
            else:
                num=int(np.mean(keys))
        num=round(num/3)
        return num

    def Angle_cau(self,angle1,angle2):
        sub=abs(int(angle1)-int(angle2))
        sub=int(sub/30)
        return sub

    def Distance(self,a,b):

        dis=np.linalg.norm(a-b)
        return dis

    def Angle_cau2(self,point1,point2):
        x1=point1[0]
        y1=point1[1]
        x2=point2[0]
        y2=point2[1]
        angle = 0
        dx = x2 - x1
        dy = y2 - y1
        if x2 == x1:
            angle = math.pi / 2.0
            if y2 == y1:
                angle = 0.0
            elif y2 < y1:
                angle = 3.0 * math.pi / 2.0
        elif x2 > x1 and y2 > y1:
            angle = math.atan(dx / dy)
        elif x2 > x1 and y2 < y1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif x2 < x1 and y2 < y1:
            angle = math.pi + math.atan(dx / dy)
        elif x2 < x1 and y2 > y1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        angle= round(angle * 180 / math.pi)
        angle=int(angle/30)
        return angle

    def ReadNoise(self,path):
        noise=pd.read_table(path)
        dict={}
        for i in range(len(noise)):
            dict[str(noise.iloc[i,0])]=float(noise.iloc[i,1])
        return dict

    def ReadRoadComplexity(self,path):
        road=pd.read_table(path)
        dict={}
        for i in range(len(road)):
            dict[str(road.iloc[i,0])]=float(road.iloc[i,1])
        return dict

    def ReadSpeed(self,dir):
        speeds=pd.read_table(r'F:\global-map-matching-dataset'+'/'+dir+'/speed.txt',header=None)
        return speeds.values.tolist()








