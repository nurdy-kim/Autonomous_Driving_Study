import gym
import time
import yaml
from argparse import Namespace

import numpy as np

class PurePursuit:
    def __init__(self,conf,wb):
        # import wp
        self.waypoints = self.load_waypoints(conf)
        
        # const for wp
        self.current_wp = 0
        self.desire_wp = 0
        self.current_pos = [0,0,0]
        self.nearest_distance = 0
        
        # const for cars
        self.wb = wb
        self.Ld = 2.0
        self.MU = 1.0489
        self.MASS = 3.74

        # current state of cars
        self.current_velocity = 5.0
        self.current_steering = 0

        # const for calc
        self.PI = 3.141592
        self.GRAVITY_ACC = 9.81
        
    
    def load_waypoints(self,conf):
        file_wps = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        wps = np.vstack((file_wps[:,conf.wpt_xind], file_wps[:,conf.wpt_yind])).T

        return wps
    
    def get_distance(self, origin, target):
        _dx = origin[0] - target[0]
        _dy = origin[1] - target[1]

        return np.sqrt(_dx**2 + _dy**2)
    
    def find_current_pos(self):
        _diffs = []
        
        for i in range(self.waypoints.shape[0]):
            _diffs.append(self.get_distance(self.current_pos[0:2],self.waypoints[i,:]))
        
        _ld_diffs = []
        for i in range(self.waypoints.shape[0]):
            _ld_diffs.append(np.abs(_diffs[i] - self.Ld))
        
        # goal = np.argmin(_ld_diffs)
        print("Ld :", np.argmin(_ld_diffs), "curr :", np.argmin(_diffs))

        self.current_wp = np.argmin(_diffs)
        self.desire_wp = np.argmin(_ld_diffs)

        
    def planner(self, obs):
        self.current_pos = [obs['poses_x'][0],obs['poses_y'][0],obs['poses_theta'][0]]
        self.find_current_pos()
        # print("steering : ", steering, "velocity : ", velocity)
        # print(self.desire_wp)
        
        return [1.0,0.0]



# Main
work = {'mass':3.463388126201571, 'lf':0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
with open('config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

env = gym.make('f110_gym:f110-v0', map= conf.map_path, map_ext=conf.map_ext,num_agents=1)
obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
env.render()

pp = PurePursuit(conf, 0.17145+0.15875)

laptime = 0.0 
start = time.time()

while not done:
    speed, steer = pp.planner(obs)

    obs, step_reward, done, info = env.step(np.array([[steer,speed]]))
    laptime += step_reward
    env.render(mode='human')

print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)