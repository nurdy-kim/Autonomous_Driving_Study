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

        # current state of cars
        self.current_speed = 0
        self.current_steering = 0
    
    def load_waypoints(self,conf):
        file_wps = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        wps = np.vstack((file_wps[:,conf.wpt_xind], file_wps[:,conf.wpt_yind])).T

        return wps
    
    def get_distance(self, origin, target):
        _dx = origin[0] - target[0]
        _dy = origin[1] - target[1]

        return np.sqrt(_dx**2 + _dy**2)

    def find_nearest_wp(self):
        wp_idx = self.current_wp
        self.nearest_distance = self.get_distance(self.current_pos, self.waypoints[wp_idx])
        
        while True:
            wp_idx += 1
            if wp_idx >= len(self.waypoints) - 1:
                wp_idx = 0
            
            temp_dist = self.get_distance(self.current_pos, self.waypoints[wp_idx])

            if temp_dist < self.nearest_distance:
                self.nearest_distance = temp_dist
                self.current_wp = wp_idx
            elif temp_dist > (self.nearest_distance) or (wp_idx == self.current_wp):
                break
    def find_lookahead_wp(self):
        wp_idx = self.current_wp
        while True:
            if wp_idx >= len(self.waypoints) - 1:
                wp_idx = 0
            
            distance = self.get_distance(self.current_pos, self.waypoints[wp_idx])

            if distance >= self.Ld:
                break
            
            wp_idx += 1
        
        self.desire_wp = wp_idx
    def steering_controller(self):
        _dx = self.current_pos[0]  - self.waypoints[self.desire_wp][0]
        _dy = self.current_pos[1]  - self.waypoints[self.desire_wp][1]
        _I = np.sqrt(_dx**2 + _dy**2)

        _k = (2*_dx) / (_I**2)
        steering = np.arctan(_k*self.wb)

        return steering
    
    # def speed_controller(self, obs):
    #     front = obs['scans'][0][359]
    def planner(self, obs):
        self.current_pos = [obs['poses_x'][0],obs['poses_y'][0],obs['poses_theta'][0]]
        self.find_nearest_wp()

        self.find_lookahead_wp()
        steering = self.steering_controller()
        
        return [1.0,steering]


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