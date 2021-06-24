from re import I
import time
import yaml
from argparse import Namespace
import numpy as np
import gym

class ODGPF:
    def __init__(self):
        self.MU = 1.0489
        self.GRAVITY = 9.806
        self.MAX_RANGE = 30.0
        self.GAMMA = 1.0
        
        self.CAR_WIDTH = 1.0
        self.CAR_MASS = 3.463388126201571
        
        self.SCAN_INTERVAL = 0.00435185185
        self.OBS_THRESHOLD = 5.0
        self.DETECTION_START = 179
        self.DETECTION_END = 899

    def idx2deg(self,idx):
        _res = (idx-540) * self.SCAN_INTERVAL
        return _res
    
    def deg2idx(self,ang):
        _res = (ang/self.SCAN_INTERVAL) + 540
        return _res

    def define_obstacles(self, scan):
        obstacles = []
        flag = False
        start_idx = 0
        end_idx = 0
        obstacle_count = 0
        
        for i in range(self.DETECTION_START,self.DETECTION_END+1):
            if scan[i] <= self.OBS_THRESHOLD and flag == False:
                flag = True
                start_idx = i
            elif flag == True:
                if scan[i+1] > self.OBS_THRESHOLD or i+1 > self.DETECTION_END:
                    flag = False
                    obstacle_count += 1
                    end_idx = i

                    avg_distance = np.average(scan[start_idx:end_idx])
                    avg_angle = self.idx2deg(end_idx - start_idx)
                    center_angle = self.idx2deg((end_idx + start_idx)//2)

                    _a = avg_distance * np.tan(avg_angle/2) + self.CAR_WIDTH/2
                    _dist = self.MAX_RANGE - avg_distance
                    _A_k = _dist * np.exp(0.5)
                    phi_k = 2*np.arctan2(_a, avg_distance)
                    
                    obstacles.append([start_idx, end_idx, center_angle, phi_k, _A_k])
        
        return obstacles

    def att_field(self,goal):
        # Calculate for Attraction Field
        att_field_list = np.zeros(1080)  # FOR SIMULATOR should change some
        goal = (-540 + goal) * self.SCAN_INTERVAL
        for i in range(len(att_field_list)):
            idx2deg = (-540 + i ) * self.SCAN_INTERVAL
            att_field_list[i] = self.GAMMA * np.fabs((goal - idx2deg))

        return att_field_list

    def rep_field(self, obstacles):
        # Calculate for Repulsive Field
        # Obstacles Array = [start_idx, end_idx, center_angle, phi_k, _A_k]
        rep_field_list = np.zeros(1080)
        for i in range(len(rep_field_list)):
            for j in range(len(obstacles)):
                temp1 = (obstacles[j][2] - self.idx2deg(i))**2
                temp2 = (obstacles[j][3]) ** 2
                rep_field_list[i] += obstacles[j][4] * np.exp(-0.5 * (temp1 / temp2))
        
        return rep_field_list

    def total_field(self, att_field_list, rep_field_list):
        return att_field_list + rep_field_list
    
    def planner(self, obs):
        goal = [0,9]
        self.obs = obs
        scan = obs['scans'][0]
        
        goal_theta = self.get_Goal(goal)
        
        obstacles = self.define_obstacles(scan)

        rep_list = self.rep_field(obstacles)
        att_list = self.att_field(goal_theta)
        total_list = self.total_field(att_list, rep_list)
        set_heading = self.idx2deg(np.argmin(total_list))
        speed = self.set_speed()
        
        return [set_heading,speed]
    
    def set_speed(self):
        max_speed = self.set_maxspeed()
        current_speed = self.obs['linear_vels_x'][0]

        if current_speed < max_speed:
            current_speed += 1.0
            return current_speed
        elif current_speed == max_speed:
            return current_speed
        else:
            current_speed = current_speed * 0.9
            return current_speed


    def get_Goal(self, goal_cord):
        current_x = self.obs['poses_x'][0]
        current_y = self.obs['poses_y'][0]
        
        target_x = goal_cord[0]
        target_y = goal_cord[1]

        dx = target_x - current_x
        dy = target_y - current_y
        degree = np.arctan2(dy,dx)

        return degree
    
    def set_maxspeed(self):
        front_distance = self.obs['scans'][0][539]

        maximum_speed = np.sqrt(2*self.MU*self.GRAVITY*front_distance)
        return maximum_speed
        

if __name__ =='__main__':
    
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()
    odg_pf = ODGPF()
    
    laptime = 0.0
    start = time.time()
    goal_pos = [0,9]
    while not done:
        speed, steer = odg_pf.planner(obs)
        # speed, steer = [1.0, 0]
        print("X : ", obs['poses_x'][0])
        print("Y : ", obs['poses_y'][0])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
