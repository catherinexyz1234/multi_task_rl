from mazelab.generators import morris_water_maze
import numpy as np
from mazelab.generators import random_maze
'''
x = random_maze(width=50, height=50, complexity=.05, density=.75)
print(x)

start_idx = [[0, 0]]
goal_idx = [[49, 49]]
env_id = 'RandomMaze-v0'
'''
import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color


class Maze(BaseMaze):
    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [self.start_idx])
        goal = Object('goal', 3, color.goal, False, [self.goal_idx])
        return free, obstacle, agent, goal

from mazelab import BaseEnv
from mazelab import VonNeumannMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete


class Env(BaseEnv):
    def __init__(self,x,start_idx,goal_idx):
        super().__init__()
        self.start_idx=start_idx
        self.goal_idx=goal_idx
        self.maze = Maze(x,start_idx,goal_idx)
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self):
        self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.goal.positions = self.goal_idx
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
