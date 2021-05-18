from mazelab.generators import random_maze
import test
import gym

x = random_maze(width=50, height=50, complexity=.5, density=.5)
print(x)

start_idx = [[0, 0]]
goal_idx = [[49, 49]]
env_id = 'RandomMaze1-v0'

gym.envs.register(id=env_id, entry_point=test.Env, max_episode_steps=200,kwargs={'x': True,'start_idx':True,'goal_idx':True} )
import cv2

env = gym.make(env_id,x=x,start_idx=start_idx, goal_idx=goal_idx)
env.reset()
img = env.render('rgb_array')
cv2.imwrite("foo.jpg",img)
