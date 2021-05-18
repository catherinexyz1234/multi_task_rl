# Created with Python AI
import numpy as np
from mazelab.generators import random_maze
import natsort
import cv2
import gym
import test
import glob
import ppo_reptile
from PIL import Image
from mazelab.solvers import dijkstra_solver
from keras.preprocessing import image
from mazelab.solvers import dijkstra_solver
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Conv2DTranspose, Reshape, Dropout, concatenate, Concatenate, multiply, add, MaxPooling2D, Lambda, Activation, subtract, Flatten, Dense
from keras.applications. resnet50 import ResNet50
import tensorflow as tf
from keras.models import Model
crop_shape = (256, 256,3)
def build_resnet50(image_input):
    image_input=Input(shape=image_input)
    with tf.device('/cpu:0'):
        resnet_model = ResNet50(input_tensor=image_input,include_top=False, weights='imagenet')
        #resnet_model.summary()
    x = resnet_model.get_layer('conv2_block2_1_conv').output
    #x = Conv2D(16, (3,3), padding="same", activation="relu")(x)
    resnet_model.trainable = False
    with tf.device('/cpu:0'):
        model=Model(inputs=resnet_model.input, outputs=x)
    return model

#use GMM as clustering method
resnet50 = build_resnet50(crop_shape)
resnet50.trainable = False
model = GaussianMixture(n_components=4)
start_idx = [[0, 0]]
goal_idx = [[49, 49]]
env_id = 'RandomMaze-v'
gfg = np.random.uniform(0, 1, 100)
X=[]
for i in range(400):
    x = random_maze(width=50, height=50, complexity=.75, density=gfg[i])
    X.append(x)
    gym.envs.register(id=env_id+str(i), entry_point=test.Env, max_episode_steps=500,kwargs={'x': True,'start_idx':True,'goal_idx':True} )
    env = gym.make(env_id+str(i),x=x,start_idx=start_idx, goal_idx=goal_idx)
    env.reset()
    img = env.render('rgb_array')
    cv2.imwrite("env/"+str(i)+".jpg",img)
X=np.asarray(X)
nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))
model.fit(X)
yhat = model.predict(X)
# retrieve unique clusters
clusters = np.unique(yhat)
print(clusters)
t1=[]
t2=[]
t3=[]
t4=[]
tt=[t1,t2,t3,t4]
centers = np.empty(shape=(model.n_components, X.shape[1]))
for i in range(model.n_components):
    density = scipy.stats.multivariate_normal(cov=modelcovariances_[i], mean=model.means_[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]

s1=0
s2=0
s3=0
s4=0
ss=[s1,s2,s3,s4]
for i in range(400):
  temp=centers[yhat[i], :]
  ss[yhat[i]]=ss[yhat[i]]+np.linalg_norm(temp-X[i])
  tt[yhat[i]].append((X[i],i,np.linalg_norm(temp-X[i])))
  


ppo = PPO()

#env_id = 'RandomMaze1-v'
stepsize0=0.1
tasks=[]

for i in range(1):
    for task in range(4):
        stepsize0=0.1
        D_sub=[]
        tasks.append(ppo_reptile.PPO())
        gfg1 = np.random.uniform(0, (1/4)*task, 100)
        for j in range(len(tt[task])):
            x=tt[task][j][0]
            #x = random_maze(width=50, height=50, complexity=.75, density=gfg1[j])
            #gym.envs.register(id=env_id+str(tt[task][j][1]), entry_point=test.Env, max_episode_steps=500,kwargs={'x': True,'start_idx':True,'goal_idx':True} )
            env = gym.make(env_id+str(tt[task][j][1]),x=x,start_idx=start_idx, goal_idx=goal_idx)
            env.reset()
            #sample state trajectory
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r=0
            w=tt[task][j][2]/ss[task]
            while True:
                impassable_array = env.unwrapped.maze.to_impassable()
                motions = env.unwrapped.motions
                start = env.unwrapped.maze.objects.agent.positions[0]
                goal = env.unwrapped.maze.objects.goal.positions[0]
                actions = dijkstra_solver(impassable_array, motions, start, goal)
                buffer_a.append(actions[0])
                
                a = tasks[task].choose_action(s)
                s_, r, r1, done = env.step(a,actions[0])
                buffer_s.append(s)
                s = s_
                ep_r += r1
                buffer_r.append(r1)
                
                if done:
                  v_s_ = ppo.get_v(s_)
                  discounted_r = []
                  for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                   
                  discounted_r.reverse()
                  bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                  buffer_s, buffer_a, buffer_r = [], [], []
                  tasks[task].update(bs, ba, br)
                
                
                if ep == 0: all_ep_r.append(ep_r)
                else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
                print(
                    'Ep: %i' % ep,
                    "|Ep_r: %i" % ep_r,
                    ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
                )
          
            actor_weights_after, critic_weights_after= tasks[task].save_model()
            # print(actor_weights_after[-1])
            stepsize=stepsize0*(1-j/100)
            tasks[task].meta_update(stepsize, actor_weights_before, actor_weights_after, critic_weights_before, critic_weights_after,w)
          
## tasks are our targeted array containing all params we need
