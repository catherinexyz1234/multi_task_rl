# Created with Python AI

"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from env import Reacher
from copy import copy
ITR=40  # number of tasks
EP_MAX = 200  # for single task
EP_LEN = 20
GAMMA = 0.9
A_LR = 1e-4
C_LR = 2e-4
BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM = 10,10,2
A_DIM = 8
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][0]        # choose the method for optimization


class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
        self.critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

        # actor
        self.pi, self.pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, oldpi_params)]


        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = self.pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, self.pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
            # add entropy to boost exploration
            # entropy=pi.entropy()
            # self.aloss-=0.1*entropy
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.actor_var = [v for v in tf.trainable_variables() if v.name.split('/')[0] == "pi"]
        self.critic_var= [v for v in tf.trainable_variables() if v.name.split('/')[0] == "critic"]
        
        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for i in range(A_UPDATE_STEPS):
                # print('updata: ',i)
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        # print([v.name for v in self.critic_var],self.sess.run(self.critic_var))
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            # l1 = tf.layers.batch_normalization(tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable), training=True)
            '''the action mean mu is set to be scale 10 instead of 360, avoiding useless shaking and one-step to goal!'''
            mu =  10.*tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable) # softplus to make it positive
            # in case that sigma is 0
            sigma +=1e-4
            self.mu=mu
            self.sigma=sigma
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a ,mu, sigma= self.sess.run([self.sample_op, self.mu, self.sigma], {self.tfs: s})
        # print('s: ',s)
        # print('a: ', a)
        # print('mu, sigma: ', mu,sigma)
        return np.clip(a[0], -360, 360)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    def save_model(self, ):
        self.actor_weights=self.sess.run(self.actor_var)
        self.critic_weights=self.sess.run(self.critic_var)
        return self.actor_weights, self.critic_weights
    # def value(self, ):
    #     return self.sess.run(self.actor_weights_before)[-1]
    def meta_update(self, stepsize, a_be, a_af, c_be, c_af,weight):
        ''' actor meta update '''
        new_actor_weights=stepsize*np.array(a_af)+(1-stepsize)*np.array(a_be)
        
        with tf.variable_scope('pi'):
            self.meta_update_pi = [pi.assign(new) for pi, new in zip(self.pi_params, new_actor_weights)]
        # print(new_actor_weights[-1])
        # print(self.sess.run(self.meta_update_pi)[-1])

        self.sess.run(self.update_oldpi_op)

        ''' critic meta update '''
        new_critic_weights=stepsize*weight*np.array(c_af)+(1-stepsize)*np.array(c_be)
        with tf.variable_scope('critic'):
            self.meta_update_critic = [cri.assign(new) for cri, new in zip(self.critic_params, new_critic_weights)]
        # print(new_critic_weights[-1])
        # print(self.sess.run(self.meta_update_critic)[-1])
        # print(self.sess.run(self.critic_var[-1]))
    
    def restore_model(self, a_te, c_te):
        # restore the actor
        with tf.variable_scope('pi'):
            self.restore_pi = [pi.assign(te) for pi, te in zip(self.pi_params, np.array(a_te))]
        self.sess.run(self.restore_pi)

        self.sess.run(self.update_oldpi_op)

        # restore the critic

        with tf.variable_scope('critic'):
            self.restore_critic =[cri.assign(te) for cri, te in zip(self.critic_params, np.array(c_te))]
        self.sess.run(self.restore_critic)



    


def sample_task():
    range_pose=0.3
    target_pose=np.random.rand(2)*range_pose + [0.5, 0.5]
    screen_size=1000
    target_pose=target_pose*screen_size

    
    env=Reacher(target_pos=target_pose, render=True)
    return env, target_pose



# env=Reacher(render=True)
ppo = PPO()
stepsize0=0.1
test_env, t=sample_task()
itr_test_rewards=[]
for itr in range (ITR):
    env, target_position =sample_task()
    print('Task {}: target position {}'.format(itr+1, target_position))
    actor_weights_before, critic_weights_before= ppo.save_model()
    all_ep_r = []
    for ep in range(EP_MAX):
        # print(actor_weights_before[-1])
        s = env.reset()
        s=s/100. # scale the inputs
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            # env.render()
            a = ppo.choose_action(s)
            s_, r, done = env.step(a)
            s_=s_/100.
            buffer_s.append(s)
            buffer_a.append(a)
            # print('r, norm_r: ', r, (r+8)/8)
            '''the normalization makes reacher's reward almost same and not work'''
            # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            buffer_r.append(r)
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )
        # if ep % 500==0:
        #     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        #     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_reptile.png')
    
    
    actor_weights_after, critic_weights_after= ppo.save_model()
    # print(actor_weights_after[-1])
    stepsize=stepsize0*(1-itr/ITR)
    '''
    w' = w_before + (w_after-w_before) * stepsize ->
    w' = stepsize * w_after + (1-stepsize) * w_before
    '''
    ppo.meta_update(stepsize, actor_weights_before, actor_weights_after, critic_weights_before, critic_weights_after)



    # test 1 episode on test_env
    actor_weights_test, critic_weights_test= ppo.save_model()
    all_ep_r = []
    print('-------------- TEST --------------- ')
    for ep in range(EP_MAX):
        s = env.reset()
        s=s/100. # scale the inputs
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            a = ppo.choose_action(s)
            s_, r, done = env.step(a)
            s_=s_/100.
            buffer_s.append(s)
            buffer_a.append(a)
            # print('r, norm_r: ', r, (r+8)/8)
            '''the normalization makes reacher's reward almost same and not work'''
            # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            buffer_r.append(r)
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
        all_ep_r.append(ep_r)
        # if ep == 0: all_ep_r.append(ep_r)
        # else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        # print(
        #     'Ep: %i' % ep,
        #     "|Ep_r: %i" % ep_r,
        #     ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        # )
        # if ep % 500==0:
        #     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        #     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.savefig('./ppo_reptile.png')
    # restore before test
    ppo.restore_model(actor_weights_test, critic_weights_test)
    itr_test_rewards.append(np.average(np.array(all_ep_r)))
    plt.plot(np.arange(len(itr_test_rewards)), itr_test_rewards)
    plt.savefig('./ppo_reptile.png')
