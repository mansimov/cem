import gym
from normalized_env import NormalizedEnv
from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import tempfile
import sys
import argparse
import datetime

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--env-id', type=str, default="Reacher-v1",
                    help="Environment id")
parser.add_argument('--init-var', type=float, default=0.1,
                    help="Initial Std of parameters")
parser.add_argument('--num-ep', type=int, default=200,
                    help="Number of episodes")
parser.add_argument('--elite-frac', type=float, default=0.2,
                    help="Elite fraction")
parser.add_argument('--niter', type=int, default=250,
                    help="Number of iterations")
parser.add_argument('--extra-std', type=float, default=0.001,
                    help="Extra std")
parser.add_argument('--seed', type=int, default=1,
                    help="Random Seed")

class CEMAgent(object):
    def __init__(self, env, args):
        self.env = env
        self.config = config = args
        self.config.max_pathlength = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        self.config.std_decay_time = self.config.niter / 2
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True # don't take full gpu memory
        self.session = tf.Session(config=tf_config)
        self.end_count = 0
        self.train = True
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, 2 * env.observation_space.shape[0] + env.action_space.shape[0]], name="obs")
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.shape[0]))

        self.action_dist_n = action_dist_n = create_net(self.obs, [10,5], [True,True], env.action_space.shape[0])
        var_list = tf.trainable_variables()
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.session.run(tf.global_variables_initializer())

        th = self.gf()
        self.th_mean = np.zeros(th.shape)
        self.th_var = np.ones(th.shape) * self.config.init_var
        self.n_elite = int(np.round(self.config.elite_frac * self.config.num_ep))

    def learn(self, iteration):
        extra_var_multiplier = max((1.0-iteration/float(self.config.std_decay_time)),0)
        returns, ths = [], []
        for episode in range(self.config.num_ep):
            # sample weights first
            self.prev_obs *= 0.0
            self.prev_action *= 0.0
            obs = self.env.reset()
            sample_std = np.sqrt(self.th_var + np.square(self.config.extra_std) * extra_var_multiplier)
            th = self.th_mean + sample_std * np.random.randn(self.th_mean.shape[0])
            ths.append(th)
            self.sff(th)
            rewards = 0
            for step in range(self.config.max_pathlength):
                obs = np.expand_dims(obs, 0)
                obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)
                action = self.session.run(self.action_dist_n, {self.obs: obs_new})[0]
                self.prev_obs = obs
                self.prev_action = np.expand_dims(action, 0)
                obs, reward, terminal, _ = self.env.step(action)
                rewards += reward
                if terminal:
                    break
            returns.append(rewards)
        t2 = datetime.datetime.now()
        returns = np.array(returns)
        elite_inds = returns.argsort()[-self.n_elite:]
        elite_ths = np.array(ths)
        elite_ths = elite_ths[elite_inds, :]
        print "\n********** Iteration %i ************" % iteration
        print "Average sum of rewards per episode {}".format(returns.mean())
        self.th_mean = np.mean(elite_ths, axis=0)
        self.th_var = np.var(elite_ths, axis=0)

if __name__ == '__main__':

    args = parser.parse_args()
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    env = gym.make(args.env_id)
    env = NormalizedEnv(env, normalize_obs=True)

    agent = CEMAgent(env, args)
    for iteration in range(args.niter):
        agent.learn(iteration)
