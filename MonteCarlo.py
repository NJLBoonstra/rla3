#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax
from Q_learning import QLearningAgent


class MonteCarloAgent(QLearningAgent):
    def update(self, s, a, r, s_next, done):
        G = [0 for _ in range(len(s) + 1)]
        for i in range(len(s) - 1, 0):
            G[i] = r[i] + self.gamma * G[i+1]
            self.Q_sa[s[i], a[i]] += (self.learning_rate
                                      * (G[i] - self.Q_sa[s[i], a[i]]))

    def learn(self, budget, plot=True):
        timesteps, max_ep_len = budget
        rewards = []
        ts = 0

        while True:
            if ts >= timesteps:
                break
    
            s = self.env.reset()

            s_t = []
            a_t = []
            r_t = []
            for _ in range(max_ep_len):
                if ts >= timesteps:
                    break

                a = self.select_action(s)
                s_, r, done = self.env.step(s)
                s_t.append(s_)
                a_t.append(a)
                r_t.append(r)
                ts += 1
                s = s_
                if done:
                    break

            self.update(s_t, a_t, r_t, None, None)
            # Add episode rewards to timestep rewards
            rewards += r_t
        return rewards


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''
    eps_or_temp = None
    if policy == "egreedy":
        eps_or_temp = epsilon
    elif policy == "softmax":
        eps_or_temp = temp

    mca = MonteCarloAgent(policy, learning_rate, gamma, eps_or_temp)

    return mca.learn((n_timesteps, max_episode_length), plot)


def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                          policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
