#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax
from Q_learning import QLearningAgent


class NstepQLearningAgent(QLearningAgent):
    def __init__(self, policy: str, learning_rate: float, gamma: float,
                 eps_or_temp: float, n: int):
        super().__init__(policy, learning_rate, gamma, eps_or_temp)
        self.n = n

    def update(self, s, a, r, done):
        e_len = len(a)
        for t in range(e_len):
            m = min(self.n, e_len - 1)
            tm = min(t + m, e_len - 1)
            # check if s_t+m is terminal
            if done[tm]:
                G = [(self.gamma**i) * r[t] for i in range(m)]
            else:
                G = [(self.gamma**i * r[t]
                      + self.gamma**m * np.max(self.Q_sa[s[tm]]))
                     for i in range(m)]
            G = np.sum(G)
            self.Q_sa[s[t], a[t]] += (self.learning_rate
                                      * (G - self.Q_sa[s[t], a[t]]))

    def learn(self, budget, plot=True):
        max_timesteps, episode_length = budget

        t = 0
        rewards = []
        while t < max_timesteps:
            s_t, r_t, a_t, d_t = [], [], [], []

            s = self.env.reset()
            s_t.append(s)

            cur_episode = 0

            while t < max_timesteps and cur_episode < episode_length:
                a = self.select_action(s)

                s, r, d = self.env.step(a)

                # Add everything to the register
                s_t.append(s)
                a_t.append(a)
                r_t.append(r)
                d_t.append(d)

                t += 1
                cur_episode += 1

                if d:
                    break

            self.update(s_t, a_t, r_t, d_t)
            rewards += r_t

        return rewards


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''
    eps_or_temp = None
    if policy == "egreedy":
        eps_or_temp = epsilon
    elif policy == "softmax":
        eps_or_temp = temp

    nla = NstepQLearningAgent(policy, learning_rate, gamma, eps_or_temp, n)

    return nla.learn((n_timesteps, max_episode_length), plot)


def test():
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                       policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
