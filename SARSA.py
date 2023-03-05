#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Q_learning import QLearningAgent
from Helper import softmax, argmax


class SarsaAgent(QLearningAgent):
    def update(self, s, a, r, s_next, a_next):
        G = r + self.gamma * self.Q_sa[s_next, a_next]
        self.Q_sa[s, a] = (self.Q_sa[s, a]
                           + self.learning_rate * (G - self.Q_sa[s, a]))

    def learn(self, budget, plot=True):
        if plot:
            print("SarsaAgent.learn:")
            print(f"\tpolicy = {self.policy}, value = {self.eps_or_temp}")
        s = self.env.reset()
        a = self.select_action(s)

        rewards = []
        episode = 1
        episode_reward = 0

        for _ in range(budget):
            s_, r, done = self.env.step(a)
            a_ = self.select_action(s_)

            self.update(s, a, r, s_, a_)

            rewards.append(r)
            episode_reward += r
            if done:
                s = self.env.reset()
                a = self.select_action(s)

                if plot:
                    print(f"Episode {episode} reward = {episode_reward}")

                episode += 1
                episode_reward = 0
                continue

            s = s_
            a = a_

        return rewards


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy',
          epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep '''
    eps_or_temp = None
    if policy == "egreedy":
        eps_or_temp = epsilon
    elif policy == "softmax":
        eps_or_temp = temp

    sla = SarsaAgent(policy, learning_rate, gamma, eps_or_temp)

    return sla.learn(n_timesteps, plot)


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma,
                    policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
