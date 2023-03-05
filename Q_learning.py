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
from Helper import softmax, argmax, LearningCurvePlot


class QLearningAgent:

    def __init__(self, policy: str, learning_rate: float, gamma: float,
                 eps_or_temp: float):
        # check if policy exists
        pol_fn = f"policy_{policy}"
        if hasattr(self, pol_fn) and callable(getattr(self, pol_fn)):
            self.select_action = getattr(self, pol_fn)
        else:
            raise RuntimeError(f"{policy} is not a valid policy")

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_or_temp = eps_or_temp
        self.policy = policy
        self.env = StochasticWindyGridworld(False)
        self.n_states = self.env.n_states
        self.n_actions = self.env.n_actions
        self.Q_sa = np.zeros((self.env.n_states, self.env.n_actions))
        self.rng = np.random.default_rng()

    def policy_egreedy(self, s):
        if self.rng.random() < self.eps_or_temp:
            return np.random.randint(0, self.n_actions)

        return argmax(self.Q_sa[s, ])

    def policy_softmax(self, s):
        return argmax(softmax(self.Q_sa[s, ], self.eps_or_temp))

    def update(self, s, a, r, s_next, done):
        G = r + self.gamma * np.max(self.Q_sa[s_next, ])

        q_sa = self.Q_sa[s, a] + self.learning_rate * (G - self.Q_sa[s, a])

        self.Q_sa[s, a] = q_sa

        return q_sa

    def learn(self, budget, plot=True):
        if plot:
            print("QLearningAgent.learn:")
            print(f"\tpolicy = {self.policy}, value = {self.eps_or_temp}")
        s = self.env.reset()

        rewards = []
        episode_reward = 0
        episode = 1

        for _ in range(budget):
            a = self.select_action(s)
            s_, r, done = self.env.step(a)

            self.update(s, a, r, s_, done)

            s = self.env.reset() if done else s_

            rewards.append(r)
            episode_reward += r

            if done:
                if plot:
                    print(f"Episode {episode}: reward = {episode_reward}")
                episode += 1
                episode_reward = 0

        return rewards


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    eps_or_temp = None
    if policy == "egreedy":
        eps_or_temp = epsilon
    elif policy == "softmax":
        eps_or_temp = temp

    qla = QLearningAgent(policy, learning_rate, gamma, eps_or_temp)

    return qla.learn(n_timesteps, plot)


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon,
                         temp, plot)
    # print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
