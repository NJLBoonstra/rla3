#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''
        return argmax(self.Q_sa[s,])

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # Calculate the vector we are going to sum
        values = [p_sas[i] * (r_sas[i] + (self.gamma * np.max(self.Q_sa[i, ])))
                  for i in range(len(p_sas))]
        v = np.sum(values)

        self.Q_sa[s, a] = v

        return v


def Q_value_iteration(env: StochasticWindyGridworld, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    i = 0
    while True:
        delta = 0

        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                x = QIagent.Q_sa[s, a]

                p_ssa, r_sas = env.model(s, a)
                q_sa = QIagent.update(s, a, p_ssa, r_sas)

                delta = max(delta, np.abs(x - q_sa))

        if delta < threshold:
            break

        i += 1
        # Plot current Q-value estimates & print max error
        # env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(i, delta))

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # View optimal policy
    done = False
    s = env.reset()
    # r = reward_per_step
    reward_per_step = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        reward_per_step.append(r)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        s = s_next

    mean_reward_per_timestep = np.mean(reward_per_step)
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))


if __name__ == '__main__':
    experiment()
