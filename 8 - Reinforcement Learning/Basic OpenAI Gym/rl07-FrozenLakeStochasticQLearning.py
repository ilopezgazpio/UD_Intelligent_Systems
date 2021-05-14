#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import torch
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

plt.style.use('ggplot')

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

''' Hyperparameters '''
gamma = 0.95 # regulates power for future rewards
learning_rate = 0.9 # regulates power for fresh information gate

''' Q table '''
Q = torch.zeros( [number_of_states, number_of_actions] )

num_episodes = 1000
steps_total = []
rewards_total = []

for i_episode in range(num_episodes):
    state = env.reset()
    step = 0

    while True:
        step += 1
        random_values = Q[state] + torch.rand(1, number_of_actions) / 1000

        ''' MAX Q(S', A') '''
        action = torch.max(random_values,1)[1][0]

        ''' Do actual move '''
        new_state, reward, done, info = env.step(action.item())

        ''' Current memory cell (left part of Q-learning equation) '''
        current_memory = Q[state, action]

        ''' Input gate cell (right part of Q-learning equation) '''
        input_gate = reward + gamma * torch.max(Q[new_state])

        ''' Bellman equation for stochastic environment '''        
        Q[state, action] = (1 - learning_rate) * current_memory + learning_rate * input_gate

        state = new_state

        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            print("Episode finished after {} steps".format(step))
            break
        
print(Q)
        
print("Percent of episodes finished successfully: {}".format(sum(rewards_total)/num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {}".format(sum(rewards_total[-100:])/100))

print("Average number of steps: {}".format(sum(steps_total)/num_episodes))
print("Average number of steps (last 100 episodes): {}".format(sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
plt.show()
