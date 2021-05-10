#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import torch
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

env = gym.make('FrozenLakeNotSlippery-v0')
plt.style.use('ggplot')


''' Environment characteristics '''
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

''' Q table '''
Q = torch.zeros( [number_of_states, number_of_actions] )
gamma = 1

num_episodes = 1000
steps_total = []
rewards_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    step = 0

    while True:
        step += 1
        random_values = Q[state] + torch.rand(1, number_of_actions) / 1000
        action = torch.max(random_values,1)[1][0]
        new_state, reward, done, info = env.step(action.item())

        # Bellman equation
        Q[state, action] = reward + gamma * torch.max(Q[new_state])        
        state = new_state

        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            print("Episode finished after {} steps".format(step) )
            break
        
print(Q)
        
print("Percent of episodes finished successfully: {0}".format(sum(rewards_total)/num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {0}".format(sum(rewards_total[-100:])/100))

print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
print("Average number of steps (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
plt.show()
