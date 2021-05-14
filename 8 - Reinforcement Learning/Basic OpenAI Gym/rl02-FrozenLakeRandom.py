#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import gym
import torch
import matplotlib.pyplot as plt

num_episodes = 1000

env = gym.make('FrozenLake-v0')
steps_total = []
rewards_total = []

for i_episode in range(num_episodes):
    state = env.reset()
    step = 0

    while True:
        step += 1
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        
        #print(new_state)
        #print(info)
        time.sleep(0.4)
        env.render()
        
        if done:
            steps_total.append(step)
            rewards_total.append(reward)

            print("Episode finished after {} steps".format(step))
            break
        
print("Percent of eps finished successfully: {}".format(sum(rewards_total)/num_episodes))
print("Percent of eps finished successfully (last 100 episodes): {}".format(sum(rewards_total[-100:])/100))

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

