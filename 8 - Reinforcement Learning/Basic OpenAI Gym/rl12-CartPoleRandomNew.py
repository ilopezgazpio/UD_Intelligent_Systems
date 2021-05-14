#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import torch
import random
import matplotlib.pyplot as plt

'''Environment'''
env = gym.make('CartPole-v0')

'''Parameters'''
seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
num_episodes = 1000
steps_total = []


for i_episode in range(num_episodes):
    state = env.reset()
    step = 0

    while True:
        step += 1
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        
        #print(new_state)
        #print(info)
        #env.render()
        
        if done:
            steps_total.append(step)
            print("Episode finished after {} steps".format(step))
            break

print("Average reward: {}".format(sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): {}".format(sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

env.close()
env.env.close()
