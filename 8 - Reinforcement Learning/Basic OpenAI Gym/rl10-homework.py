#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import time
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

''' Environment '''
env = gym.make('Taxi-v3')
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

''' Agent hyperparameters '''
gamma = 0.98
learning_rate = 0.85
egreedy = 0.05

''' Q table '''
Q = torch.zeros( [number_of_states, number_of_actions] )

''' Training parameters '''
num_episodes = 5000
steps_total = []
rewards_total = []

for i_episode in range(num_episodes):  
    state = env.reset()
    step = 0
    score = 0

    while True:       
        step += 1

        random_for_egreedy = torch.rand(1)[0]

        # Observe Q table and take best action
        if random_for_egreedy > egreedy:      
            pr_actions = Q[state] + torch.rand(1, number_of_actions) / 1000      
            action = torch.max(pr_actions,1)[1].item()
            
        # Epsilon greedy  based random action
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)        
        score += reward

        # Bellman equation for stochastic scenarios (Q-learning based on temporal difference)
        memory_unit = Q[state, action]
        input_gate = reward + gamma * torch.max(Q[new_state])        
        Q[state, action] = (1 - learning_rate) * memory_unit  + learning_rate * input_gate
        
        state = new_state
        
        #time.sleep(0.4)
        #env.render()
        #print(new_state)
        #print(info)
        
        if done:
            steps_total.append(step)
            rewards_total.append(score)
            print("Episode finished after {} steps".format(step))
            print(score)
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
plt.bar(torch.arange(len(steps_total[200:])), steps_total[200:], alpha=0.6, color='red', width=5)
plt.show()

