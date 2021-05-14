#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import torch
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

''' Environmnet parameters '''
number_of_states = env.observation_space.n
number_of_actions =  env.action_space.n

''' Value iteration parameters - offline learning '''
# V values table - Value for each state ( Value function V(S)), collects value for each of the 16 states
V = torch.zeros( [number_of_states] )

''' HYPERPARAMETERS '''
gamma = 0.9

rewards_total = []
steps_total = []

''' Function to return best possible move out of total possible moves given a state and the V values'''
def best_action(state, V):
    actions_V = torch.zeros(number_of_actions)

    for action_possible in range(number_of_actions):

        # Explore hypothetical action
        for prob, new_state, reward, _ in env.env.P[state][action_possible]:
            actions_V[action_possible] += (prob * (reward + gamma * V[new_state]) )
    
    max_value, index = torch.max(actions_V, 0)
    
    return max_value, index
    


''' Function to build V values from scratch visiting all possible states '''
def value_iteration():
    V = torch.zeros( number_of_states )
    # this is value based on experiments, after that many iterations values don't change significantly any more

    max_iterations = 1500
    for _ in range(max_iterations):

        for state in range(number_of_states):
            max_value, _ = best_action(state, V)
            V[state] = max_value.item()

    return V



''' Function to build policy out of the V table'''
def build_policy(V):
    policy = torch.zeros(number_of_states)

    for state in range(number_of_states):
        _ , index = best_action(state, V)
        policy[state] = index.item()
    
    return policy


# RUN VALUE-ITERATION TRAINING ALGORITHM (Off-Line)
V = value_iteration()
policy = build_policy(V)
num_episodes = 1000

for i_episode in range(num_episodes):
    state = env.reset()    
    step = 0
    
    while True:
        step += 1     
        action = policy[state]
        new_state, reward, done, info = env.step(action.item())        
        state = new_state

        if done:
            rewards_total.append(reward)
            steps_total.append(step)
            break

print(V)


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
