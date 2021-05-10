#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor

env = gym.make('CartPole-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)


'''Hyperparameters'''
learning_rate = 0.01
num_episodes = 20000
gamma = 0.85
egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 500


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * math.exp(-1. * steps_done / egreedy_decay )
    return epsilon


'''Observations'''
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs, number_of_outputs)
        
    def forward(self, x):
        output = self.linear1(x)
        return output

    
class QNet_Agent(object):

    def __init__(self):
        
        '''Neural network module'''
        self.nn = NeuralNetwork().to(device)

        ''' Loss Function to train '''
        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()

        '''Optimizer'''
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
    
    
    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                # Avoid gradients this time, learning is done in optimize
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()        
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self, state, action, new_state, reward, done):       
        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor([reward]).to(device)
        
        if done:
            target_value = reward
        else:
            #NN is used to predict Q table for a given state --> Q[state, actions] 
            new_state_values = self.nn(new_state).detach() # detach is used to avoid gradients at this time, we want to train NN with state, not with next_state
            max_new_state_values = torch.max(new_state_values)
            target_value = reward + gamma * max_new_state_values
        
        predicted_value = self.nn(state)[action]
        
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



''' Play the environment '''
qnet_agent = QNet_Agent()
steps_total = []
frames_total = 0 


for i_episode in range(num_episodes):
    state = env.reset()
    step = 0

    while True:
        step += 1
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        #action = env.action_space.sample() # for random action
        action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done, info = env.step(action)
        qnet_agent.optimize(state, action, new_state, reward, done )
        
        state = new_state
        
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
