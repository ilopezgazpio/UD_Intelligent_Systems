#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Generate fake data
W = 2
b = 0.3
x = torch.arange(100, dtype=torch.float).unsqueeze(1).to(device)
y = W * x + b


class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(1,1)
        
    def forward(self, x):
        output = self.linear1(x)
        return output
    

mynn = NeuralNetwork().to(device)


# Hyperparameters
learning_rate = 0.01
num_episodes = 1000
loss_func = nn.MSELoss()
#loss_func = nn.SmoothL1Loss() # Hubber Loss
optimizer = optim.Adam(params=mynn.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)

for i_episode in range(num_episodes):
    predicted_value = mynn(x)
    loss = loss_func(predicted_value, y)

    # Training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i_episode % 50 == 0:
        print("Episode {} loss {} ".format(i_episode, loss.item()))
    
    
plt.figure(figsize=(12,5))
plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), alpha=0.6, color='black')
plt.plot(x.data.cpu().numpy(), predicted_value.data.cpu().numpy(), alpha=0.6, color='red')

plt.show()

