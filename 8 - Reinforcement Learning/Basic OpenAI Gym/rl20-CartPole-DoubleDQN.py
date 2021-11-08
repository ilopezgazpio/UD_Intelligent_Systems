#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time

import matplotlib.pyplot as plt

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

# Environment
seed_value = 23
env = gym.make('CartPole-v0')
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

# Hyperparameters
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

learning_rate = 0.001
num_episodes = 500
gamma = 0.9999
hidden_layer = 64
replay_mem_size = 50000
batch_size = 32
update_target_frequency = 500
double_dqn = True
egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500
report_interval = 10
score_to_solve = 195
clip_error = True


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = ( self.position + 1 ) % self.capacity


    def sample(self, batch_size):
        # Returns list of states, list of actions, list of new_states, ...
        return zip(*random.sample(self.memory, batch_size))


    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,hidden_layer)
        self.linear2 = nn.Linear(hidden_layer,number_of_outputs)

        self.activation = nn.Tanh()
        #self.activation = nn.ReLU()


    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        return output2

class QNet_Agent(object):

    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)

        self.update_target_counter = 0

    def select_action(self,state,epsilon):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:

            with torch.no_grad():

                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()
        else:
            action = env.action_space.sample()

        return action


    def optimize(self):

        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)

        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)

        if double_dqn:
            # Use learning net only to get indexes of actions (ignore the Q values)
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]
            # ...[0] gives access to values and [1] to indexes

            # Produce Q values using stable net, but instead of taking max Q value
            # we use the index of the best actions from the learning net to get best Q values
            # This is expected to reduce overestimation of Q values
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)

        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]

        #Q[state, action] = reward + gamma * torch.max(Q[new_state])
        # small trick 1 - done, as done is a vector of batch_size done values. Automatic broadcasting
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values

        # gather takes elements indexed by the attribute per rows / columns
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()

        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)

        self.optimizer.step()

        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.update_target_counter += 1


''' Play the environment'''

memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

steps_total = []
frames_total = 0
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):

    state = env.reset()
    step = 0

    while True:

        step += 1
        frames_total += 1

        epsilon = calculate_epsilon(frames_total)

        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done, info = env.step(action)

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()

        state = new_state

        if done:
            steps_total.append(step)

            mean_reward_100 = sum(steps_total[-100:])/100

            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After {} episodes ".format(i_episode))
                solved_after = i_episode
                solved = True

            if (i_episode % report_interval == 0):

                print("\n*** Episode {} *** \
                       \nAv.reward: [last {}]: {}, [last 100]: {}, [all]: {} \
                       \nepsilon: {}, frames_total: {}".format(i_episode, report_interval, sum(steps_total[-report_interval:])/report_interval, mean_reward_100, sum(steps_total)/len(steps_total), epsilon, frames_total))

                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            break

print("\n\n\n\nAverage reward: {}".format(sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): {}".format(sum(steps_total[-100:])/100))

if solved:
    print("Solved after {} episodes".format(solved_after))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

env.close()
env.env.close()
