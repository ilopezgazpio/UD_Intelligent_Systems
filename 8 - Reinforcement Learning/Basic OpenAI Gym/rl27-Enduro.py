#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Important to note:
In previous implementations (like Pong) we fed our Neural Network with a single image from video output.
Here we use 4 stacked images, which then serve as input for our NN.
This approach was briefly mentioned in DQN Potential Improvements video.

We also did some experiments (comparing to Pong implementation) with atari_wrappers.py (updated file also attached) and preprocess_frame method to make it more resources efficient.
More details on http://ai.atamai.biz/post/stackedimages/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time
import os.path
import numpy as np

import matplotlib.pyplot as plt
from atari_wrappers import make_atari, wrap_deepmind

plt.style.use('ggplot')

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

# Environment
env_id = "EnduroNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env,frame_stack=True, pytorch_img=True)
seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

# Output
directory = './EnduroVideos/'
env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id%5==0,force=True)

# Hyperparameters
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

learning_rate = 0.00025
num_episodes = 200
gamma = 0.99
hidden_layer = 512

replay_mem_size = 100000
batch_size = 32
update_target_frequency = 5000
double_dqn = True
egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 15000
report_interval = 10
score_to_solve = 310
clip_error = True
normalize_image = True
file2save = 'enduro_save.pth'
save_model_frequency = 10000
resume_previous_training = False


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * math.exp(-1. * steps_done / egreedy_decay )
    return epsilon


def load_model():
    return torch.load(file2save)


def save_model(model):
    torch.save(model.state_dict(), file2save)


def preprocess_frame(frame):
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame


def plot_results():
    plt.figure(figsize=(12,5))
    plt.title("Rewards")
    plt.plot(rewards_total, alpha=0.6, color='red')
    plt.savefig("Enduro-results.png")
    plt.close()


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
        return zip(*random.sample(self.memory, batch_size))


    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.advantage1 = nn.Linear(7*7*64,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)

        self.value1 = nn.Linear(7*7*64,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)

        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()


    def forward(self, x):

        if normalize_image:
            x = x / 255

        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)

        output_conv = output_conv.view(output_conv.size(0), -1) # flatten

        output_advantage = self.advantage1(output_conv)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)

        output_value = self.value1(output_conv)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)

        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final


class QNet_Agent(object):

    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=self.nn.parameters(), lr=learning_rate)

        self.number_of_frames = 0

        if resume_previous_training and os.path.exists(file2save):
            print("Loading previously saved model ... ")
            self.nn.load_state_dict(load_model())


    def select_action(self,state,epsilon):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:

            with torch.no_grad():

                state = preprocess_frame(state)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else:
            action = env.action_space.sample()

        return action


    def optimize(self):

        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = [ preprocess_frame(frame) for frame in state ]
        state = torch.cat(state)
        new_state = [ preprocess_frame(frame) for frame in new_state ]
        new_state = torch.cat(new_state)

        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]

            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]


        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)

        self.optimizer.step()

        if self.number_of_frames % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        if self.number_of_frames % save_model_frequency == 0:
            save_model(self.nn)

        self.number_of_frames += 1


''' Play the environment'''
memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

rewards_total = []

frames_total = 0
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):

    state = env.reset()
    score = 0

    while True:

        frames_total += 1
        epsilon = calculate_epsilon(frames_total)

        action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done, info = env.step(action)

        # env.render()

        score += reward

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()

        state = new_state

        if done:
            rewards_total.append(score)

            mean_reward_100 = sum(rewards_total[-100:])/100

            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After {} episodes ".format(i_episode))
                solved_after = i_episode
                solved = True

            if (i_episode % report_interval == 0 and i_episode > 0):
                plot_results()

                print("\n*** Episode {} *** \
                       \nAv.reward: [last {}]: {}, [last 100]: {}, [all]: {} \
                       \nepsilon: {}, frames_total: {}".format(i_episode, report_interval, sum(rewards_total[-report_interval:])/report_interval, mean_reward_100, sum(rewards_total)/len(rewards_total), epsilon, frames_total))

                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            break

print("\n\n\n\nAverage reward: {}".format(sum(rewards_total)/num_episodes))
print("Average reward (last 100 episodes): {}".format(sum(rewards_total[-100:])/100))

if solved:
    print("Solved after {} episodes".format(solved_after))

env.close()
env.env.close()
