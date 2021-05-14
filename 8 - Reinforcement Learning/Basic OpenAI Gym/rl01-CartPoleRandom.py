#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt

num_episodes = 1000

env = gym.make('CartPole-v1')

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
            print("Episode finished after %i steps" % step )
            break
                
print("Average number of steps: {}".format(sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()
