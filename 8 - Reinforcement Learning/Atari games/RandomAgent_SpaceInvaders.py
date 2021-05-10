''' 
   Dependencies that must be satisfied
   
   - openAIgym ( conda install -c conda-forge gym-atari)
   - gym[atari]( conda install -c conda-forge gym-atari or pip install gym[atari] )
'''

import gym
import random


env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions_possible = env.action_space.n
env.unwrapped.get_action_meanings()


episodes = 50

for episode in range(episodes):
    
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward

        print('Episode: {} Score: {}'.format(episode, score))

env.close()









