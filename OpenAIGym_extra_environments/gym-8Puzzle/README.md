#gym-8Puzzle

Openai-gym environment for the 8-puzzle AI problem.

For a complete description of the environment and its rules check [8-Puzzle problem](http://en.wikipedia.org/wiki/Fifteen_puzzle) link on Wikipedia.

## Instalation
```
pip3 install -e gym-8Puzzle
```
or
```
cd gym-8Puzzle
pip3 install -e .

```

## Usage
create an instance of the environment with 
```
import gym
env = gym.make('gym_8Puzzle:8Puzzle-v0')
env.action_space
env.action_space.n
env.action_space.sample()
env.observation_space
env.observation_space.low
env.observation_space.high
env.reset()
env.render()
env.env.state

```


## Copyright
Inigo Lopez-Gazpio (inigo.lopezgazpio@deusto.es)