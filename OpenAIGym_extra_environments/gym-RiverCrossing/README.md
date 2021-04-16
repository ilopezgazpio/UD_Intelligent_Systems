#gym-RiverCrossing

Openai-gym environment for the River Crossing AI problem.

For a complete description of the environment and its rules check [River crossing puzzle](https://en.wikipedia.org/wiki/River_crossing_puzzle#:~:text=A%20river%20crossing%20puzzle%20is,may%20be%20safely%20left%20together) link on Wikipedia.

## Instalation
```
pip3 install -e gym-RiverCrossing
```
or
```
cd gym-RiverCrossing
pip3 install -e .

```

## Usage
create an instance of the environment with 
```
import gym
env = gym.make('gym_RiverCrossing:RiverCrossing-v0')
env.action_space
env.action_space.n
env.observation_space
env.observation_space.low
env.observation_space.high
env.reset()
env.render()
env.env.state

```


## Copyright
Inigo Lopez-Gazpio (inigo.lopezgazpio@deusto.es)