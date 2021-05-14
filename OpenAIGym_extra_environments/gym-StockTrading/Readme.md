#gym-StockTrading

Openai-gym environment for Stock Trading.

## Instalation
```
pip3 install -e gym-StockTrading
```
or
```
cd gym-StockTrading
pip3 install -e .

```

## Usage
create an instance of the environment with 
```
import gym
env = gym.make('gym_StockTrading:StockTrading-v0')
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