from abc import ABC, abstractmethod
from gym import spaces
from copy import deepcopy
from ..frontier.Node import Node

class Agent(ABC):

    ''' notes
     No argument must be mandatory
     Receive in constructor:
     initial observation
     Reference of environment
    '''


    # iteration count limiter

    # History logging per iteration and plotting

    def __heuristic__(self, node: Node):
        ''' Function to get the utility value of a node '''
        return 0

    def __expand_node__(self, node: Node):
        '''
        action_space must be an openai gym Space accepted by environment.step(action) for every action in action_space
        Note that not all openai gym environments implement the __is_applicable__ internal method.
        '''

        action_space = node.env.action_space
        assert isinstance(action_space, spaces.Space)

        expanded = list()

        if isinstance(action_space, spaces.Discrete):
            for action in range(action_space.n):
                if node.env.env.__is_applicable__(action):
                    env = deepcopy(node.env)
                    observation, reward, done, info = env.step(action)
                    new_node = Node( env,
                                     observation,
                                     reward,
                                     done,
                                     info,
                                     self.__heuristic__(env),
                                     node.cost + reward,
                                     node.action_history + [action],
                                     node.depth + 1
                                     )
                    expanded.append(new_node)
        else:
            raise ("Box / Tuple spaces not yet implemented")

        return expanded
