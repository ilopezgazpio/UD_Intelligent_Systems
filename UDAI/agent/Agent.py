from abc import ABC, abstractmethod
from gym import spaces
from copy import deepcopy
from UDAI.frontier.Node import Node
from UDAI.frontier.Frontier import Frontier
from UDAI.reporting.Report import Report

class Agent(ABC):

    # TODO iteration count limiter

    def __init__(self):
        self.step = 0
        self.initial_node = None
        self.final_node = None
        self.frontier = Frontier()
        self.reporting = Report()


    def __initialize_frontier__(self, initial_node : Node):
        self.initial_node = initial_node
        self.frontier.__insert_one_left__(initial_node)


    def __printlog__(self, print_every=100):
        if (self.step % print_every == 0):
            print(self.reporting.log.tail(1))


    def __blind_search__(self, max_steps=5000, depth_limit=None):
        '''
        This method implements the main search loop and must be implemented with the particularities of each search agent
        '''
        # Dummy node for initialization, done = false
        current = Node()

        while not current.done and len(self.frontier.nodes) > 0 and self.step <= max_steps:
            current = self.frontier.__get_one_left__()
            expanded = self.__expand_node__(current)

            # Algorithm agnostic implementation on what filter to apply on expanded nodes (such as a depth limit for DLS)
            expanded = self.frontier.expanded_post_processing_function(expanded)

            # Algorithm agnostic implementation on how to insert expanded nodes in frontier
            self.frontier.frontier_insertion_function(expanded)

            self.reporting.__append__(current.done, current.cost, len(expanded),
                                      self.frontier.__get_frontier_max_depth__(), len(self.frontier.nodes),
                                      current.depth, current.cost)
            self.__printlog__(print_every=1)

            self.step += 1

        if current.done:
            self.final_node = current


    def __save_iteration__(self):
        pass


    def __heuristic__(self, node: Node) -> float:
        '''

        Parameters
        ----------
        node

        Returns
        -------
        Utility value of a node (from the standpoint of the agent)

        '''

        return 0


    def __expand_node__(self, node: Node) -> list:
        '''

        Parameters
        ----------
        node

        Returns
        -------
        a list of Nodes consequence of expanding node

        Note
        -----
        action_space must be an openai gym Space accepted by environment.step(action) for every action in action_space
        Note that not all openai gym environments implement the __is_applicable__ internal method.
        If so __expand_node__ should be overwritten accordingly :)

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
