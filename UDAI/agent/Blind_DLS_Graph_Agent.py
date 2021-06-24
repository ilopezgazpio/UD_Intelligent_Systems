from UDAI.agent.Agent import Agent
from UDAI.frontier.Node import Node


class Blind_DLS_Graph_Agent(Agent):

    def __init__(self):
        super().__init__()


    def blind_search(self, first_node:Node, max_steps=5000, max_depth=10):
        # Set frontier ordering function
        # BFS integrates expanded as FIFO
        # Graph search needs to remember visited nodes before inserting extended nodes to frontier
        self.__initialize_frontier__(first_node)
        self.frontier.frontier_insertion_function = self.frontier.__insert_all_left_not_visited__
        self.frontier.expanded_post_processing_function = self.frontier.__remove_max_depth_nodes__
        self.frontier.max_depth_nodes = max_depth
        super().__blind_search__(max_steps)
