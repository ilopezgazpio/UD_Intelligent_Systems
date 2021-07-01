from UDAI.agent.Agent import Agent
from UDAI.frontier.Node import Node


class Blind_UCS_Graph_Agent(Agent):

    def __init__(self):
        super().__init__()


    def blind_search(self, first_node: Node, max_steps=5000):
        # Set frontier ordering function
        # UCS integrates expanded as regard de cost
        self.__initialize_frontier__(first_node)
        # self.frontier.expanded_post_processing_function = self.frontier.__absolute_cost_function__
        self.frontier.frontier_insertion_function = self.frontier.__insert_ordered_cost_not_visited__
        super().__blind_search__(max_steps)