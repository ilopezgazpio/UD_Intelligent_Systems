from collections import deque
from UDAI.frontier.Node import Node


class Frontier:

    def __init__(self, nodes = deque()):
        self.nodes = nodes
        self.frontier_insertion_function = lambda x: x #identity function
        self.frontier_ordering_function = lambda x: x #identity function
        self.frontier_post_processing_function = lambda x: x  # identity function


    def __insert_one_left__(self, new_node : Node) -> None:
        self.nodes.appendleft(new_node)


    def __insert_one_right__(self, new_node: Node) -> None:
        self.nodes.appendright(new_node)


    def __get_one_left__(self) -> Node:
        return self.nodes.popleft()


    def __get_one_right__(self) -> Node:
        return self.nodes.pop()


    def __insert_all_left__(self, list_nodes: list) -> None:
        self.nodes.extendleft(list_nodes)


    def __insert_all_right__(self, list_nodes: list) -> None:
        self.nodes.extend(list_nodes)

    def __get_frontier_max_depth__(self):
        return max ( [node.depth for node in self.nodes] )