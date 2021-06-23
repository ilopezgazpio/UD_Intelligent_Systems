from collections import deque
from collections import defaultdict
from UDAI.frontier.Node import Node


class Frontier:

    def __init__(self):
        self.nodes = deque()
        self.visited_nodes = defaultdict(bool) # default behaviour, node visited = False
        self.frontier_insertion_function = lambda x: x #identity function
        self.frontier_ordering_function = lambda x: x #identity function
        self.frontier_post_processing_function = lambda x: x  # identity function


    def __insert_one_left__(self, new_node : Node) -> None:
        self.nodes.appendleft(new_node)


    def __insert_one_right__(self, new_node: Node) -> None:
        self.nodes.append(new_node)


    def __get_one_left__(self) -> Node:
        return self.nodes.popleft()


    def __get_one_right__(self) -> Node:
        return self.nodes.pop()


    def __insert_all_left__(self, list_nodes: list) -> None:
        self.nodes.extendleft(list_nodes)


    def __insert_all_right__(self, list_nodes: list) -> None:
        self.nodes.extend(list_nodes)


    def __insert_all_right_not_visited__(self, list_nodes: list) -> None:
        for node in list_nodes:
            if not self.visited_nodes[tuple(node.observation.reshape(-1).tolist())]:
                self.__insert_one_right__(node)
                self.visited_nodes.update( {tuple(node.observation.reshape(-1).tolist()) : True} )

    def __insert_all_left_not_visited__(self, list_nodes: list) -> None:
        for node in reversed(list_nodes):
            if not self.visited_nodes[tuple(node.observation.reshape(-1).tolist())]:
                self.__insert_one_left__(node)
                self.visited_nodes.update({tuple(node.observation.reshape(-1).tolist()): True})

    def __get_frontier_max_depth__(self):
        return max ( [node.depth for node in self.nodes], default=0 )