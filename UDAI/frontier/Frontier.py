from collections import deque

class Frontier:

    def __init__(self, nodes = deque()):
        self.nodes = nodes