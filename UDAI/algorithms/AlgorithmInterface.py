from abc import ABC, abstractmethod
from ..frontier.Frontier import Frontier

class AlgorithmInterface(ABC):

    @abstractmethod
    def sort(self, frontier: Frontier) -> Frontier:
        None

