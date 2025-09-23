from abc import ABC, abstractmethod

class RewardFunction(ABC):

    @abstractmethod
    def calculate(self, state, action, next_state):
        pass
