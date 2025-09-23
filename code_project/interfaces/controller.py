from abc import ABC, abstractmethod

class Controller(ABC):
    @abstractmethod
    def compute_action(self, state):
        pass

    @abstractmethod
    def update_params(self, params):
        pass

    @abstractmethod
    def reset(self):
        pass
