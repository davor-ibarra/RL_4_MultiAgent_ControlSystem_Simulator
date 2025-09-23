from abc import ABC, abstractmethod
import numpy as np

class DynamicSystem(ABC):
    @abstractmethod
    def apply_action(self, state, action, t, dt):
        pass

    @abstractmethod
    def reset(self, initial_conditions):
        pass
