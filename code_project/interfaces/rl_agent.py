from abc import ABC, abstractmethod

class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def reset_agent(self):
        pass
