from abc import ABC, abstractmethod
from typing import Tuple, Any

class RewardFunction(ABC):

    @abstractmethod
    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """
        Calculates the reward and a stability score for the given transition.

        Args:
            state: The state before the action.
            action: The action taken (e.g., force applied or agent action dict).
            next_state: The state after the action.
            t: The current time.

        Returns:
            A tuple containing:
            - reward (float): The calculated reward value for the agent's learning process.
            - stability_score (float): A score indicating system stability (e.g., w_stab, typically between 0 and 1).
                                        Defaults to 1.0 if not calculated.
        """
        pass