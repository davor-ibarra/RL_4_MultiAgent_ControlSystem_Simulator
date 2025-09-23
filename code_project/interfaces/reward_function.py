from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional

class RewardFunction(ABC):
    """
    Interface for reward calculation components.
    Defines methods to calculate reward and stability, and potentially update internal stats.
    """

    @abstractmethod
    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """
        Calculates the instantaneous reward and a stability score for the given transition.

        Args:
            state: The state before the action was taken.
            action: The action taken (e.g., force applied, agent's action choice).
            next_state: The resulting state after the action was applied and dt passed.
            t: The current simulation time.

        Returns:
            A tuple containing:
            - reward (float): The calculated reward value for this step.
            - stability_score (float): A score indicating system stability (e.g., w_stab),
                                       typically between 0 and 1. Defaults to 1.0 if not
                                       calculated or applicable.
        """
        pass

    @abstractmethod
    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates internal statistics of any components used by the reward function
        (like an adaptive stability calculator), based on data from a completed episode.
        Implementations can leave this empty if no adaptive components are used.

        Args:
            episode_metrics_dict (Dict): Dictionary with lists of metrics from the episode.
            current_episode (int): The index of the completed episode.
        """
        pass