# interfaces/reward_function.py
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional

# 5.1: Interfaz sin cambios funcionales, docstrings mejorados.
class RewardFunction(ABC):
    """
    Interface for reward calculation components.
    Defines methods to calculate reward and stability per step, and update stats.
    """

    @abstractmethod
    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """
        Calculates the instantaneous reward value and a stability score (w_stab)
        for the transition from `state` to `next_state` given `action` at time `t`.

        The method for calculating the reward value depends on the implementing class's
        configuration (e.g., 'gaussian' or using an injected 'stability_calculator').

        The stability score (w_stab) should be calculated if a StabilityCalculator
        component is available, otherwise it should default to a neutral value (e.g., 1.0).

        Args:
            state (Any): State before the action.
            action (Any): Action taken (e.g., force).
            next_state (Any): Resulting state after action and dt.
            t (float): Current simulation time.

        Returns:
            Tuple[float, float]: A tuple (reward_value, stability_score).
                                 Values should be finite floats. Implementations should
                                 handle internal errors and return defaults (e.g., 0.0, 1.0)
                                 instead of raising exceptions for calculation issues.
        """
        pass

    @abstractmethod
    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates internal statistics of components used by the reward function
        (like an adaptive stability calculator), based on data from a completed episode.
        Implementations can leave this empty if no adaptive components are used.

        Args:
            episode_metrics_dict (Dict): Dictionary with lists of metrics from the episode.
            current_episode (int): The index of the completed episode.
        """
        pass