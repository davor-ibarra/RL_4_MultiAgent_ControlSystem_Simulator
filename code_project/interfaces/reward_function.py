# interfaces/reward_function.py
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional

# 5.1: Interfaz sin cambios funcionales, docstrings mejorados.
class RewardFunction(ABC):
    """
    Interface for reward calculation components.
    Defines methods to calculate reward per step, and update stats.
    """

    @abstractmethod
    def calculate(self, 
                  state_dict: Dict[str, Any], 
                  action_a: Any, 
                  next_state_dict: Dict[str, Any], 
                  current_episode_time_sec: float,
                  dt_sec: float,
                  goal_reached_in_step: bool
                  ) -> float:
        """
        Calculates the instantaneous reward value for the transition from 
        `state_dict` to `next_state_dict` given `action_a`.

        The method for calculating the reward value depends on the implementing class's
        configuration and may include penalties or bonuses.

        Args:
            state_dict (Dict[str, Any]): State dictionary before the action.
            action_a (Any): Action taken (e.g., force).
            next_state_dict (Dict[str, Any]): Resulting state dictionary after action and dt.
            current_episode_time_sec (float): Current simulation time within the episode.
            dt_sec (float): The time duration of the current simulation step.
            goal_reached_in_step (bool): Goal reached in step flag.


        Returns:
            float: The calculated reward value.
                   Values should be finite floats. Implementations should
                   handle internal errors and return defaults (e.g., 0.0)
                   instead of raising exceptions for calculation issues.
        """
        pass

    @abstractmethod
    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates internal statistics of components used by the reward function
        (like an adaptive stability calculator, if the reward function *also* uses it for some reason,
         though primary adaptive updates for stability score happen directly in BaseStabilityCalculator).
        Implementations can leave this empty if no adaptive components are used directly by the reward logic.

        Args:
            episode_metrics_dict (Dict): Dictionary with lists of metrics from the episode.
            current_episode (int): The index of the completed episode.
        """
        pass

    @abstractmethod
    def reset(self):
        """Resets the reward calculators params to its absolute initial configuration."""
        pass

    @abstractmethod
    def get_params_log(self) -> Dict[str, Any]:
        """
        Returns a dictionary of reward parameters for logging purposes.
        This method centralizes the exposure of loggable data.
        """
        pass