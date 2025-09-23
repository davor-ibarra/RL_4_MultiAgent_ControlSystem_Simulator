# interfaces/stability_calculator.py
from abc import ABC, abstractmethod
from typing import Any, Dict

# 8.1: Interfaz sin cambios funcionales, docstrings mejorados.
class BaseStabilityCalculator(ABC):
    """
    Abstract base class for stability calculation components.
    Defines the interface for calculating instantaneous stability scores (w_stab)
    and potentially stability-based rewards. Also supports updating internal
    reference statistics for adaptive calculators.
    """

    @abstractmethod
    def calculate_instantaneous_stability(self, state: Any) -> float:
        """
        Calculates an instantaneous stability score (typically between 0 and 1)
        based on the current state. This score (w_stab) might be used by the
        RewardFunction or directly in some RewardStrategies (e.g., Shadow Baseline).

        Args:
            state (Any): The current state vector or representation.

        Returns:
            float: Calculated stability score (w_stab), clamped to [0, 1].
                   Should return a default value (e.g., 1.0 or 0.0) on calculation error.
        """
        pass

    @abstractmethod
    def calculate_stability_based_reward(self, state: Any) -> float:
        """
        Calculates a reward value based *solely* on the system's stability,
        derived from the current state. Used when the main reward calculation
        method is configured to 'stability_calculator'.

        Args:
            state (Any): The current state vector or representation.

        Returns:
            float: Calculated reward value based on stability.
                   Should return a default value (e.g., 0.0) on calculation error.
        """
        pass

    @abstractmethod
    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates internal reference statistics (like mean 'mu' and std dev 'sigma')
        based on data collected during a completed episode.
        Intended for adaptive calculators (e.g., IRA). Non-adaptive calculators
        should provide an empty implementation (`pass`).

        Args:
            episode_metrics_dict (Dict): Dictionary with lists of metrics from the episode.
            current_episode (int): Episode number that just finished.
        """
        pass

    @abstractmethod
    def get_current_adaptive_stats(self) -> Dict:
        """
        Returns the current internal reference statistics (e.g., mu, sigma per variable)
        used by the calculator. Intended for logging and debugging adaptive calculators.

        Returns:
            Dict: Dictionary with current adaptive statistics (e.g.,
                  {'angle': {'mu': 0.1, 'sigma': 0.5}, ...}) or {} if non-adaptive.
        """
        pass