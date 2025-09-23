from abc import ABC, abstractmethod
from typing import Any, Dict

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
            state: The current state vector or representation.

        Returns:
            float: The calculated stability score (e.g., w_stab). Should be clamped to [0, 1].
                   Returns 1.0 or 0.0 in case of calculation errors or invalid state.
        """
        pass

    @abstractmethod
    def calculate_stability_based_reward(self, state: Any) -> float:
        """
        Calculates a reward value based *solely* on the system's stability,
        derived from the current state. This is used when the main reward calculation
        method is set to 'stability_calculator'.

        Args:
            state: The current state vector or representation.

        Returns:
            float: The calculated reward value based on stability.
                   Return value depends on the specific calculator's formula (e.g., exp(-lambda*Z^2)).
                   Returns 0.0 or a default low value on error.
        """
        pass

    @abstractmethod
    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates internal reference statistics (like mean 'mu' and std dev 'sigma')
        based on data collected during a completed episode.
        This method is primarily intended for adaptive calculators (e.g., IRA).
        Non-adaptive calculators should provide an empty implementation (`pass`).

        Args:
            episode_metrics_dict (Dict): Dictionary containing lists of metrics
                                         collected during the episode.
            current_episode (int): The episode number that just finished.
        """
        pass

    @abstractmethod
    def get_current_adaptive_stats(self) -> Dict:
         """
         Returns the current internal reference statistics (e.g., mu, sigma per variable)
         used by the calculator. Intended for logging and debugging adaptive calculators.
         Returns an empty dictionary if the calculator is not adaptive or has no stats.

         Returns:
            Dict: A dictionary containing the current adaptive statistics,
                  e.g., {'angle': {'mu': 0.1, 'sigma': 0.5}, ...} or {}.
         """
         pass