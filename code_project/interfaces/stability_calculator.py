from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseStabilityCalculator(ABC):
    """
    Abstract base class for stability calculation components.
    Defines the interface for calculating instantaneous stability scores
    and stability-based rewards. Also supports updating internal statistics.
    """

    @abstractmethod
    def calculate_instantaneous_stability(self, state: Any) -> float:
        """
        Calculates an instantaneous stability score (typically between 0 and 1)
        based on the current state.

        Args:
            state: The current state vector or representation.

        Returns:
            float: The calculated stability score (e.g., w_stab).
        """
        pass

    @abstractmethod
    def calculate_stability_based_reward(self, state: Any) -> float:
        """
        Calculates a reward value based on the system's stability, derived
        from the current state.

        Args:
            state: The current state vector or representation.

        Returns:
            float: The calculated reward value.
        """
        pass

    @abstractmethod
    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Updates internal reference statistics (like mean 'mu' and std dev 'sigma')
        based on data collected during a completed episode.
        This method is intended for adaptive calculators (e.g., IRA). Non-adaptive
        calculators can provide an empty implementation.

        Args:
            episode_metrics_dict (Dict): Dictionary containing lists of metrics
                                          collected during the episode.
            current_episode (int): The episode number that just finished.
        """
        pass

    # Optional: Add a method to retrieve current stats for logging
    def get_current_adaptive_stats(self) -> Dict:
         """
         Returns the current internal reference statistics (mu, sigma).
         Returns an empty dict if not applicable.
         """
         return {}