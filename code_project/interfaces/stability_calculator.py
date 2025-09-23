from abc import ABC, abstractmethod
from typing import Any, Dict

class StabilityCalculator(ABC):
    """Interface for components that calculate a system stability score."""

    @abstractmethod
    def calculate_instantaneous_stability(self, state: Any) -> float:
        """
        Calculates an instantaneous stability score based on the current state.

        Args:
            state: The current state vector or representation of the system.

        Returns:
            A float representing the stability score (typically between 0 and 1,
            where 1 is more stable).
        """
        pass

    @abstractmethod
    def calculate_stability_based_reward(self, state: Any) -> float:
        """
        Calculates a reward value derived directly from the stability metric.

        Args:
            state: The current state vector or representation of the system.

        Returns:
            A float representing the calculated reward based on stability.
        """
        pass

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Optionally updates the internal reference statistics (e.g., mu, sigma)
        based on data from a completed episode. Implementation depends on the specific calculator.

        Args:
            episode_metrics_dict (Dict): Dictionary containing lists of metrics
                                         collected during the episode.
            current_episode (int): The episode number that just finished.
        """
        # Default implementation does nothing, concrete classes override if needed.
        pass