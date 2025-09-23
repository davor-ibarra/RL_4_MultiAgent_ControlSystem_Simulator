from abc import ABC, abstractmethod
from typing import Any, Dict, List

class MetricsCollector(ABC):
    """
    Interface for collecting metrics during a simulation episode.
    """
    @abstractmethod
    def log(self, metric_name: str, metric_value: Any):
        """
        Logs a single value for a specific metric during the current step or interval.

        Args:
            metric_name (str): The name of the metric (e.g., 'pendulum_angle', 'reward').
            metric_value (Any): The value of the metric. Should handle numerical types
                                and potentially NaN or None.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, List[Any]]:
        """
        Returns all metrics collected during the current episode.

        Returns:
            Dict[str, List[Any]]: A dictionary where keys are metric names and values
                                  are lists of collected values for that metric during the episode.
                                  Should also include the 'episode' ID.
        """
        pass

    @abstractmethod
    def reset(self, episode_id: int):
        """
        Clears all previously collected metrics and sets the ID for the next episode.

        Args:
            episode_id (int): The ID of the new episode about to start.
        """
        pass

    # Optional: Define specific logging methods for common complex data if needed
    # @abstractmethod
    # def log_q_values(self, agent: 'RLAgent', state: Any): pass
    #
    # @abstractmethod
    # def log_td_errors(self, errors: Dict[str, float]): pass