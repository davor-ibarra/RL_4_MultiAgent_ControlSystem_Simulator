# interfaces/metrics_collector.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# 4.1: Interfaz sin cambios funcionales.
# 4.2: Eliminados métodos específicos de log comentados (`log_q_values`, etc.)
#      de la interfaz. La implementación concreta puede tenerlos como helpers.
class MetricsCollector(ABC):
    """
    Interface for collecting metrics during a simulation episode.
    """
    @abstractmethod
    def log(self, metric_name: str, metric_value: Any):
        """
        Logs a single value for a specific metric at the current step/interval.
        Implementations should handle non-finite values gracefully (e.g., log as NaN).

        Args:
            metric_name (str): The name of the metric (e.g., 'pendulum_angle', 'reward').
            metric_value (Any): The value of the metric.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, List[Any]]:
        """
        Returns all metrics collected during the current episode.

        Returns:
            Dict[str, List[Any]]: A dictionary where keys are metric names and values
                                  are lists of collected values for that metric.
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