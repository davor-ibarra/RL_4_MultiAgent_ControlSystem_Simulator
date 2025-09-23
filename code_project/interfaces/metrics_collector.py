# interfaces/metrics_collector.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Set

class MetricsCollector(ABC):
    """
    Interface for collecting raw metrics during a simulation episode based on
    simulation events. Implementations are responsible for extracting data
    from the context dictionaries provided at each event, based on their
    own internal configuration.
    """
    @abstractmethod
    def log_on_episode_start(self, context: Dict[str, Any]):
        """
        Logs initial metrics at the very beginning of an episode using the
        provided context. Should initialize all metric lists.

        Args:
            context (Dict[str, Any]): A dictionary containing relevant objects
                                      at the start of the episode, e.g.,
                                      {'episode_id': 0, 'initial_state': ...,
                                       'controller': ..., 'agent': ...}.
        """
        pass

    @abstractmethod
    def log_on_step(self, context: Dict[str, Any]):
        """
        Logs high-frequency metrics available after every single simulation step.
        Also responsible for aligning data by padding low-frequency metrics.

        Args:
            context (Dict[str, Any]): A dictionary containing relevant objects
                                      and values for the step, e.g.,
                                      {'time': 0.01, 'state': ..., 'reward': ...,
                                       'controller': ..., 'agent': ...}.
        """
        pass

    @abstractmethod
    def log_on_decision_boundary(self, context: Dict[str, Any]):
        """
        Updates an internal cache with low-frequency metrics available at the
        agent's decision boundaries.

        Args:
            context (Dict[str, Any]): A dictionary containing relevant objects
                                      at the decision boundary, e.g.,
                                      {'decision_id': 1, 'agent': ...,
                                       'next_agent_s_dict': ...}.
        """
        pass

    @abstractmethod
    def log_on_episode_end(self, context: Dict[str, Any]):
        """
        Logs final metrics for the episode, such as the termination reason.

        Args:
            context (Dict[str, Any]): A dictionary containing final information,
                                      e.g., {'termination_reason': 'goal_reached'}.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, List[Any]]:
        """
        Returns all raw metrics collected during the current episode.

        Returns:
            Dict[str, List[Any]]: A dictionary where keys are metric names and
                                  values are lists of collected values.
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

    @abstractmethod
    def get_summary_directives(self) -> Tuple[Set[str], Set[str]]:
        """
        Returns the directives for summarizing episode data.
        
        Returns:
            A tuple containing two sets:
            - A set of keys for metrics that need their last valid value.
            - A set of keys for metrics that require statistical aggregation.
        """
        pass