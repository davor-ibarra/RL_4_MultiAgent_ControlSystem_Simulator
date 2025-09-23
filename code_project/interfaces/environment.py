# interfaces/environment.py
from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict # For type hinting

# 3.1: Interfaz sin cambios funcionales, pero se refinan docstrings y tipos.
# 3.2: Eliminadas propiedades comentadas.
class Environment(ABC):
    """
    Interface for reinforcement learning environments.
    Defines methods for stepping the simulation, resetting, checking termination,
    and updating internal statistics.
    """
    @abstractmethod
    def step(self) -> Tuple[Any, Tuple[float, float], Any]:
        """
        Advances the environment by one time step (dt). This typically involves:
        1. Calculating the control action (using the injected Controller).
        2. Applying the action to the dynamic system (using the injected DynamicSystem).
        3. Calculating the reward and stability score (using the injected RewardFunction).
        4. Updating the internal state and time.

        Note: Agent action selection and learning are handled externally by the
              Simulation Manager, which calls this step method.

        Returns:
            Tuple[Any, Tuple[float, float], Any]: A tuple containing:
            - next_state (Any): The state of the environment after the time step.
            - reward_stability (Tuple[float, float]): A tuple (reward, w_stab) for this step.
                                                      Implementations should ensure these are finite numbers.
            - info (Any): Additional diagnostic information (e.g., control force applied).
                          Can be None or an empty dict if not used.

        Raises:
            RuntimeError: If called before reset() or if a critical internal error occurs.
        """
        pass

    @abstractmethod
    def reset(self, initial_conditions: Any) -> Any:
        """
        Resets the environment to a starting state defined by initial_conditions.
        This should reset the internal state of the system, controller (partially/fully),
        agent (e.g., epsilon decay), and internal time.

        Args:
            initial_conditions (Any): The initial state vector or configuration.

        Returns:
            Any: The initial state of the environment after the reset.

        Raises:
            RuntimeError: If resetting fails critically.
        """
        pass

    @abstractmethod
    def check_termination(self, config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """
        Checks if the current episode should terminate based on defined criteria
        (e.g., state limits, stabilization goals) found within the provided config.
        This method usually does NOT check for the maximum time limit, which is
        typically handled by the simulation loop.

        Args:
            config (Dict[str, Any]): The main simulation configuration dictionary,
                                     providing access to termination limits/criteria.

        Returns:
            Tuple[bool, bool, bool]: A tuple indicating termination conditions:
                                     (limit_exceeded, goal_reached, other_condition).
                                     Specific meaning depends on implementation (e.g.,
                                     (angle_or_cart_limit_exceeded, stabilized, False)).
        """
        pass

    @abstractmethod
    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Triggers an update of internal statistics within the environment's
        reward function or its components (e.g., adaptive stability calculator),
        using data collected during the completed episode. Should be safe to call
        even if the reward function is not adaptive.

        Args:
            episode_metrics_dict (Dict): Dictionary containing lists of metrics
                                         collected during the just-finished episode.
            current_episode (int): The index of the episode that just finished.
        """
        pass