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
    def step(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Advances the environment by one time step (dt).

        Use internal Controller and DynamicSystem to calculate action and next state.
        Does NOT calculate rewards.

        Returns:
            Tuple containing:
            - state_dict (Dict[str, Any]): The state of the environment after the time step.
            - stab_info (Dict[str, Any]): Stability metrics for this step.
            - subenv_context (Dict[str, Any]): Context including 'done_flags', 'raw_metrics' (e.g. Lagrangian terms),
                                             and 'action_applied'.
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
    def check_termination(self) -> Tuple[bool, bool, bool]:
        """
        Checks if the current episode should terminate based on defined criteria
        (e.g., state limits, stabilization goals) found within the provided config.
        This method usually does NOT check for the maximum time limit, which is
        typically handled by the simulation loop.
        
        Returns:
            Tuple[bool, bool, bool]: A tuple indicating termination conditions:
                                     (limit_exceeded, goal_reached, other_condition).
                                     Specific meaning depends on implementation (e.g.,
                                     (angle_or_cart_limit_exceeded, stabilized, False)).
        """
        pass
    
    @abstractmethod
    def get_params_log(self) -> Dict[str, Any]:
        """
        Returns a dictionary of environment parameters for logging purposes.
        This method centralizes the exposure of loggable data.
        """
        pass