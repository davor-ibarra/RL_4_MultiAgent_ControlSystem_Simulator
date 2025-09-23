# interfaces/controller.py
from abc import ABC, abstractmethod
from typing import Any, Dict

# 1.1: Interfaz sin cambios funcionales, pero se refinan docstrings y tipos.
class Controller(ABC):
    """
    Interface for control system components.
    Defines methods for computing actions, updating parameters,
    getting targets, and resetting internal states and policies.
    """
    @abstractmethod
    def compute_action(self, state: Any) -> float:
        """
        Computes the control action based on the current system state.

        Args:
            state (Any): The current state vector or representation of the system
                         (structure depends on the specific system).

        Returns:
            float: The calculated control action (e.g., force, torque).
                   Should return 0.0 or a sensible default if calculation fails internally.
        """
        pass

    @abstractmethod
    def update_params(self, new_gains_dict: Dict[str, float]):
        """
        Updates the controller's parameters from a dictionary of new gains.
        Each controller implementation is responsible for picking the gains it cares about.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """
        Returns the current parameters (gains) of the controller.

        Returns:
            Dict[str, float]: A dictionary containing the current controller gains
                             (e.g., {'kp': 10.0, 'ki': 5.0, 'kd': 0.1}).
        """
        pass

    @abstractmethod
    def get_target(self) -> Any:
        """
        Returns the current target or setpoint the controller is trying to achieve.
        The type of the return value depends on the controller's nature
        (e.g., a float for a simple setpoint, a more complex structure for trajectory tracking).

        Returns:
            Any: The current target value or-configuration.
        """
        pass

    @abstractmethod
    def set_target(self, new_target: Any):
        """
        Updates the controller's target or setpoint dynamically.
        Essential for cascade control structures.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the controller to its absolute initial configuration. This includes:
        - Resetting primary parameters (e.g., PID gains) to their originally configured values.
        - Resetting all internal state variables (e.g., integral error, previous error).
        - Resetting any adaptive or learned components within the controller.
        """
        pass

    @abstractmethod
    def reset_internal_state(self):
        """
        Resets only the internal state variables of the controller (e.g., error terms, filters)
        without resetting the main parameters (e.g., PID gains) or adaptive components.
        Useful for starting a new simulation trial (like an episode) where controller
        parameters learned or set previously should persist, but transient states need clearing.
        """
        pass

    @abstractmethod
    def reset_policy(self, reset_level: str):
        """
        Resets the controller's policy or parameters based on the specified level.
        This offers more granular control over resetting than `reset()` or `reset_internal_state()`.

        Args:
            reset_level (str): A string identifierمكان المستوى المطلوب لإعادة التعيين. 
                               Examples: 
                               - "full_params_and_state": Equivalent to `reset()`.
                               - "internal_state_only": Equivalent to `reset_internal_state()`.
                               - "adaptive_components": Resets only learned/adaptive parts.
                               - "to_safe_defaults": Resets to a known safe parameter set.
                               Implementations should define which levels they support.
        """
        pass

    @abstractmethod
    def get_params_log(self) -> Dict[str, Any]:
        """
        Returns a dictionary of controller parameters for logging purposes.
        This method centralizes the exposure of loggable data.
        """
        pass