# interfaces/controller.py
from abc import ABC, abstractmethod
from typing import Any, Dict

# 1.1: Interfaz sin cambios funcionales, pero se refinan docstrings y tipos.
class Controller(ABC):
    """
    Interface for control system components.
    Defines methods for computing actions, updating parameters,
    and resetting internal states.
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
    def update_params(self, kp: float, ki: float, kd: float):
        """
        Updates the internal parameters (gains) of the controller.
        Specific parameter names (kp, ki, kd) are used assuming PID-like control.
        A more generic controller might accept a Dict[str, Any].

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
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
    def reset(self):
        """
        Resets the controller to its initial configuration, including both
        parameters (gains) and internal states (like integral error).
        """
        pass

    @abstractmethod
    def reset_internal_state(self):
        """
        Resets only the internal state variables of the controller (e.g., error terms)
        without resetting the main parameters (gains). Used when gains should persist
        across episodes but internal calculations need clearing.
        """
        pass