from abc import ABC, abstractmethod
from typing import Any, Dict

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
            state: The current state vector or representation of the system.

        Returns:
            The calculated control action (e.g., force, torque).
        """
        pass

    @abstractmethod
    def update_params(self, kp: float, ki: float, kd: float):
        """
        Updates the internal parameters (gains) of the controller.
        Specific parameter names (kp, ki, kd) are used for PID controllers,
        consider a more generic Dict[str, Any] for broader applicability if needed.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """
        Returns the current parameters (gains) of the controller.

        Returns:
            A dictionary containing the current controller gains (e.g., {'kp': 10.0, ...}).
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the controller to its initial parameters and clears internal states
        (like integral error, previous error).
        """
        pass

    @abstractmethod
    def reset_internal_state(self):
        """
        Resets only the internal state variables of the controller (e.g., error terms)
        without resetting the main parameters (gains). Used when gains persist
        across episodes but internal calculations need clearing.
        """
        pass