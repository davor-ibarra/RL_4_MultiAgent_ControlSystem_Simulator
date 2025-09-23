from abc import ABC, abstractmethod
import numpy as np
from typing import Any # Import Any for type hint

class DynamicSystem(ABC):
    """
    Interface for dynamic system models.
    Defines methods for applying actions and resetting the system state.
    """
    @abstractmethod
    def apply_action(self, state: Any, action: float, t: float, dt: float) -> Any:
        """
        Applies a control action to the system and computes the next state
        after a time step dt.

        Args:
            state: The current state vector or representation.
            action: The control action applied (e.g., force).
            t: The current simulation time.
            dt: The time step duration.

        Returns:
            The next state vector or representation after dt.
        """
        pass

    @abstractmethod
    def reset(self, initial_conditions: Any) -> Any:
        """
        Resets the system state to the given initial conditions.

        Args:
            initial_conditions: The desired starting state vector or configuration.

        Returns:
            The initial state vector after resetting.
        """
        pass