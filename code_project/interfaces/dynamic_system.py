# interfaces/dynamic_system.py
from abc import ABC, abstractmethod
import numpy as np # Importar numpy para type hint potencial
from typing import Any # Import Any for type hint

# 2.1: Interfaz sin cambios funcionales, pero se refinan docstrings y tipos.
class DynamicSystem(ABC):
    """
    Interface for dynamic system models.
    Defines methods for applying actions and resetting the system state.
    """
    @abstractmethod
    def apply_action(self, state: Any, action: float, t: float, dt: float) -> Any:
        """
        Applies a control action to the system and computes the next state
        after a time step dt, typically by solving the system's dynamics.

        Args:
            state (Any): The current state vector or representation (e.g., np.ndarray).
            action (float): The control action applied (e.g., force).
            t (float): The current simulation time.
            dt (float): The time step duration.

        Returns:
            Any: The next state vector or representation after dt. The implementing class
                 should handle potential numerical errors during integration and return
                 a valid state (possibly the previous state) if errors occur.
        """
        pass

    @abstractmethod
    def reset(self, initial_conditions: Any) -> Any:
        """
        Resets the system state to the given initial conditions.

        Args:
            initial_conditions (Any): The desired starting state vector or configuration.

        Returns:
            Any: The validated and potentially normalized initial state vector after resetting.
        """
        pass