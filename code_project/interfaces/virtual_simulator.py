from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class VirtualSimulator(ABC):
    """
    Interface for components that can run virtual simulations of an environment
    interval without affecting the main environment's state.
    """

    @abstractmethod
    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        """
        Runs a virtual simulation for a specified time interval.

        Args:
            initial_state_vector: The starting state vector for the virtual simulation.
            start_time: The starting time (t) for the virtual simulation.
            duration: The duration of the virtual interval to simulate.
            controller_gains_dict: A dictionary specifying the PID gains
                                   {'kp': value, 'ki': value, 'kd': value}
                                   to be used *throughout* this virtual simulation.

        Returns:
            The total accumulated reward during the virtual simulation interval.
            Returns 0.0 or a default low value if simulation fails.
        """
        pass