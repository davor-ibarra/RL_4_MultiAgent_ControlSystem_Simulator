from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class VirtualSimulator(ABC):
    """
    Interface for components that can run virtual simulations of an environment
    interval without affecting the main environment's state. Used for calculating
    counterfactual rewards in strategies like Echo Baseline.
    """

    @abstractmethod
    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        """
        Runs a self-contained virtual simulation for a specified time interval
        using a specific set of controller gains.

        Args:
            initial_state_vector: The starting state vector (e.g., numpy array)
                                  for the virtual simulation.
            start_time: The starting time (t) for the virtual simulation.
            duration: The duration (e.g., decision_interval) of the virtual
                      interval to simulate.
            controller_gains_dict: A dictionary specifying the *fixed* PID gains
                                   {'kp': value, 'ki': value, 'kd': value}
                                   to be used *throughout* this specific virtual simulation run.

        Returns:
            float: The total accumulated reward calculated during the virtual simulation
                   interval using the provided gains. Returns 0.0 or a default low value
                   if the simulation fails or encounters errors.
        """
        pass