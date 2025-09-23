# interfaces/virtual_simulator.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

# 9.1: Interfaz sin cambios funcionales, docstrings mejorados.
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
                                controller_gains_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Runs a self-contained virtual simulation for a specified time interval
        using a specific set of controller gains. Uses copies of the system,
        controller, and reward function components internally.

        Args:
            initial_state_vector (Any): Starting state vector (e.g., numpy array)
                                        for the virtual simulation.
            start_time (float): Starting time (t) for the virtual simulation.
            duration (float): Duration (e.g., decision_interval) of the virtual
                              interval to simulate.
            controller_gains_dict (Dict[str, float]): Dictionary specifying the *fixed*
                                                     PID gains {'kp': v, 'ki': v, 'kd': v}
                                                     to be used *throughout* this virtual run.

        Returns:
            Tuple[float, float]: A tuple containing:
                                 - Total accumulated reward calculated during the virtual simulation.
                                 - Average w_stab (stability score) during the virtual simulation.
                                 Should return (0.0, 1.0) or similar defaults if the
                                 simulation fails or encounters errors internally.
        """
        pass