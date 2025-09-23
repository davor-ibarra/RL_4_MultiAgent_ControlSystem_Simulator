from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict # For type hinting

class Environment(ABC):
    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[Any, float, Any]:
        """
        Applies actions to the environment, advances one timestep, calculates reward.

        Args:
            actions: The actions selected by the agent (e.g., {'kp': 0, 'ki': 1, 'kd': 2}).

        Returns:
            A tuple containing:
            - next_state: The state of the environment after the step.
            - reward: The reward received during the step.
            - info: Additional information (e.g., control force applied).
        """
        pass

    @abstractmethod
    def reset(self, initial_conditions: Any) -> Any:
        """
        Resets the environment to a starting state.

        Args:
            initial_conditions: The initial state vector or configuration.

        Returns:
            The initial state of the environment.
        """
        pass

    @abstractmethod
    def check_termination(self, config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """
        Checks if the current episode should terminate based on defined criteria.
        Does NOT check for time limit, as that's handled in the main loop.

        Args:
            config: The simulation configuration dictionary.

        Returns:
            A tuple indicating termination conditions:
            (angle_exceeded, cart_exceeded, stabilized).
        """
        pass

    # select_action is usually handled directly by the agent, no need for abstract method here
    # unless the environment itself modifies or interprets agent actions.
    # @abstractmethod
    # def select_action(self, state: Any) -> Dict[str, Any]:
    #     pass