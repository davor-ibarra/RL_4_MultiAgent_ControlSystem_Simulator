from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict # For type hinting

class Environment(ABC):
    """
    Interface for reinforcement learning environments.
    Defines methods for stepping the simulation, resetting, checking termination,
    and updating internal statistics.
    """
    @abstractmethod
    def step(self) -> Tuple[Any, Tuple[float, float], Any]:
        """
        Advances the environment by one time step (dt). This typically involves:
        1. Calculating the control action based on the current state.
        2. Applying the action to the dynamic system.
        3. Calculating the reward and stability score for the transition.
        4. Updating the internal state and time.

        Note: Agent action selection and learning are typically handled outside
              this method by the Simulation Manager, which interacts with the agent.
              This step method focuses on the environment's dynamics and reward calculation.

        Returns:
            A tuple containing:
            - next_state (Any): The state of the environment after the time step.
            - reward_stability (Tuple[float, float]): A tuple containing the instantaneous
                                                      reward and stability score (reward, w_stab)
                                                      calculated for this step.
            - info (Any): Additional information about the step (e.g., control force applied).
        """
        pass

    @abstractmethod
    def reset(self, initial_conditions: Any) -> Any:
        """
        Resets the environment to a starting state defined by initial_conditions.
        This should also reset the internal state of the system, controller (partially or fully),
        and potentially trigger the agent's reset logic (like epsilon decay).

        Args:
            initial_conditions: The initial state vector or configuration.

        Returns:
            The initial state of the environment after the reset.
        """
        pass

    @abstractmethod
    def check_termination(self, config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """
        Checks if the current episode should terminate based on defined criteria
        (e.g., angle limits, cart position limits, stabilization).
        This method does NOT typically check for the maximum time limit, as that
        is usually handled by the main simulation loop.

        Args:
            config: The main simulation configuration dictionary, providing access
                    to termination limits and criteria.

        Returns:
            A tuple indicating termination conditions:
            (angle_exceeded, cart_exceeded, stabilized).
        """
        pass

    @abstractmethod
    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Triggers an update of internal statistics within the environment's
        reward function or its components (e.g., adaptive stability calculator),
        using data collected during the completed episode.

        Args:
            episode_metrics_dict (Dict): Dictionary containing lists of metrics
                                         collected during the just-finished episode.
            current_episode (int): The index of the episode that just finished.
        """
        pass

    # It might be useful to expose components if needed externally, though DI is preferred
    # @property
    # @abstractmethod
    # def system(self) -> DynamicSystem: pass
    #
    # @property
    # @abstractmethod
    # def controller(self) -> Controller: pass
    #
    # @property
    # @abstractmethod
    # def agent(self) -> RLAgent: pass