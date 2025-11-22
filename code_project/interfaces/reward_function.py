# interfaces/reward_function.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class RewardFunction(ABC):
    """Interface for reward calculation components.

    Implementations should provide a `calculate` method that returns a dictionary
    of reward parameters (e.g., `'total_reward'`). An optional
    `reward_components` dictionary can be supplied containing pre‑computed metric
    values such as Lagrangian terms. This keeps the reward calculator agnostic to
    the environment's internal state representation.
    """

    @abstractmethod
    def calculate(
        self,
        state_dict: Dict[str, Any],
        action_a: Any,
        next_state_dict: Dict[str, Any],
        current_episode_time_sec: float,
        dt_sec: float,
        goal_reached_in_step: bool,
        reward_components: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate the instantaneous reward.

        Args:
            state_dict: State before the action.
            action_a: Action taken.
            next_state_dict: State after the action.
            current_episode_time_sec: Current simulation time.
            dt_sec: Duration of the step.
            goal_reached_in_step: Whether the goal was reached.
            reward_components: Optional pre‑computed metrics (e.g., Lagrangian
                terms). Implementations may ignore this if they compute metrics
                internally.

        Returns:
            Dict[str, Any]: Reward parameters, typically containing a
                `'total_reward'` entry.
        """
        pass

    @abstractmethod
    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Update internal statistics based on episode metrics.

        Implementations can leave this empty if no adaptive components are used.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the reward calculator to its initial configuration."""
        pass

    @abstractmethod
    def get_params_log(self) -> Dict[str, Any]:
        """Return a dictionary of reward parameters for logging purposes."""
        pass