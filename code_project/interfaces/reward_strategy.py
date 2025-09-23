from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Union, Tuple

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

class RewardStrategy(ABC):
    """
    Interface for different reward calculation strategies used in the agent's learning step.
    """

    @abstractmethod
    def compute_reward_for_learning(
        self,
        gain: str,                           # The specific gain ('kp', 'ki', 'kd') being updated
        interval_reward: float,              # R_real: The actual reward accumulated during the interval
        avg_w_stab: float,                   # Average stability score during the interval (for Shadow)
        reward_dict: Dict[str, float],       # Dictionary of differential rewards (for Echo)
        agent_state_dict: Dict[str, Any],    # State dictionary S at the start of the interval
        agent: 'PIDQLearningAgent',          # Reference to the agent instance (for accessing tables)
        action_taken_idx: int,               # The index of the action taken (0, 1, or 2)
        current_state_indices: tuple,        # Discretized state indices for S
        **kwargs                            # For future extensions
    ) -> float:
        """
        Calculates the specific reward value to be used in the Q-learning update for a given gain.
        This method might also update internal structures (like baseline tables).

        Args:
            gain (str): The gain ('kp', 'ki', 'kd') whose Q-table is being updated.
            interval_reward (float): The total reward accumulated in the real environment
                                     during the decision interval.
            avg_w_stab (float): Average stability score during the interval.
            reward_dict (Dict[str, float]): Pre-calculated differential rewards (used by Echo).
                                             Format: {'kp': R_diff_kp, 'ki': R_diff_ki, 'kd': R_diff_kd}
            agent_state_dict (Dict[str, Any]): The agent's state dictionary at the beginning
                                               of the interval (S).
            agent (PIDQLearningAgent): The agent instance itself, providing access to Q-tables,
                                       baseline tables, visit counts etc.
            action_taken_idx (int): Index of the action taken for this gain (0, 1, 2).
            current_state_indices (tuple): The tuple of discretized state indices corresponding
                                           to agent_state_dict for the specific gain's table.
            kwargs: Additional keyword arguments.

        Returns:
            float: The reward value to be used in the TD error calculation for the specified gain.
        """
        pass