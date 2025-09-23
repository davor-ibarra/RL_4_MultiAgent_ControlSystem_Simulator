from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Union, Tuple, Optional

# Avoid circular import for type hinting
if TYPE_CHECKING:
    # Import the specific agent type that uses these strategies if needed
    from components.agents.pid_qlearning_agent import PIDQLearningAgent
    # Or use the generic RLAgent if methods don't depend on specifics
    # from interfaces.rl_agent import RLAgent

class RewardStrategy(ABC):
    """
    Interface for different reward calculation strategies used specifically
    within the agent's learning step (e.g., Q-update).
    """

    @abstractmethod
    def compute_reward_for_learning(
        self,
        # --- Context for the update ---
        gain: str,                          # The specific gain ('kp', 'ki', 'kd') being updated
        agent: 'PIDQLearningAgent',         # Agent instance for accessing tables (Q, B, V)
        # --- State Information ---
        current_agent_state_dict: Dict[str, Any], # State S at the start of the interval (dictionary)
        current_state_indices: tuple,             # Discretized state indices for S (tuple)
        # --- Action Information ---
        actions_dict: Dict[str, int],             # Actions taken for ALL gains in this interval
        action_taken_idx: int,                    # Action index (0, 1, 2) for the specific 'gain'
        # --- Raw Reward/Stability Information from the interval ---
        interval_reward: float,                   # R_real: Actual reward accumulated during interval
        avg_w_stab: float,                        # Average stability score (w_stab) during interval
        # --- Pre-calculated Differential Rewards (for specific strategies like Echo) ---
        reward_dict: Optional[Dict[str, float]],  # Optional: Dict of R_diff (e.g., {'kp': R_diff_kp})
        # --- Optional Extra Arguments ---
        **kwargs
    ) -> float:
        """
        Calculates the specific reward value to be used in the learning update rule
        (e.g., TD error calculation) for a given gain's Q-table or policy update.
        This method might also update internal structures of the strategy or agent
        (like baseline tables B(s) in Shadow Baseline).

        Args:
            gain (str): The gain ('kp', 'ki', 'kd') whose Q-table/policy is being updated.
            agent (PIDQLearningAgent): The agent instance (provides access to Q, B, V tables).
            current_agent_state_dict (Dict[str, Any]): Agent's state dictionary at the start
                                                       of the decision interval (S).
            current_state_indices (tuple): Discretized state indices for S for the specific gain.
            actions_dict (Dict[str, int]): Dictionary of actions taken for *all* gains
                                           during the interval (e.g., {'kp': 1, 'ki': 0, 'kd': 1}).
            action_taken_idx (int): Index of the action (0, 1, or 2) taken specifically
                                    for the 'gain' being updated.
            interval_reward (float): The total reward accumulated in the real environment
                                     during the decision interval (R_real).
            avg_w_stab (float): Average stability score (w_stab) during the interval.
                                Crucial for baseline updates in Shadow mode.
            reward_dict (Optional[Dict[str, float]]): Optional dictionary containing pre-calculated
                                                      differential rewards (R_real - R_counterfactual).
                                                      Primarily used by Echo Baseline.
            kwargs: Additional keyword arguments for future extensions.

        Returns:
            float: The reward value (R_learn) to be used directly in the learning update
                   (e.g., in the TD error calculation: R_learn + gamma * max Q(s') - Q(s)).
                   This might be R_real, R_real - B(s), R_diff, or some other value
                   depending on the strategy.
        """
        pass