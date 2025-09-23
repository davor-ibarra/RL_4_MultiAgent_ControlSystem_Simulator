# interfaces/reward_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Union, Tuple, Optional

# Avoid circular import for type hinting
if TYPE_CHECKING:
    # 6.1: Usar interfaz genérica RLAgent
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller # Necesario para ShadowBaseline

class RewardStrategy(ABC):
    """
    Interface for different reward calculation strategies used specifically
    within the agent's learning step (e.g., Q-update). Defines how the
    raw reward/stability information from an interval translates into the
    scalar reward value used for learning updates (R_learn).
    """

    @abstractmethod
    def compute_reward_for_learning(
        self,
        # --- Context for the update ---
        gain: str,                          # The specific gain ('kp', 'ki', 'kd') being updated
        # 6.2: Usar interfaz RLAgent
        agent: 'RLAgent',                   # Agent instance for accessing its state/tables (Q, B, V)
        controller: 'Controller',           # Controller instance (needed for Shadow baseline)
        # --- State Information ---
        current_agent_state_dict: Dict[str, Any], # State S (dict) at the start of the interval
        current_state_indices: tuple,             # Discretized state indices for S (tuple)
        # --- Action Information ---
        actions_dict: Dict[str, int],             # Actions taken for ALL gains in this interval (A)
        action_taken_idx: int,                    # Action index (0, 1, 2) for the specific 'gain'
        # --- Raw Reward/Stability Information from the interval ---
        interval_reward: float,                   # R_real: Actual reward accumulated during interval
        avg_w_stab: float,                        # Average stability score (w_stab) during interval
        # --- Pre-calculated Differential Rewards (for Echo Baseline) ---
        reward_dict: Optional[Dict[str, float]],  # Optional: Dict of R_diff (e.g., {'kp': R_diff_kp})
        # --- Optional Extra Arguments ---
        **kwargs
    ) -> float:
        """
        Calculates the specific reward value (R_learn) to be used in the agent's
        learning update rule (e.g., TD error calculation) for a given gain's
        Q-table or policy update, based on the experience gathered during the
        last decision interval.

        This method might also trigger updates to internal structures of the
        agent or strategy itself (like baseline tables B(s) in Shadow Baseline).

        Args:
            gain (str): Gain ('kp', 'ki', 'kd') whose Q-table/policy is being updated.
            agent (RLAgent): The agent instance.
            controller (Controller): The controller instance.
            current_agent_state_dict (Dict[str, Any]): Agent's state dictionary S.
            current_state_indices (tuple): Discretized state indices for S.
            actions_dict (Dict[str, int]): Actions A taken for all gains.
            action_taken_idx (int): Action index (0, 1, 2) for this specific 'gain'.
            interval_reward (float): Total real reward (R_real) during the interval.
            avg_w_stab (float): Average stability score (w_stab) during the interval.
            reward_dict (Optional[Dict[str, float]]): Pre-calculated differential rewards (for Echo).
            kwargs: Additional keyword arguments for future extensions.

        Returns:
            float: The reward value (R_learn) to be used directly in the learning update.
                   Should be a finite float.
        """
        pass