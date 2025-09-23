from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

class EchoBaselineRewardStrategy(RewardStrategy):
    """
    Uses pre-calculated differential rewards from the reward_dict passed from
    the simulation manager (which runs the virtual simulations).
    """
    def compute_reward_for_learning(
        self,
        gain: str,
        interval_reward: float, # Ignored
        avg_w_stab: float, # Ignored
        reward_dict: Dict[str, float],
        agent_state_dict: Dict[str, Any], # Ignored
        agent: 'PIDQLearningAgent', # Ignored
        action_taken_idx: int, # Ignored
        current_state_indices: tuple, # Ignored
        **kwargs
    ) -> float:
        """ Returns the differential reward R_real - R_counterfactual for the specific gain. """
        diff_reward = reward_dict.get(gain, 0.0)
        if gain not in reward_dict:
            logging.warning(f"Gain '{gain}' not found in echo_baseline reward dictionary: {reward_dict}. Using 0 reward.")
        return diff_reward