from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

class GlobalRewardStrategy(RewardStrategy):
    """
    Uses the total accumulated interval reward directly for learning all gains.
    """
    def compute_reward_for_learning(
        self,
        gain: str,
        interval_reward: float,
        avg_w_stab: float,
        reward_dict: Dict[str, float],
        agent_state_dict: Dict[str, Any],
        agent: 'PIDQLearningAgent',
        action_taken_idx: int,
        current_state_indices: tuple,
        **kwargs
    ) -> float:
        """ Returns the global interval reward. """
        # Other arguments (avg_w_stab, reward_dict, etc.) are ignored.
        return interval_reward