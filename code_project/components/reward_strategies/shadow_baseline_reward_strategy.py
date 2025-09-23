from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, TYPE_CHECKING
import logging
import numpy as np

if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

class ShadowBaselineRewardStrategy(RewardStrategy):
    """
    Calculates reward based on the difference from a learned baseline B(s).
    Updates the baseline table B(s) when the 'maintain' action is taken.
    """
    def __init__(self, beta: float = 0.1):
        self.beta = beta
        logging.info(f"ShadowBaselineRewardStrategy initialized with beta={self.beta}")
        self.all_gains = ['kp', 'ki', 'kd']

    def compute_reward_for_learning(
        self,
        gain: str,
        interval_reward: float,
        avg_w_stab: float,
        reward_dict: Dict[str, float], # Ignored
        agent_state_dict: Dict[str, Any], # Ignored (indices used)
        agent: 'PIDQLearningAgent',
        action_taken_idx: int,
        current_state_indices: tuple,
        actions_dict: Dict[str, int],
        **kwargs
    ) -> float:
        """ Calculates differential reward and updates baseline if action is 'maintain'. """
        reward_for_q_update = 0.0

        # Check if baseline table exists for this gain
        if gain not in agent.baseline_tables_np:
            logging.warning(f"Baseline table for gain '{gain}' not found in agent. Using global reward {interval_reward}.")
            return interval_reward

        try:
            baseline_table = agent.baseline_tables_np[gain]
            baseline_value = baseline_table[current_state_indices]

            # --- Implementación de lógica estricta para actualizar B(s) ---
            update_baseline = False
            if action_taken_idx == 1: # Action for the current gain is 'maintain'
                # Check actions for the *other* two gains
                other_gains = [g for g in self.all_gains if g != gain]
                other_actions_not_maintain = True
                for other_gain in other_gains:
                    other_action = actions_dict.get(other_gain)
                    if other_action == 1: # If any other action is also 'maintain'
                        other_actions_not_maintain = False
                        break
                    elif other_action is None: # Handle case where action is missing (should not happen ideally)
                         logging.warning(f"ShadowBaseline: Missing action for other gain '{other_gain}' in actions_dict: {actions_dict}. Cannot confirm isolation.")
                         other_actions_not_maintain = False # Treat as non-isolated if data missing
                         break

                if other_actions_not_maintain:
                    update_baseline = True # Condition met: update B(s) for 'gain'

            # --- Calculate reward and update baseline based on the condition ---
            if update_baseline:
                # Update the baseline value B(s)
                # Asegurar que avg_w_stab es un número válido
                valid_w_stab = avg_w_stab if isinstance(avg_w_stab, (float, int, np.number)) and not np.isnan(avg_w_stab) else 1.0
                delta_B = self.beta * valid_w_stab * (interval_reward - baseline_value)
                baseline_table[current_state_indices] = baseline_value + delta_B
                # For Q-update, use the original interval reward when isolating 'maintain'
                reward_for_q_update = interval_reward
                # logging.debug(f"ShadowBaseline: Updated B(s) for gain '{gain}', state {current_state_indices}. R_Q={interval_reward:.4f}")
            else:
                # Use the differential reward R_real - B(s) for Q-update
                reward_for_q_update = interval_reward - baseline_value
                # logging.debug(f"ShadowBaseline: Using R_diff for gain '{gain}', state {current_state_indices}. R_Q={reward_for_q_update:.4f}")

        except IndexError:
            logging.error(f"IndexError accessing baseline table '{gain}' for indices {current_state_indices}. Shape: {baseline_table.shape}. Using global reward {interval_reward}.")
            reward_for_q_update = interval_reward
        except KeyError as e:
             logging.error(f"KeyError accessing actions_dict in ShadowBaseline for gain '{gain}': {e}. Actions: {actions_dict}. Using global reward {interval_reward}.")
             reward_for_q_update = interval_reward
        except Exception as e:
            logging.error(f"Unexpected error in ShadowBaseline strategy for gain '{gain}': {e}. Using global reward {interval_reward}.", exc_info=True)
            reward_for_q_update = interval_reward

        # Asegurar que la recompensa no sea NaN
        return float(reward_for_q_update) if not np.isnan(reward_for_q_update) else 0.0