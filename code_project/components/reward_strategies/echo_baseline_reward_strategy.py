from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
import numpy as np # Para NaN

# Evitar importación circular
if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class EchoBaselineRewardStrategy(RewardStrategy):
    """
    Estrategia de Recompensa Echo Baseline.
    Utiliza recompensas diferenciales pre-calculadas (R_real - R_counterfactual)
    proporcionadas por el Simulation Manager a través del `reward_dict`.
    """
    def __init__(self):
        logger.info("EchoBaselineRewardStrategy inicializada.")

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,
        agent: 'PIDQLearningAgent', # Ignorado aquí
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Ignorado aquí
        current_state_indices: tuple,             # Ignorado aquí
        # --- Action ---
        actions_dict: Dict[str, int],             # Ignorado aquí
        action_taken_idx: int,                    # Ignorado aquí
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # Ignorado aquí
        avg_w_stab: float,                        # Ignorado aquí
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # R_diff = R_real - R_cf(maintain)
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Devuelve la recompensa diferencial pre-calculada para la ganancia específica.

        Retorna R_diff[gain] si está disponible en reward_dict, si no, 0.0 con warning.
        """
        reward_for_q_update = 0.0 # Valor por defecto

        if reward_dict is None:
            logger.warning(f"EchoBaseline: reward_dict es None para ganancia '{gain}'. Usando recompensa 0.")
            return 0.0

        if not isinstance(reward_dict, dict):
             logger.warning(f"EchoBaseline: reward_dict no es un diccionario para ganancia '{gain}'. Usando recompensa 0.")
             return 0.0

        # Obtener R_diff para la ganancia actual
        diff_reward = reward_dict.get(gain)

        if diff_reward is None:
            logger.warning(f"EchoBaseline: Ganancia '{gain}' no encontrada en reward_dict: {reward_dict.keys()}. Usando recompensa 0.")
            reward_for_q_update = 0.0
        elif not isinstance(diff_reward, (float, int)):
             logger.warning(f"EchoBaseline: Valor de R_diff para ganancia '{gain}' no es numérico ({diff_reward}). Usando recompensa 0.")
             reward_for_q_update = 0.0
        elif not np.isfinite(diff_reward):
             logger.warning(f"EchoBaseline: Valor de R_diff para ganancia '{gain}' es NaN/inf ({diff_reward}). Usando recompensa 0.")
             reward_for_q_update = 0.0
        else:
            reward_for_q_update = float(diff_reward)
            # logger.debug(f"EchoBaseline: Usando R_diff={reward_for_q_update:.4f} para ganancia '{gain}'.")

        return reward_for_q_update