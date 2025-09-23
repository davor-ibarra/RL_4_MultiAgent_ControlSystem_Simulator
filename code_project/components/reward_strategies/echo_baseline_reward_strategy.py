from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller # Añadir Controller

# 8.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class EchoBaselineRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia Echo Baseline. Usa recompensas diferenciales (R_diff) pre-calculadas.
    R_learn = R_diff_g = R_real - R_cf_g.
    Implementa RewardStrategy.
    """
    def __init__(self, **kwargs): # Aceptar kwargs por si se añaden params futuros
        logger.info("EchoBaselineRewardStrategy inicializada.")
        if kwargs: logger.warning(f"EchoBaseline recibió params inesperados: {kwargs}")

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ganancia actual ('kp', 'ki', 'kd')
        agent: 'RLAgent',                   # Ignorado (no actualiza estado interno del agente)
        controller: 'Controller',           # Ignorado
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Ignorado
        current_state_indices: tuple,             # Ignorado
        # --- Action ---
        actions_dict: Dict[str, int],             # Ignorado
        action_taken_idx: int,                    # Ignorado
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real (Ignorado, implícito en R_diff)
        avg_w_stab: float,                        # Ignorado
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # R_diff = R_real - R_cf(maintain_g)
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Devuelve la recompensa diferencial pre-calculada (R_diff) para la ganancia.
        """
        reward_for_q_update = 0.0 # Default

        # 8.2: Validar reward_dict (Fail-Fast si es requerido pero None/inválido)
        if reward_dict is None:
            # Si R_diff es None, no se puede calcular R_learn para Echo.
            logger.error(f"EchoBaseline: reward_dict (R_diff) es None para ganancia '{gain}'. No se puede calcular recompensa de aprendizaje.")
            # Devolver NaN para indicar fallo? O 0? Devolver 0 es más seguro para Q-learning.
            return 0.0
        if not isinstance(reward_dict, dict):
             logger.error(f"EchoBaseline: reward_dict (R_diff) no es un dict ({type(reward_dict)}). Usando 0.")
             return 0.0

        # Extraer la recompensa diferencial específica (R_diff_g)
        diff_reward = reward_dict.get(gain)

        # 8.3: Validar el valor R_diff_g extraído
        if diff_reward is None:
            logger.warning(f"EchoBaseline: Clave '{gain}' no encontrada en reward_dict (R_diff): {list(reward_dict.keys())}. Usando 0.")
            reward_for_q_update = 0.0
        elif pd.isna(diff_reward) or not np.isfinite(diff_reward):
             logger.warning(f"EchoBaseline: Valor R_diff para '{gain}' inválido ({diff_reward}). Usando 0.")
             reward_for_q_update = 0.0
        else:
            # El valor es válido, usarlo como R_learn
            reward_for_q_update = float(diff_reward)
            # logger.debug(f"EchoBaseline: R_learn = R_diff = {reward_for_q_update:.4f} para '{gain}'.")

        return reward_for_q_update