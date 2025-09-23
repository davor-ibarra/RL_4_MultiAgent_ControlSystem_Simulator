from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd # Para isnull/isna

# Evitar importación circular
if TYPE_CHECKING:
    # from components.agents.pid_qlearning_agent import PIDQLearningAgent
    from interfaces.rl_agent import RLAgent # Usar interfaz genérica

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class ShadowBaselineRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia de Recompensa Shadow Baseline.
    Calcula R_learn = R_real - B(s) y actualiza B(s) bajo condiciones específicas.
    """
    def __init__(self, beta: float = 0.1):
        """
        Inicializa la estrategia Shadow Baseline.

        Args:
            beta (float): Tasa de aprendizaje para actualizar el baseline B(s).
        """
        if not isinstance(beta, (float, int)) or not (0 <= beta <= 1):
             logger.warning(f"Beta ({beta}) inválido para ShadowBaseline. Usando 0.1 por defecto.")
             self.beta = 0.1
        else:
             self.beta = beta
        self.all_gains = ['kp', 'ki', 'kd'] # Lista de todas las ganancias posibles
        logger.info(f"ShadowBaselineRewardStrategy inicializada con beta={self.beta}")


    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ganancia actual ('kp', 'ki', 'kd')
        agent: 'RLAgent',                   # Instancia del agente para acceder a B(s)
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Ignorado (se usan índices)
        current_state_indices: tuple,             # Índices S para acceder a B(s)
        # --- Action ---
        actions_dict: Dict[str, int],             # Acciones {'kp':a_kp, ...} tomadas
        action_taken_idx: int,                    # Acción a_g para la ganancia 'gain'
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real acumulada en el intervalo
        avg_w_stab: float,                        # w_stab promedio en el intervalo
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # Ignorado aquí
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Calcula R_learn y actualiza B(s) si aplica. Implementa método de interfaz.
        """
        # ... (lógica mantenida como estaba, pero usando interfaz RLAgent) ...
        reward_for_q_update: float = 0.0

        # Validar entradas
        if not isinstance(avg_w_stab, (float, int)) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0
        if not isinstance(interval_reward, (float, int)) or not np.isfinite(interval_reward):
             logger.warning(f"ShadowBaseline: interval_reward inválido ({interval_reward}). Usando 0.0."); return 0.0

        # Verificar si el agente tiene el método para obtener B(s)
        if not hasattr(agent, 'get_baseline_value_for_state') or not hasattr(agent, 'baseline_tables_np'):
             logger.error(f"ShadowBaseline: Agente {type(agent).__name__} no tiene métodos/atributos necesarios para baseline. Usando R_real.")
             return interval_reward

        # Verificar si existe tabla Baseline para esta ganancia en el agente
        # Acceder al atributo directamente si es posible (como en PIDQLearningAgent)
        baseline_tables = getattr(agent, 'baseline_tables_np', None)
        if not isinstance(baseline_tables, dict) or gain not in baseline_tables:
            logger.warning(f"ShadowBaseline: Tabla Baseline para '{gain}' no encontrada en agente. Usando R_real.")
            return interval_reward

        try:
            baseline_table = baseline_tables[gain]
            # Obtener valor actual de B(s)
            # Usar método del agente para obtener el valor por índice
            baseline_value = agent.get_baseline_value_for_state(current_agent_state_dict).get(gain, np.nan)
            if pd.isna(baseline_value) or not np.isfinite(baseline_value):
                 logger.warning(f"ShadowBaseline: B(s) inválido ({baseline_value}) para '{gain}', state {current_state_indices}. Usando B(s)=0.")
                 baseline_value = 0.0

            # --- Determinar si actualizar B(s) ---
            update_baseline = False
            if action_taken_idx == 1: # Acción para 'gain' es 'maintain'
                other_gains = [g for g in self.all_gains if g != gain]
                is_isolated_maintain = True
                for other_gain in other_gains:
                    other_action = actions_dict.get(other_gain)
                    if other_action == 1 or other_action is None: # Si otra es maintain o falta info
                        is_isolated_maintain = False
                        if other_action is None: logger.warning(f"ShadowBaseline: Acción faltante para '{other_gain}'")
                        break
                if is_isolated_maintain: update_baseline = True

            # --- Calcular R_learn y Actualizar B(s) ---
            if update_baseline:
                delta_B = self.beta * avg_w_stab * (interval_reward - baseline_value)
                new_baseline = baseline_value + delta_B
                if pd.notna(new_baseline) and np.isfinite(new_baseline):
                     baseline_table[current_state_indices] = new_baseline # Actualizar tabla directamente
                     # logger.debug(f"ShadowBaseline: B(s) updated '{gain}' {current_state_indices} -> {new_baseline:.4f}")
                else:
                     logger.warning(f"ShadowBaseline: Nuevo B(s) inválido ({new_baseline}) para '{gain}'. No actualizado.")
                reward_for_q_update = interval_reward
                # logger.debug(f"ShadowBaseline: R_learn = R_real = {reward_for_q_update:.4f} ('{gain}')")
            else:
                reward_for_q_update = interval_reward - baseline_value
                # logger.debug(f"ShadowBaseline: R_learn = R_real - B(s) = {reward_for_q_update:.4f} ('{gain}')")

        except IndexError:
            logger.error(f"ShadowBaseline: IndexError Baseline '{gain}' {current_state_indices}. Shape: {baseline_table.shape}. Usando R_real.")
            reward_for_q_update = interval_reward
        except KeyError as e:
             logger.error(f"ShadowBaseline: KeyError '{gain}': {e}. Actions: {actions_dict}. Usando R_real.")
             reward_for_q_update = interval_reward
        except Exception as e:
            logger.error(f"ShadowBaseline: Error inesperado '{gain}': {e}. Usando R_real.", exc_info=True)
            reward_for_q_update = interval_reward

        # Asegurar valor final finito
        if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
             logger.warning(f"ShadowBaseline: R_learn final inválido ({reward_for_q_update}). Devolviendo 0.0.")
             return 0.0
        else:
             return float(reward_for_q_update)