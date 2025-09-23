from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
import numpy as np
import pandas as pd # Para isnull/isna

# Evitar importación circular
if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class ShadowBaselineRewardStrategy(RewardStrategy):
    """
    Estrategia de Recompensa Shadow Baseline.
    Calcula la recompensa para el aprendizaje como R_real - B(s), donde B(s) es
    un baseline aprendido para cada estado.
    Actualiza B(s) solo cuando la acción para la ganancia específica es 'maintain'
    Y las acciones para las OTRAS ganancias NO son 'maintain'.
    La actualización de B(s) usa un factor beta y la puntuación de estabilidad w_stab.
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
        agent: 'PIDQLearningAgent',         # Instancia del agente para acceder a B(s)
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
        Calcula R_learn para Q-update y actualiza B(s) si se cumplen las condiciones.
        Si acción para 'gain' es 'maintain' y las otras no -> Actualiza B(s), R_learn = R_real
        En otro caso -> No actualiza B(s), R_learn = R_real - B(s)
        """
        reward_for_q_update: float = 0.0 # Valor por defecto

        # Validar entradas necesarias
        if not isinstance(avg_w_stab, (float, int)) or not np.isfinite(avg_w_stab):
             logger.warning(f"ShadowBaseline: avg_w_stab inválido ({avg_w_stab}) para ganancia '{gain}'. Usando 1.0.")
             avg_w_stab = 1.0
        if not isinstance(interval_reward, (float, int)) or not np.isfinite(interval_reward):
             logger.warning(f"ShadowBaseline: interval_reward inválido ({interval_reward}) para ganancia '{gain}'. Usando 0.0.")
             # Si R_real es inválido, ¿qué hacer? Devolver 0 puede ser lo más seguro.
             return 0.0

        # Comprobar si existe tabla Baseline para esta ganancia en el agente
        if gain not in agent.baseline_tables_np:
            logger.warning(f"ShadowBaseline: Tabla Baseline para ganancia '{gain}' no encontrada en agente. Usando R_real ({interval_reward:.4f}) como recompensa.")
            return interval_reward

        try:
            baseline_table = agent.baseline_tables_np[gain]
            # Obtener valor actual de B(s)
            baseline_value = float(baseline_table[current_state_indices])
            # Validar baseline_value (puede ser NaN si tabla se corrompió?)
            if pd.isna(baseline_value) or not np.isfinite(baseline_value):
                 logger.warning(f"ShadowBaseline: Valor B(s) inválido ({baseline_value}) para ganancia '{gain}', estado {current_state_indices}. Usando B(s)=0.")
                 baseline_value = 0.0

            # --- Determinar si se debe actualizar B(s) ---
            update_baseline = False
            if action_taken_idx == 1: # Acción para 'gain' es 'maintain'
                # Verificar acciones de las OTRAS ganancias
                other_gains = [g for g in self.all_gains if g != gain]
                is_isolated_maintain = True # Asumir que sí hasta encontrar lo contrario
                for other_gain in other_gains:
                    other_action = actions_dict.get(other_gain)
                    if other_action == 1: # Si OTRA acción también es 'maintain'
                        is_isolated_maintain = False
                        break
                    elif other_action is None: # Acción faltante
                         logger.warning(f"ShadowBaseline: Acción faltante para ganancia '{other_gain}' en actions_dict: {actions_dict}. No se puede confirmar aislamiento.")
                         is_isolated_maintain = False
                         break

                if is_isolated_maintain:
                    update_baseline = True # Condición cumplida

            # --- Calcular R_learn y Actualizar B(s) (si aplica) ---
            if update_baseline:
                # Actualizar B(s) usando R_real
                # delta_B = beta * w_stab * (R_real - B(s))
                delta_B = self.beta * avg_w_stab * (interval_reward - baseline_value)
                new_baseline = baseline_value + delta_B

                # Actualizar tabla B(s) (asegurando que no sea NaN/inf)
                if pd.notna(new_baseline) and np.isfinite(new_baseline):
                     baseline_table[current_state_indices] = new_baseline
                     # logger.debug(f"ShadowBaseline: B(s) actualizado para '{gain}' {current_state_indices}. "
                     #              f"B_prev={baseline_value:.4f}, R_real={interval_reward:.4f}, w_stab={avg_w_stab:.4f}, delta={delta_B:.4f} -> B_new={new_baseline:.4f}")
                else:
                     logger.warning(f"ShadowBaseline: Nuevo valor B(s) inválido ({new_baseline}) para '{gain}'. No se actualizó la tabla.")

                # Para el Q-update en este caso, usar R_real directamente
                reward_for_q_update = interval_reward
                # logger.debug(f"ShadowBaseline: Usando R_learn = R_real = {reward_for_q_update:.4f} para '{gain}' (B(s) actualizado).")

            else:
                # No actualizar B(s), usar R_learn = R_real - B(s)
                reward_for_q_update = interval_reward - baseline_value
                # logger.debug(f"ShadowBaseline: Usando R_learn = R_real - B(s) = {interval_reward:.4f} - {baseline_value:.4f} = {reward_for_q_update:.4f} para '{gain}'.")

        except IndexError:
            logger.error(f"ShadowBaseline: IndexError accediendo a tabla Baseline '{gain}' para índices {current_state_indices}. Shape: {baseline_table.shape}. Usando R_real.")
            reward_for_q_update = interval_reward
        except KeyError as e:
             logger.error(f"ShadowBaseline: KeyError accediendo a actions_dict para determinar aislamiento para ganancia '{gain}': {e}. Actions: {actions_dict}. Usando R_real.")
             reward_for_q_update = interval_reward
        except Exception as e:
            logger.error(f"ShadowBaseline: Error inesperado calculando recompensa/actualizando baseline para ganancia '{gain}': {e}. Usando R_real.", exc_info=True)
            reward_for_q_update = interval_reward

        # Asegurar que la recompensa final no sea NaN/inf
        if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
             logger.warning(f"ShadowBaseline: R_learn final es inválido ({reward_for_q_update}). Devolviendo 0.0.")
             return 0.0
        else:
             return float(reward_for_q_update)
