# components/reward_strategies/shadow_baseline_reward_strategy.py
from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

# Evitar importación circular
if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent # Usar interfaz genérica}
    from interfaces.controller import Controller

logger = logging.getLogger(__name__)

class ShadowBaselineRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia de Recompensa Shadow Baseline.
    Calcula R_learn = R_real - B(s) y actualiza B(s) si la ganancia 'gain'
    NO cambió su valor numérico y las otras DOS SÍ cambiaron su valor.
    """
    def __init__(self, beta: float = 0.1):
        """
        Inicializa la estrategia Shadow Baseline.

        Args:
            beta (float): Tasa de aprendizaje para actualizar el baseline B(s).
        """
        if not isinstance(beta, (float, int)) or not (0 <= beta <= 1):
             logger.warning(f"Beta ({beta}) inválido para ShadowBaseline. Usando 0.1.")
             self.beta = 0.1
        else:
             self.beta = beta
        self.all_gains = ['kp', 'ki', 'kd'] # Lista de todas las ganancias posibles
        logger.info(f"ShadowBaselineRewardStrategy inicializada con beta={self.beta}")
    
    def _values_changed(self, val_prev: float, val_curr: float) -> bool:
        """Comprueba si dos valores flotantes son diferentes usando np.isclose."""
        # Retorna True si CAMBIARON (NO son cercanos)
        return not np.isclose(val_prev, val_curr, rtol=1e-5, atol=1e-8)

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ganancia actual ('kp', 'ki', 'kd')
        agent: 'RLAgent',                   # Instancia del agente para acceder/actualizar B(s)
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Necesario para obtener valor B(s)
        current_state_indices: tuple,             # Índices S para actualizar tabla B(s)
        # --- Action ---
        actions_dict: Dict[str, int],             # Acciones {'kp':a_kp, ...} tomadas
        action_taken_idx: int,                    # Acción a_g para la ganancia 'gain'
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real acumulada en el intervalo
        avg_w_stab: float,                        # w_stab promedio en el intervalo
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # Ignorado
        # --- Controller for gains values ---
        controller: 'Controller',
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Calcula R_learn y actualiza B(s) si corresponde. Implementa método de interfaz.
        """
        reward_for_q_update: float = 0.0

        # Validar entradas numéricas clave
        if not isinstance(avg_w_stab, (float, int)) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0
        if not isinstance(interval_reward, (float, int)) or not np.isfinite(interval_reward):
             logger.warning(f"ShadowBaseline: interval_reward inválido ({interval_reward}). Usando 0.0 para R_learn."); return 0.0

        # Validar que el agente tenga lo necesario (método y atributo de tabla)
        if not hasattr(agent, 'get_baseline_value_for_state') or not hasattr(agent, 'baseline_tables_np'):
             logger.error(f"ShadowBaseline: Agente {type(agent).__name__} no tiene métodos/atributos necesarios para baseline. Usando R_real como R_learn.")
             return interval_reward

        baseline_tables = getattr(agent, 'baseline_tables_np', None)
        if not isinstance(baseline_tables, dict) or gain not in baseline_tables:
            logger.warning(f"ShadowBaseline: Tabla Baseline para '{gain}' no encontrada en agente. Usando R_real como R_learn.")
            return interval_reward
        
        if not hasattr(controller, 'get_params') or not hasattr(controller, 'prev_kp') or not hasattr(controller, 'prev_ki') or not hasattr(controller, 'prev_kd'):
             logger.error(f"ShadowBaseline: Controlador {type(controller).__name__} no tiene atributos 'prev_k' necesarios. Usando R_real.")
             return interval_reward

        baseline_table = baseline_tables[gain] # Tabla NumPy para la ganancia actual

        try:
            # --- Obtener valores de ganancias PREVIOS y ACTUALES --- <<<--- MODIFICADO
            current_gains = controller.get_params()
            k_g_curr = current_gains.get(gain, np.nan)
            k_g_prev = getattr(controller, f'prev_{gain}', np.nan)

            other_gains = [g for g in self.all_gains if g != gain]
            k_other1_curr = current_gains.get(other_gains[0], np.nan)
            k_other1_prev = getattr(controller, f'prev_{other_gains[0]}', np.nan)
            k_other2_curr = current_gains.get(other_gains[1], np.nan)
            k_other2_prev = getattr(controller, f'prev_{other_gains[1]}', np.nan)

            # Validar que obtuvimos valores numéricos
            all_k_values = [k_g_prev, k_g_curr, k_other1_prev, k_other1_curr, k_other2_prev, k_other2_curr]
            if any(pd.isna(k) or not np.isfinite(k) for k in all_k_values):
                logger.warning(f"ShadowBaseline: No se pudieron obtener valores de ganancia válidos (prev/curr). Prev: ({k_g_prev}, {k_other1_prev}, {k_other2_prev}), Curr: ({k_g_curr}, {k_other1_curr}, {k_other2_curr}). Usando R_real.")
                return interval_reward

            # --- Evaluar Condición de Aislamiento --- <<<--- MODIFICADO
            gain_maintained = not self._values_changed(k_g_prev, k_g_curr)
            other1_changed = self._values_changed(k_other1_prev, k_other1_curr)
            other2_changed = self._values_changed(k_other2_prev, k_other2_curr)

            isolation_condition_met = gain_maintained and other1_changed and other2_changed
            # --- FIN Evaluar Condición de Aislamiento ---

            # --- Obtener B(s) (sin cambios) ---
            baseline_value = agent.get_baseline_value_for_state(current_agent_state_dict).get(gain, np.nan)
            if pd.isna(baseline_value) or not np.isfinite(baseline_value):
                 baseline_value = 0.0

            # --- Calcular R_learn y Actualizar B(s) --- <<<--- MODIFICADO
            if isolation_condition_met:
                # Condición cumplida: usar recompensa diferencial y actualizar B(s)
                reward_for_q_update = interval_reward - baseline_value

                # Calcular y actualizar B(s)
                delta_B = self.beta * avg_w_stab * (interval_reward - baseline_value)
                new_baseline = baseline_value + delta_B

                if pd.notna(new_baseline) and np.isfinite(new_baseline):
                    if current_state_indices is not None and all(idx >= 0 for idx in current_state_indices):
                        try: # Añadir try-except aquí por si acaso
                             baseline_table[current_state_indices] = new_baseline
                             # logger.debug(f"ShadowBaseline (ISOLATION MET): B(s) updated '{gain}' {current_state_indices} -> {new_baseline:.4f} (R_real={interval_reward:.4f}, w_stab={avg_w_stab:.4f})")
                             # logger.debug(f"ShadowBaseline (ISOLATION MET): R_learn = R_real - B(s) = {interval_reward:.4f} - {baseline_value:.4f} = {reward_for_q_update:.4f} for '{gain}'")
                        except IndexError:
                             logger.error(f"ShadowBaseline: IndexError al actualizar B(s) para '{gain}' en {current_state_indices}. Shape: {baseline_table.shape}.")
                    else:
                         logger.warning(f"ShadowBaseline: Índices inválidos ({current_state_indices}) al intentar actualizar B(s) para '{gain}'.")
                else:
                     logger.warning(f"ShadowBaseline: Nuevo B(s) inválido ({new_baseline}) calculado para '{gain}'. No se actualizó.")

            else:
                # Condición NO cumplida: usar recompensa global y NO actualizar B(s)
                reward_for_q_update = interval_reward
                # logger.debug(f"ShadowBaseline (ISOLATION NOT MET): R_learn = R_real = {reward_for_q_update:.4f} for '{gain}'")
            # --- FIN Calcular R_learn y Actualizar B(s) ---

        # --- Bloque except (sin cambios) ---
        except IndexError:
            logger.error(f"ShadowBaseline: IndexError acceso tabla Baseline '{gain}' para índices {current_state_indices}. Shape: {baseline_table.shape}. Usando R_real.")
            reward_for_q_update = interval_reward
        except KeyError as e:
             logger.error(f"ShadowBaseline: KeyError '{gain}': {e}. Actions: {actions_dict}. Usando R_real.")
             reward_for_q_update = interval_reward
        except Exception as e:
            logger.error(f"ShadowBaseline: Error inesperado calculando R_learn para '{gain}': {e}. Usando R_real.", exc_info=True)
            reward_for_q_update = interval_reward

        # --- Devolución final (sin cambios) ---
        if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
             logger.warning(f"ShadowBaseline: R_learn final inválido ({reward_for_q_update}). Devolviendo 0.0.")
             return 0.0
        else:
             return float(reward_for_q_update)