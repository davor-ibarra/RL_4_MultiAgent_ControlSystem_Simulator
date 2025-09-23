from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller # Importar para type hint

# 10.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class ShadowBaselineRewardStrategy(RewardStrategy): # Implementar Interfaz RewardStrategy
    """
    Estrategia Shadow Baseline. R_learn = R_real - B(s).
    Actualiza B(s) usando tasa beta si la ganancia se mantuvo y las otras cambiaron.
    Implementa RewardStrategy.
    """
    def __init__(self, beta: float = 0.1):
        """
        Inicializa la estrategia Shadow Baseline.

        Args:
            beta (float): Tasa de aprendizaje para actualizar el baseline B(s).

        Raises:
            ValueError: Si beta no está en [0, 1].
        """
        # 10.2: Validar beta (Fail-Fast en init)
        if not isinstance(beta, (float, int)) or not (0 <= beta <= 1):
             msg = f"Beta ({beta}) inválido para ShadowBaseline. Debe estar en [0, 1]."
             logger.critical(msg)
             raise ValueError(msg)
        self.beta = float(beta)
        self.all_gains = ['kp', 'ki', 'kd'] # Lista de todas las ganancias posibles
        logger.info(f"ShadowBaselineRewardStrategy inicializada con beta={self.beta}")

    def _values_changed(self, val_prev: float, val_curr: float) -> bool:
        """Comprueba si dos valores flotantes son significativamente diferentes."""
        # Retorna True si CAMBIARON (NO son cercanos)
        # Manejar NaN: si alguno es NaN, se considera cambio (a menos que ambos sean NaN)
        if pd.isna(val_prev) and pd.isna(val_curr): return False # Ambos NaN -> sin cambio
        if pd.isna(val_prev) or pd.isna(val_curr): return True  # Uno es NaN -> cambio
        # Si ambos son finitos, usar isclose
        return not np.isclose(val_prev, val_curr, rtol=1e-5, atol=1e-8)

    def compute_reward_for_learning(
        self,
        # --- Context ---
        gain: str,                          # Ganancia actual ('kp', 'ki', 'kd')
        agent: 'RLAgent',                   # Instancia del agente para acceder/actualizar B(s)
        controller: 'Controller',           # Instancia del controlador para obtener ganancias prev/curr
        # --- State ---
        current_agent_state_dict: Dict[str, Any], # Necesario para obtener/actualizar B(s)
        current_state_indices: tuple,             # Índices S para tabla B(s)
        # --- Action ---
        actions_dict: Dict[str, int],             # Ignorado (implícito en cambio de K)
        action_taken_idx: int,                    # Ignorado
        # --- Raw Reward/Stability ---
        interval_reward: float,                   # R_real acumulada en el intervalo
        avg_w_stab: float,                        # w_stab promedio en el intervalo
        # --- Pre-calculated Differential Rewards ---
        reward_dict: Optional[Dict[str, float]],  # Ignorado
        # --- Optional ---
        **kwargs
    ) -> float:
        """
        Calcula R_learn = R_real - B(s) y actualiza B(s) si la condición de aislamiento se cumple.
        """
        reward_for_q_update: float = 0.0

        # 10.3: Validar entradas numéricas clave (devolver 0 si R_real es inválido)
        if pd.isna(avg_w_stab) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0 # Default w_stab
        if pd.isna(interval_reward) or not np.isfinite(interval_reward):
             logger.warning(f"ShadowBaseline: interval_reward inválido ({interval_reward}). Usando 0.0 para R_learn.")
             return 0.0

        # 10.4: Validar que el agente y controlador tengan los métodos/atributos necesarios
        #       Usar hasattr para checks seguros.
        if not hasattr(agent, 'get_baseline_value_for_state') or not hasattr(agent, 'baseline_tables_np'):
             logger.error(f"ShadowBaseline: Agente {type(agent).__name__} sin métodos/atributos baseline. Usando R_real.")
             return interval_reward
        if not hasattr(controller, 'get_params') or not all(hasattr(controller, f'prev_{g}') for g in self.all_gains):
             logger.error(f"ShadowBaseline: Controlador {type(controller).__name__} sin get_params o atributos 'prev_k*'. Usando R_real.")
             return interval_reward

        baseline_tables = getattr(agent, 'baseline_tables_np', None)
        if not isinstance(baseline_tables, dict) or gain not in baseline_tables:
            logger.warning(f"ShadowBaseline: Tabla Baseline para '{gain}' no encontrada. Usando R_real.")
            return interval_reward
        baseline_table = baseline_tables[gain]

        # Validar índices de estado
        if current_state_indices is None:
             logger.warning(f"ShadowBaseline: Índices de estado (S) son None para '{gain}'. Usando R_real.")
             return interval_reward

        try:
            # --- Obtener valores de ganancias Previos y Actuales ---
            current_gains = controller.get_params()
            k_g_curr = current_gains.get(gain)
            k_g_prev = getattr(controller, f'prev_{gain}')

            other_gains = [g for g in self.all_gains if g != gain]
            k_other1_curr = current_gains.get(other_gains[0])
            k_other1_prev = getattr(controller, f'prev_{other_gains[0]}')
            k_other2_curr = current_gains.get(other_gains[1])
            k_other2_prev = getattr(controller, f'prev_{other_gains[1]}')

            # Validar que obtuvimos valores (pueden ser NaN si hubo error antes)
            all_k_values = [k_g_prev, k_g_curr, k_other1_prev, k_other1_curr, k_other2_prev, k_other2_curr]
            if any(k is None for k in all_k_values): # Check for None first
                logger.warning(f"ShadowBaseline: No se pudieron obtener todos los valores de ganancia (prev/curr). Usando R_real.")
                return interval_reward

            # --- Evaluar Condición de Aislamiento ---
            gain_maintained = not self._values_changed(k_g_prev, k_g_curr)
            other1_changed = self._values_changed(k_other1_prev, k_other1_curr)
            other2_changed = self._values_changed(k_other2_prev, k_other2_curr)
            isolation_condition_met = gain_maintained and other1_changed and other2_changed

            # --- Obtener B(s) ---
            # get_baseline_value_for_state devuelve dict {gain: value}
            baseline_value_dict = agent.get_baseline_value_for_state(current_agent_state_dict)
            baseline_value = baseline_value_dict.get(gain, np.nan) # Extraer valor para gain actual
            # Usar 0.0 como default si B(s) es NaN o no existe aún
            if pd.isna(baseline_value) or not np.isfinite(baseline_value):
                 baseline_value = 0.0

            # --- Calcular R_learn ---
            reward_for_q_update = interval_reward - baseline_value

            # --- Actualizar B(s) si la condición se cumple ---
            if isolation_condition_met:
                # Actualizar B(s) = B(s) + beta * w_stab * (R_real - B(s))
                # (R_real - B(s)) es precisamente reward_for_q_update calculado arriba
                delta_B = self.beta * avg_w_stab * reward_for_q_update
                new_baseline = baseline_value + delta_B

                # Actualizar tabla solo si el nuevo valor es finito
                if pd.notna(new_baseline) and np.isfinite(new_baseline):
                    try:
                        baseline_table[current_state_indices] = new_baseline
                        # logger.debug(f"ShadowBaseline: B(s) updated '{gain}' {current_state_indices} -> {new_baseline:.4f}")
                    except IndexError:
                        logger.error(f"ShadowBaseline: IndexError al actualizar B(s) para '{gain}' en {current_state_indices}. Shape: {baseline_table.shape}.")
                        # No actualizar B(s) pero R_learn ya está calculado como R_real - B(s)_old
                else:
                    logger.warning(f"ShadowBaseline: Nuevo B(s) inválido ({new_baseline}) calculado para '{gain}'. No se actualizó B.")
            # else: logger.debug(f"ShadowBaseline: Condición aislamiento no cumplida para '{gain}'. No se actualiza B(s). R_learn = R_real - B(s)")


        except AttributeError as e: # Error al acceder a prev_k* o baseline_tables_np
             logger.error(f"ShadowBaseline: Error de atributo para '{gain}': {e}. Usando R_real.", exc_info=True)
             reward_for_q_update = interval_reward # Usar R_real si falla B(s)
        except KeyError as e: # Error al acceder a current_gains
             logger.error(f"ShadowBaseline: KeyError obteniendo ganancias para '{gain}': {e}. Usando R_real.")
             reward_for_q_update = interval_reward
        except IndexError: # Error al acceder a baseline_table[indices]
             logger.error(f"ShadowBaseline: IndexError acceso tabla Baseline '{gain}' para índices {current_state_indices}. Shape: {baseline_table.shape}. Usando R_real.")
             reward_for_q_update = interval_reward # Usar R_real si falla B(s)
        except Exception as e: # Otros errores
            logger.error(f"ShadowBaseline: Error inesperado calculando R_learn para '{gain}': {e}. Usando R_real.", exc_info=True)
            reward_for_q_update = interval_reward # Usar R_real como fallback seguro

        # Devolver R_learn (asegurando que sea finito)
        return float(reward_for_q_update) if np.isfinite(reward_for_q_update) else 0.0