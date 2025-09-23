# components/reward_strategies/shadow_baseline_reward_strategy.py
from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, Union, List
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

logger = logging.getLogger(__name__) # Logger específico del módulo

class ShadowBaselineRewardStrategy(RewardStrategy):
    # Atributo declarativo para SimulationManager
    needs_virtual_simulation: bool = False
    required_auxiliary_tables: List[str] = ['baseline']

    def __init__(self, beta: float = 0.1, baseline_init_value: float = 0.0, **other_params: Any):
        # Parámetros vienen de config['...']['strategy_params']['shadow_baseline']
        if not isinstance(beta, (float, int)) or not (0 <= beta <= 1):
             msg = f"Beta ({beta}) for ShadowBaseline must be in [0, 1]."
             logger.critical(f"[ShadowBaselineStrategy] {msg}"); raise ValueError(msg)
        self.beta = float(beta)

        if not isinstance(baseline_init_value, (float, int)):
            msg = f"baseline_init_value ({baseline_init_value}) for ShadowBaseline must be numeric."
            logger.critical(f"[ShadowBaselineStrategy] {msg}"); raise TypeError(msg)
        # baseline_init_value se usa en PIDQLearningAgent para inicializar B(s), no directamente aquí.
        # La estrategia podría almacenarlo si necesitara influir en la inicialización.

        self.all_gains = ['kp', 'ki', 'kd']
        logger.info(f"[ShadowBaselineStrategy] Initialized with beta={self.beta}, baseline_init_value (for agent)={baseline_init_value}.")
        if other_params:
            logger.warning(f"[ShadowBaselineStrategy] Received unexpected parameters: {other_params.keys()}")


    def _values_changed(self, val_prev: float, val_curr: float) -> bool:
        if pd.isna(val_prev) and pd.isna(val_curr): return False
        if pd.isna(val_prev) or pd.isna(val_curr): return True
        return not np.isclose(val_prev, val_curr, rtol=1e-5, atol=1e-8)

    def compute_reward_for_learning(
        self, gain: str, agent: 'RLAgent', controller: 'Controller',
        current_agent_state_dict: Dict[str, Any], current_state_indices: tuple,
        actions_dict: Dict[str, int], action_taken_idx: int,
        interval_reward: float, avg_w_stab: float,
        reward_dict: Optional[Dict[str, float]], # No usado por Shadow, pero parte de la firma
        **kwargs
    ) -> float:
        reward_for_q_update: float = 0.0

        # Validar recompensa de intervalo
        if pd.isna(interval_reward) or not np.isfinite(interval_reward):
             logger.warning(f"[ShadowBaselineStrategy:compute_reward] Invalid interval_reward ({interval_reward}) for gain '{gain}'. Using 0.0 for R_learn.")
             return 0.0
        _interval_reward_float = float(interval_reward) # Asegurar que es float

        # Validar avg_w_stab
        _avg_w_stab = float(avg_w_stab) if pd.notna(avg_w_stab) and np.isfinite(avg_w_stab) else 1.0

        # --- Validar capacidades del agente y controlador ---
        # El agente debe poder obtener/actualizar tablas auxiliares.
        # La interfaz RLAgent ya define get_auxiliary_table_value y update_auxiliary_table_value.
        # La estrategia también debe poder obtener los nombres de las tablas auxiliares que el agente maneja,
        # y verificar si 'baseline' está entre ellas.
        agent_aux_tables = agent.get_auxiliary_table_names()
        if 'baseline' not in agent_aux_tables:
            logger.error(f"[ShadowBaselineStrategy:compute_reward] Agent does not manage a 'baseline' auxiliary table (managed: {agent_aux_tables}). Using R_real for gain '{gain}'.")
            return _interval_reward_float
        
        # El controlador debe tener prev_gains (esto ya se maneja bien por PIDController)
        if not (hasattr(controller, 'get_params') and all(hasattr(controller, f'prev_{g}') for g in self.all_gains)):
             logger.error(f"[ShadowBaselineStrategy:compute_reward] Controller missing prev_gain attributes for gain '{gain}'. Using R_real.")
             return _interval_reward_float

        if current_state_indices is None:
             logger.warning(f"[ShadowBaselineStrategy:compute_reward] State indices (S) are None for gain '{gain}'. Using R_real.")
             return _interval_reward_float

        try:
            current_gains = controller.get_params()
            k_g_curr = current_gains.get(gain)
            k_g_prev = getattr(controller, f'prev_{gain}')
            other_gains_list = [g_other for g_other in self.all_gains if g_other != gain]
            
            if not other_gains_list or len(other_gains_list) < 2: # Debería haber 2 otras ganancias
                logger.error(f"[ShadowBaselineStrategy:compute_reward] Logic error: Could not determine 2 other gains for base gain '{gain}'. Using R_real.")
                return _interval_reward_float

            k_other1_curr = current_gains.get(other_gains_list[0])
            k_other1_prev = getattr(controller, f'prev_{other_gains_list[0]}')
            k_other2_curr = current_gains.get(other_gains_list[1])
            k_other2_prev = getattr(controller, f'prev_{other_gains_list[1]}')

            all_k_vals = [k_g_prev, k_g_curr, k_other1_prev, k_other1_curr, k_other2_prev, k_other2_curr]
            if any(pd.isna(k) or k is None for k in all_k_vals): # Chequear None y NaN
                logger.warning(f"[ShadowBaselineStrategy:compute_reward] Could not retrieve all prev/curr gains as valid numbers for gain '{gain}'. Gains: {all_k_vals}. Using R_real.")
                return _interval_reward_float

            gain_maintained = not self._values_changed(k_g_prev, k_g_curr) # type: ignore
            other1_changed = self._values_changed(k_other1_prev, k_other1_curr) # type: ignore
            other2_changed = self._values_changed(k_other2_prev, k_other2_curr) # type: ignore
            isolation_condition_met = gain_maintained and other1_changed and other2_changed

            # Obtener B(S) usando el método genérico del agente
            baseline_s_g = agent.get_auxiliary_table_value(table_name='baseline', gain=gain, state_indices=current_state_indices)
            
            # Si no hay valor de baseline (e.g., estado nuevo), usar 0.0 o el valor de inicialización de la estrategia
            if baseline_s_g is None:
                # Usar self.baseline_init_value que se pasó al agente
                # o un default si el agente no lo expone (lo que sería un fallo de diseño).
                # El agente lo usa para inicializar la tabla, así que debería ser consistente.
                # Por ahora, asumamos que si es None aquí, es porque el estado no se ha visitado
                # o la tabla no se inicializó con el valor correcto. Default a 0.0 es seguro.
                # El agente PIDQLearningAgent debería inicializar la tabla 'baseline' con self.baseline_init_value.
                logger.debug(f"[ShadowBaselineStrategy:compute_reward] Baseline for gain '{gain}', state {current_state_indices} is None. Using 0.0 for B(s).")
                baseline_s_g = 0.0
            
            baseline_s_g = float(baseline_s_g) # Asegurar que es float

            reward_for_q_update = _interval_reward_float - baseline_s_g
            # logger.debug(f"[ShadowBaselineStrategy:compute_reward] Gain '{gain}': R_real={_interval_reward_float:.3f}, B(S)={baseline_s_g:.3f}, R_learn={reward_for_q_update:.3f}. Isolation: {isolation_condition_met}")


            if isolation_condition_met:
                delta_B = self.beta * _avg_w_stab * reward_for_q_update # reward_for_q_update es (R_real - B(s))
                new_baseline = baseline_s_g + delta_B
                
                if pd.notna(new_baseline) and np.isfinite(new_baseline):
                    # Actualizar B(S) usando el método genérico del agente
                    agent.update_auxiliary_table_value(table_name='baseline', gain=gain, state_indices=current_state_indices, value=new_baseline)
                    # logger.debug(f"[ShadowBaselineStrategy:compute_reward] Baseline table 'baseline' for gain '{gain}' updated at {current_state_indices} -> {new_baseline:.4f} (delta_B: {delta_B:.4f})")
                else: 
                    logger.warning(f"[ShadowBaselineStrategy:compute_reward] Invalid new B(s) ({new_baseline}) for gain '{gain}'. Not updated.")
        
        except Exception as e:
            logger.error(f"[ShadowBaselineStrategy:compute_reward] Error processing for gain '{gain}': {e}. Using R_real.", exc_info=True)
            reward_for_q_update = _interval_reward_float

        return float(reward_for_q_update) if np.isfinite(reward_for_q_update) else 0.0