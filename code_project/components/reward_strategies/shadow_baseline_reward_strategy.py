# components/reward_strategies/shadow_baseline_reward_strategy.py

from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, List
import logging
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
    from interfaces.controller import Controller

logger = logging.getLogger(__name__)

class ShadowBaselineRewardStrategy(RewardStrategy):
    # Atributos declarativos
    needs_virtual_simulation: bool = False
    required_auxiliary_tables: List[str] = ['baseline'] # Requiere una tabla 'baseline'

    def __init__(self, beta: float = 0.1, baseline_init_value: float = 0.0, **other_params: Any):
        # Se asume que beta y baseline_init_value son válidos (validados por config_loader/DI).
        # logger.info(f"[ShadowBaselineStrategy] Initialized with beta={beta}, baseline_init_value={baseline_init_value}")
        self.beta_update_rate = float(beta) # beta_baseline_update -> beta_update_rate
        self.baseline_default_init_val = float(baseline_init_value) # baseline_initial_value_config -> baseline_default_init_val
                                                                # (usado si B(S) es None)
        self.pid_gain_names: List[str] = ['kp', 'ki', 'kd'] # Asumimos estas ganancias para la lógica de aislamiento

        if other_params:
            logger.warning(f"[ShadowBaselineStrategy] Received unused parameters: {list(other_params.keys())}")

    def _check_if_gains_changed(self, prev_gain_val: Optional[float], current_gain_val: Optional[float]) -> bool:
        """Comprueba si una ganancia cambió, manejando NaNs."""
        if prev_gain_val is None or current_gain_val is None: return True # Si alguno es None, se asume cambio (o estado inicial)
        if pd.isna(prev_gain_val) and pd.isna(current_gain_val): return False
        if pd.isna(prev_gain_val) or pd.isna(current_gain_val): return True
        return not np.isclose(prev_gain_val, current_gain_val, rtol=1e-5, atol=1e-8)

    def compute_reward_for_learning(
        self, 
        gain_id: str, # La ganancia para la cual se está calculando R_learn (e.g., 'kp')
        agent_instance: 'RLAgent',
        controller_instance: 'Controller',
        current_agent_s_dict: Dict[str, Any], # Estado S
        current_s_indices: tuple, # Índices de S para la tabla de 'gain_id'
        actions_taken_map: Dict[str, int], # Acciones A tomadas para todas las ganancias
        action_idx_for_gain: int, # Acción específica para 'gain_id'
        real_interval_reward: float,
        avg_interval_stability_score: float,
        differential_rewards_map: Optional[Dict[str, float]], # No usado por Shadow
        **kwargs: Any
    ) -> float:
        
        # Asegurar que real_interval_reward y avg_interval_stability_score sean finitos
        r_real_eff = float(real_interval_reward) if pd.notna(real_interval_reward) and np.isfinite(real_interval_reward) else 0.0
        avg_stab_eff = float(avg_interval_stability_score) if pd.notna(avg_interval_stability_score) and np.isfinite(avg_interval_stability_score) else 1.0

        # 1. Obtener Baseline B(S, gain_id)
        # El agente devuelve None si la entrada no existe o la tabla no está.
        baseline_s_for_current_gain = agent_instance.get_auxiliary_table_value('baseline', gain_id, current_s_indices)
        if baseline_s_for_current_gain is None: # Si es la primera visita a este S para esta ganancia
            baseline_s_for_current_gain = self.baseline_default_init_val
            # logger.debug(f"[ShadowBaseline:compute] Gain '{gain_id}', S_indices {current_s_indices}: Baseline is None. Using default init: {self.baseline_default_init_val}")

        # 2. Calcular R_learn = R_real - B(S, gain_id)
        r_learn_val = r_real_eff - baseline_s_for_current_gain
        
        # 3. Condición de Aislamiento y Actualización del Baseline
        # La ganancia que se está aprendiendo (gain_id) DEBE haberse mantenido,
        # mientras que las OTRAS ganancias DEBEN haber cambiado.
        
        # Asumimos que el controlador tiene 'previous_kp', 'previous_ki', 'previous_kd'
        # Esto es un acoplamiento con PIDController. Una solución más genérica
        # requeriría que el agente o el SimulationManager pasen el estado anterior del controlador.
        current_ctrl_gains = controller_instance.get_params()
        
        gain_being_learned_current_val = current_ctrl_gains.get(gain_id)
        gain_being_learned_prev_val = getattr(controller_instance, f'previous_{gain_id}', None) # ej. controller.previous_kp

        other_gains_changed = True # Asumir que cambiaron si no hay otras ganancias
        if len(self.pid_gain_names) > 1:
            other_gains_changed = all(
                self._check_if_gains_changed(
                    getattr(controller_instance, f'previous_{other_g}', None),
                    current_ctrl_gains.get(other_g)
                )
                for other_g in self.pid_gain_names if other_g != gain_id
            )
            
        gain_learned_was_maintained = not self._check_if_gains_changed(gain_being_learned_prev_val, gain_being_learned_current_val)

        if gain_learned_was_maintained and other_gains_changed:
            # Actualizar B(S, gain_id)
            # Delta B = beta * w_stab * (R_real - B(S,gain)) = beta * w_stab * R_learn
            delta_b_update = self.beta_update_rate * avg_stab_eff * r_learn_val
            new_baseline_s_val = baseline_s_for_current_gain + delta_b_update
            
            # Asegurar que el nuevo baseline sea finito antes de actualizar
            if pd.notna(new_baseline_s_val) and np.isfinite(new_baseline_s_val):
                agent_instance.update_auxiliary_table_value('baseline', gain_id, current_s_indices, new_baseline_s_val)
            # else:
                # logger.warning(f"[ShadowBaseline:compute] Invalid new_baseline_s_val ({new_baseline_s_val}) for gain '{gain_id}'. Baseline not updated.")
        
        # logger.debug(f"[ShadowBaseline:compute] Gain '{gain_id}': R_real={r_real_eff:.3f}, B(S)={baseline_s_for_current_gain:.3f} => R_learn={r_learn_val:.3f}. IsolationMet={gain_learned_was_maintained and other_gains_changed}")
        return float(r_learn_val) if pd.notna(r_learn_val) and np.isfinite(r_learn_val) else 0.0