from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, TYPE_CHECKING, List
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

    def __init__(self, 
                 agent_defining_vars: List[str], 
                 beta: float = 0.1, 
                 baseline_init_value: float = 0.0, 
                 **other_params: Any):
        # logger.info(f"[ShadowBaselineStrategy] Initialized with beta={beta}, for agent vars: {agent_defining_vars}")
        self.beta_update_rate = float(beta)
        self.baseline_default_init_val = float(baseline_init_value) 
        self.agent_controlled_gains = agent_defining_vars # Store the gains this agent instance controls
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
        controllers_dict: Dict[str, 'Controller'],
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
        baseline_s = agent_instance.get_auxiliary_table_value('baseline', gain_id, current_s_indices)
        if baseline_s is None: # Si es la primera visita a este S para esta ganancia
            baseline_s = self.baseline_default_init_val
            # logger.debug(f"[ShadowBaseline:compute] Gain '{gain_id}', S_indices {current_s_indices}: Baseline is None. Using default init: {self.baseline_default_init_val}")

        # 2. Calcular R_learn = R_real - B(S, gain_id)
        r_learn_val = r_real_eff - baseline_s
        
        # 3. Condición de Aislamiento y Actualización del Baseline
        # La ganancia que se está aprendiendo (gain_id) DEBE haberse mantenido, mientras que las OTRAS ganancias DEBEN haber cambiado.
        
        # Asumimos que el controlador tiene 'previous_kp', 'previous_ki', 'previous_kd'
        # Esto es un acoplamiento con PIDController. Una solución más genérica
        # requeriría que el agente o el SimulationManager pasen el estado anterior del controlador.
        current_ctrl_gains = controllers_dict.get_params()
        
        gain_learned_current_val = current_ctrl_gains.get(gain_id)
        gain_learned_prev_val = getattr(controllers_dict, f'previous_{gain_id}', None) # ej. controller.previous_kp
        gain_learned_was_maintained = not self._check_if_gains_changed(gain_learned_prev_val, gain_learned_current_val)

        other_gains_changed = True # Asumir que cambiaron si no hay otras ganancias
        other_gains = [g for g in self.agent_controlled_gains if g != gain_id]
        if other_gains:
            other_gains_changed = all(
                self._check_if_gains_changed(
                    getattr(controllers_dict, f'previous_{other_g}', None),
                    current_ctrl_gains.get(other_g)
                ) for other_g in other_gains
            )

        if gain_learned_was_maintained and other_gains_changed:
            # Actualizar B(S, gain_id)
            # Delta B = beta * w_stab * (R_real - B(S,gain)) = beta * w_stab * R_learn
            delta_b = self.beta_update_rate * avg_stab_eff * r_learn_val
            new_baseline_val = baseline_s + delta_b
            # Asegurar que el nuevo baseline sea finito antes de actualizar
            if pd.notna(new_baseline_val) and np.isfinite(new_baseline_val):
                agent_instance.update_auxiliary_table_value('baseline', gain_id, current_s_indices, new_baseline_val)
            # else:
                # logger.warning(f"[ShadowBaseline:compute] Invalid new_baseline_s_val ({new_baseline_s_val}) for gain '{gain_id}'. Baseline not updated.")
        
        # logger.debug(f"[ShadowBaseline:compute] Gain '{gain_id}': R_real={r_real_eff:.3f}, B(S)={baseline_s_for_current_gain:.3f} => R_learn={r_learn_val:.3f}. IsolationMet={gain_learned_was_maintained and other_gains_changed}")
        return float(r_learn_val) if pd.notna(r_learn_val) and np.isfinite(r_learn_val) else 0.0