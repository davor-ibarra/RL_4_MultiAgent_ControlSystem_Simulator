from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ShadowBaselineRewardStrategy(RewardStrategy):
    """
    Estrategia Shadow Baseline (Delta Reward).
    
    Objetivo: Reducir la varianza del aprendizaje restando una línea base histórica.
    Fórmula: R_learn = R_t - beta * Baseline(s)
    Actualización: Baseline(s) <- Baseline(s) + alpha_b * (R_t - Baseline(s))
    
    Requiere: Agente con capacidad de tablas auxiliares ('baseline').
    """
    needs_virtual_simulation: bool = False
    required_auxiliary_tables: List[str] = ['baseline'] # Requiere tabla 'baseline' en agente

    def __init__(self,
                 beta: float = 0.8,
                 baseline_learning_rate: float = 0.1,
                 baseline_init_value: float = 0.0, 
                 **kwargs):
        self.beta_b = float(beta) # Factor de importancia del baseline (bias-variance trade-off)
        self.alpha_b = float(baseline_learning_rate)
        self.init_val_b = float(baseline_init_value)

    def compute_reward_for_learning(
        self, 
        gain_id: str,
        agent_instance: Dict[str, Any],
        controllers_dict: Dict[str, Any],
        current_agent_s_dict: Dict[str, Any],                           # Estado S
        current_s_indices: tuple,                                       # Índices de S para la tabla de 'gain_id'
        actions_taken_map: Dict[str, int],                              # Acciones A tomadas para todas las ganancias
        action_idx_for_gain: int,                                       # Acción específica para 'gain_id'
        real_interval_reward: float,
        avg_interval_stability_score: float,
        differential_rewards_map: Optional[Dict[str, float]] = None,    # No usado por Shadow
        reward_components: Optional[Dict] = None,
        **kwargs: Any
    ) -> float:
        # 1. Obtener R_t (Recompensa actual)
        # Se podría usar lógica de Credit Assignment aquí también si se combina,
        # por ahora usamos el reward global.
        r_t = float(real_interval_reward)
        if not np.isfinite(r_t): 
            r_t = 0.0

        # 2. Obtener Baseline B(s) actual desde la memoria del agente
        b_val = agent_instance.get_auxiliary_table_value('baseline', gain_id, current_s_indices)
        
        if b_val is None:
            b_val = self.init_val_b

        # 3. Calcular Recompensa Centrada (Advantage-like)
        # R_learn = R_t - beta * B_t
        r_learn = r_t - (self.beta_b * b_val)

        # 4. Actualizar Baseline (Aprendizaje en línea)
        # B_new = B_old + alpha * (R_t - B_old)
        b_error = r_t - b_val
        b_new = b_val + self.alpha_b * b_error
        
        # Guardar de vuelta en el agente
        agent_instance.update_auxiliary_table_value('baseline', gain_id, current_s_indices, b_new)
        # logger.debug(f"[ShadowBaseline:compute] Gain '{gain_id}': R_real={r_t:.3f}, B(S)={b_val:.3f} => R_learn={r_learn:.3f}. IsolationMet={gain_learned_was_maintained and other_gains_changed}")
        return float(r_learn)