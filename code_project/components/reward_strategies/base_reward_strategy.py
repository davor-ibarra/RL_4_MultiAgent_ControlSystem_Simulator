# components/reward_strategies/base_reward_strategy.py
from interfaces.reward_strategy import RewardStrategy
from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BaseRewardStrategy(RewardStrategy):
    """
    Estrategia de Recompensa Base con asignación de crédito y escalado multiobjetivo.
    
    Características:
    1. **Multi-Objective Scalarization**: Combina componentes de recompensa según pesos globales.
    2. **Credit Assignment**: Distribuye recompensas a sub-agentes (ganancias) basándose en su influencia.
       - Soporta 'manual_partition': Pesos explícitos por ganancia y componente.
    3. **Global Fallback**: Usa la recompensa total si no hay configuración específica.
    """
    needs_virtual_simulation: bool = False      # No usa simulaciones virtuales
    required_auxiliary_tables: List[str] = []   # No usa tablas auxiliares del agente

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_setup = self.config.get('environment', {}).get('reward_setup', {})
        self.reward_config = self.reward_setup.get('reward_config', {})
        
        # Pre-cargar configuraciones para eficiencia
        self.reward_mode = self.reward_config.get('reward_mode', 'global')
        self.mo_config = self.reward_config.get('reward_multiobjective', {})
        self.credit_assign_params = self.reward_config.get('credit_assign_params', {})
        
        logger.info(f"[BaseRewardStrategy] Initialized. Mode: {self.reward_mode}")

    def compute_reward_for_learning(self, 
                                    gain_id: str,
                                    agent_instance: Any,
                                    controllers_dict: Dict[str, Any],
                                    current_agent_state_dict: Dict[str, Any],
                                    current_state_indices: tuple,
                                    next_agent_state_dict: Dict[str, Any],
                                    actions_taken_map: Dict[str, int],
                                    action_idx_for_gain: int,
                                    real_interval_reward: float,
                                    avg_interval_stability_score: float,
                                    differential_rewards_map: Optional[Dict[str, float]] = None,
                                    reward_components: Optional[Dict[str, float]] = None
                                    ) -> float:
        """
        Calcula la recompensa final de aprendizaje (R_learn) para una ganancia específica.
        """
        
        # 1. Si no hay componentes detallados, usar el reward global directo
        if reward_components is None:
            return real_interval_reward

        # 2. Escalamiento Multi-Objetivo Global (Scalarization)
        # Si está habilitado, transformamos los componentes crudos antes de asignar crédito
        processed_components = self._apply_multiobjective_scaling(reward_components)

        # 3. Asignación de Crédito (Credit Assignment)
        if self.reward_mode == 'credit_assign':
            return self._compute_credit_assignment(gain_id, processed_components, real_interval_reward)
        
        # 4. Modo Global (Default)
        # Si no es credit assignment, devolvemos la suma ponderada global (o el total pre-calculado)
        # Si hubo escalado multiobjetivo, usamos la suma de los procesados.
        if self.mo_config.get('enabled', False):
            return sum(processed_components.values())
        
        return real_interval_reward

    def _apply_multiobjective_scaling(self, components: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica pesos y transformaciones globales a los componentes de recompensa.
        """
        if not self.mo_config.get('enabled', False):
            return components.copy()

        weights = self.mo_config.get('weights', {})
        scaled_components = {}
        
        for key, val in components.items():
            if key == 'total_reward': continue # Ignorar el total agregado
            
            # Aplicar peso si existe, default 1.0
            w = float(weights.get(key, 1.0))
            scaled_components[key] = val * w
            
        return scaled_components

    def _compute_credit_assignment(self, gain_id: str, components: Dict[str, float], global_fallback: float) -> float:
        """
        Calcula la recompensa específica para un 'gain_id' basada en la estrategia configurada.
        """
        strategy_cfg = self.credit_assign_params.get('strategy', {})
        
        # --- Estrategia: Partición Manual ---
        manual_partition = strategy_cfg.get('manual_partition', {})
        if manual_partition.get('enabled', False):
            feature_partitions = manual_partition.get('feature_partitions', {})
            
            # Buscar configuración para este gain_id específico
            # gain_id ej: 'kp_pendulum_angle'
            agent_weights = feature_partitions.get(gain_id)
            
            if not agent_weights:
                # Intentar buscar por tipo de ganancia (ej: 'kp', 'ki') si no hay específico
                gain_type = gain_id.split('_')[0]
                agent_weights = feature_partitions.get(gain_type)

            if agent_weights:
                weighted_sum = 0.0
                for feature_key, weight in agent_weights.items():
                    val = components.get(feature_key, 0.0)
                    weighted_sum += val * float(weight)
                return weighted_sum
            
            # Si no hay pesos definidos, fallback al global
            return global_fallback

        # --- Estrategia: Orthogonalization (Placeholder) ---
        orthogonalization = strategy_cfg.get('orthogonalization', {})
        if orthogonalization.get('enabled', False):
            # Aquí iría la lógica de proyección ortogonal
            # Por ahora, fallback al global
            return global_fallback

        return global_fallback