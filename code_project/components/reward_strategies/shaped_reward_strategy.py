# components/reward_strategies/shaped_reward_strategy.py
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from components.reward_strategies.base_reward_strategy import BaseRewardStrategy
from interfaces.controller import Controller

logger = logging.getLogger(__name__)

class ShapedRewardStrategy(BaseRewardStrategy):
    """
    A reward strategy that implements Potential-Based Reward Shaping.
    
    It augments the base reward with a shaping term F:
    F = gamma * Phi(s') - Phi(s)
    
    Where Phi(s) is a potential function defined as a weighted sum of Lagrangian terms:
    Phi(s) = sum(weight_i * Term_i(s))
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.shaping_config = config.get('environment', {}).get('reward_setup', {}).get('reward_shaping', {})
        self.enabled = self.shaping_config.get('enabled', False)
        self.gamma = self.shaping_config.get('gamma', 0.99)
        self.scaling_factor = self.shaping_config.get('scaling_factor', 1.0) # Lambda
        
        # Weights for the potential function terms per agent/gain
        # Structure: { 'gain_name': { 'L_e': 1.0, 'L_x': 0.5, ... } }
        self.potential_weights = self.shaping_config.get('potential_weights', {})
        
        # Store last potentials to avoid re-calculation if possible, 
        # but calculating from state is more robust to resets/jumps.
        # We will calculate Phi(s) and Phi(s') dynamically.

        logger.info(f"[ShapedRewardStrategy] Initialized. Enabled: {self.enabled}, Gamma: {self.gamma}")

    def compute_reward_for_learning(self,
                                  gain_id: str,
                                  agent_instance: Any, # PIDQLearningAgent
                                  controllers_dict: Dict[str, Any],
                                  current_agent_state_dict: Dict[str, Any],
                                  current_state_indices: Tuple[int, ...],
                                  next_agent_state_dict: Dict[str, Any],
                                  actions_taken_map: Dict[str, int],
                                  action_idx_for_gain: int,
                                  real_interval_reward: float,
                                  avg_interval_stability_score: float,
                                  differential_rewards_map: Optional[Dict[str, float]] = None,
                                  reward_components: Optional[Dict[str, float]] = None
                                  ) -> float:
        
        # 1. Compute Base Reward (using BaseRewardStrategy logic)
        base_reward = super().compute_reward_for_learning(
            gain_id, agent_instance, controllers_dict, current_agent_state_dict,
            current_state_indices, next_agent_state_dict, actions_taken_map,
            action_idx_for_gain, real_interval_reward, avg_interval_stability_score,
            differential_rewards_map, reward_components
        )
        
        if not self.enabled:
            return base_reward
            
        # 2. Calculate Reward Shaping Term
        # We need Phi(s) and Phi(s')
        
        # Extract weights for this specific agent/gain
        # If no specific weights, maybe use default or skip
        agent_weights = self.potential_weights.get(gain_id, self.potential_weights.get('default', {}))
        
        if not agent_weights:
            return base_reward # No shaping defined for this agent

        # Calculate Potentials
        # Note: reward_components contains L terms for s' (next state) calculated by LagrangianRewardCalculator
        # But we need to be consistent. 
        # If we use reward_components for s', we rely on LagrangianRewardCalculator's definition.
        # For s, we don't have pre-calculated components.
        # So we should calculate both Phi(s) and Phi(s') using the same logic here to be safe.
        
        phi_s = self._calculate_potential(current_agent_state_dict, agent_weights)
        phi_s_prime = self._calculate_potential(next_agent_state_dict, agent_weights)
        
        # Shaping term F = gamma * Phi(s') - Phi(s)
        # User mentioned F_t = gamma * N_t * V_{t+1} - V_t. Assuming N_t=1 for now as standard PBRS.
        shaping_term = self.gamma * phi_s_prime - phi_s
        
        # Apply scaling
        total_shaping = self.scaling_factor * shaping_term
        
        shaped_reward = base_reward + total_shaping
        
        # Optional: Log shaping components if possible (requires agent support or logger)
        # logger.debug(f"Gain: {gain_id}, Base: {base_reward:.4f}, Phi(s): {phi_s:.4f}, Phi(s'): {phi_s_prime:.4f}, F: {shaping_term:.4f}, Total: {shaped_reward:.4f}")
        
        return shaped_reward

    def _calculate_potential(self, state_dict: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        Calculates the potential Phi(s) given a state dictionary and weights.
        It maps state variables to Lagrangian terms (L_e, L_x, etc.).
        """
        # We need to map state variables to the L-terms expected in weights.
        # This mapping should match LagrangianRewardCalculator's logic.
        # Ideally this mapping is centralized, but for now we duplicate/adapt it.
        
        # Standard mapping for CartPole/Pendulum
        variables = {
            'L_e': state_dict.get('pendulum_angle', 0.0),
            'L_edot': state_dict.get('pendulum_velocity', 0.0),
            'L_x': state_dict.get('cart_position', 0.0),
            'L_xdot': state_dict.get('cart_velocity', 0.0),
            # L_u and L_udot are action dependent. 
            # Standard PBRS is usually on state. If weights include L_u, we treat it as 0 or need action.
            # For now, we assume potential is state-only.
            'L_u': 0.0, 
            'L_udot': 0.0
        }
        
        potential = 0.0
        for term, weight in weights.items():
            val = variables.get(term, 0.0)
            # Potential is usually negative of cost (like reward), or positive "energy".
            # If weights are positive in config and represent "importance of error", 
            # then Potential should probably be -weight * error^2 (higher error = lower potential).
            # OR, if Potential represents "Energy" (Lyapunov), it might be positive, 
            # and we want to reach Minimum Energy.
            # PBRS: F = gamma * Phi(s') - Phi(s).
            # If we want to encourage going to s' with LOWER error:
            # If Phi = -Error, then Phi(s') > Phi(s) means Error(s') < Error(s). F > 0. Good.
            # If Phi = Error, then Phi(s') < Phi(s) means Error(s') < Error(s). F < 0. Bad?
            # Wait. F = gamma * Phi(s') - Phi(s).
            # If Phi is "Goodness", we want Phi(s') > Phi(s).
            # If Error decreases, Goodness should increase.
            # So Phi should be negative of Error (or inverse).
            # Let's assume weights are positive importance of errors.
            # We define Phi = - sum (weight * val^2).
            
            # However, if the user config defines specific signs, we should respect them.
            # But typically weights are magnitudes.
            # We will assume quadratic penalty form for potential: - w * x^2
            
            potential += - weight * (val ** 2)
            
        return potential
