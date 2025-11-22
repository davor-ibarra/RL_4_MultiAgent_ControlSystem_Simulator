# components/rewards/lagrangian_reward_calculator.py

import logging
from typing import Dict, Any, Optional

from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator
from components.metrics_processing.metric_processing import MetricProcessing

logger = logging.getLogger(__name__)

class LagrangianRewardCalculator(RewardFunction):
    """Calculates rewards based on a Lagrangian‑like formulation.

    The calculator expects pre‑computed metric values (Lagrangian terms) to be
    supplied via the optional ``reward_components`` argument. This makes the
    reward function agnostic to the environment's internal state representation.
    """

    def __init__(self, config: Dict[str, Any], stability_calculator: BaseStabilityCalculator):
        logger.info("[LagrangianRewardCalculator] Initializing...")
        self.config = config
        self.stability_calculator = stability_calculator

        # Extract specific config
        reward_setup = config.get('environment', {}).get('reward_setup', {})
        self.calc_config = reward_setup.get('reward_calculation', {}).get('reward_low_observability', {})

        if not self.calc_config.get('enabled', False):
            logger.warning(
                "[LagrangianRewardCalculator] Initialized but 'reward_low_observability' is disabled in config."
            )

        self.reward_params = self.calc_config.get('reward_params', {})
        self.base_features_cfg = self.reward_params.get('reward_base', {}).get('features_base', {})
        self.extra_features_cfg = self.reward_params.get('reward_base', {}).get('features_extra', {})

        # Metric processing placeholder (may be used later)
        self.metric_processing = MetricProcessing(config)
        
        self.last_action = 0.0
        self.log_reward_params: Dict[str, float] = {}

    def calculate(
        self,
        state_dict: Dict[str, Any],
        action_a: Any,
        next_state_dict: Dict[str, Any],
        current_episode_time_sec: float,
        dt_sec: float,
        goal_reached_in_step: bool,
        reward_components: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Calculate the instantaneous reward.

        ``reward_components`` should contain the Lagrangian terms (e.g., ``L_e``,
        ``L_edot``) computed by the environment. If it is ``None`` an empty
        dictionary is used, resulting in a reward of zero.
        """
        self.log_reward_params.clear()

        variables = reward_components if reward_components is not None else {}

        total_reward = 0.0
        total_reward += self._compute_terms(self.base_features_cfg, variables)
        total_reward += self._compute_terms(self.extra_features_cfg, variables)

        # Update state for potential derivative calculations
        self.last_action = float(action_a)

        self.log_reward_params['total_reward'] = total_reward
        return self.log_reward_params.copy()

    def _compute_terms(self, features_cfg: Dict[str, Any], variables: Dict[str, float]) -> float:
        sub_total = 0.0
        for name, params in features_cfg.items():
            if name not in variables:
                continue
            val = variables[name]
            weight = params.get('weight', 0.0)
            sigma_ref = params.get('sigma_ref', 1.0)
            agg_type = params.get('agg', 'mean_sq')

            norm_val = val / sigma_ref if sigma_ref != 0 else val
            if agg_type in ('mean_sq', 'sq'):
                term_val = -weight * (norm_val ** 2)
            elif agg_type == 'abs':
                term_val = -weight * abs(norm_val)
            else:
                term_val = -weight * (norm_val ** 2)

            sub_total += term_val
            self.log_reward_params[name] = term_val
        return sub_total

    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        # Placeholder for future metric processing updates
        pass

    def reset(self):
        self.last_action = 0.0
        self.metric_processing.reset()

    def get_params_log(self) -> Dict[str, Any]:
        return self.log_reward_params