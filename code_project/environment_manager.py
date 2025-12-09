import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

from interfaces.environment import Environment
from components.metrics_processing.metric_processing import MetricProcessing

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """
    Orchestrates the physical stepping of multiple environments.
    Aggregates states, metrics, and flags from all sub-environments.
    """
    def __init__(self, 
                 env_map: Dict[str, Environment], 
                 config: Dict[str, Any],
                 metric_processing: Optional[MetricProcessing] = None):
        """
        Args:
            env_map: Dictionary mapping env_id to Environment instances.
            config: Main configuration dictionary.
            metric_processing: Optional MetricProcessing instance for shared transformations.
        """
        self.env_map = env_map
        self.config = config
        self.metric_processing = metric_processing
        self.logger = logger
        self.logger.info(f"[EnvironmentManager] Initialized with {len(self.env_map)} environments: {list(self.env_map.keys())}")

    def reset_all(self, initial_conditions_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Resets all environments.
        
        Args:
            initial_conditions_map: Optional dict {env_id: initial_conditions}. 
                                    If None, environments use their internal defaults.
        
        Returns:
            Dict containing initial states for all environments, keyed by env_id.
        """
        initial_conditions_map = initial_conditions_map or {}
        reset_responses = {}
        
        for env_id, env in self.env_map.items():
            init_cond = initial_conditions_map.get(env_id)
            # Environment.reset return type depends on implementation, usually raw state or dict
            # We trust the environment to return what's expected by the agent/sim manager
            state = env.reset(init_cond)
            reset_responses[env_id] = state
            
        return reset_responses

    def step_all(self, current_sim_time: float, dt_dict: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Advances all environments by one step.
        
        Args:
            current_sim_time: Current global simulation time.
            dt_dict: Optional dict {env_id: dt} if environments have different time steps 
                     (though typically they should align or be managed by SimMan). 
                     If None, EnvManager relies on env's internal dt.

        Returns:
            Consolidated dictionary:
            {
                env_id: {
                    "state_dict": ...,
                    "stab_info": ...,
                    "subenv_context": {
                        "done_flags": ...,
                        "raw_metrics": ...,
                        ...
                    },
                    "metric_processing": ... (optional)
                }
            }
        """
        step_results = {}
        
        for env_id, env in self.env_map.items():
            # Step the environment
            # Expected return from refactored env.step():
            # (state_dict, stab_info, subenv_context)
            # Note: subenv_context should contain 'done_flags' and 'raw_metrics'
            try:
                state_dict, stab_info, subenv_context = env.step()
                
                # Check for metric processing at step level
                processed_metrics = {}
                if self.metric_processing:
                     # Merge state and raw metrics for processing
                     metrics_to_process = state_dict.copy()
                     if 'raw_metrics' in subenv_context and isinstance(subenv_context['raw_metrics'], dict):
                         metrics_to_process.update(subenv_context['raw_metrics'])
                     
                     processed_metrics = self.metric_processing.process_step_metrics(env_id, metrics_to_process)

                env_result = {
                    "state_dict": state_dict,
                    "stab_info": stab_info,
                    "metric_processing": processed_metrics,
                    "subenv_context": subenv_context
                }
                step_results[env_id] = env_result

            except Exception as e:
                self.logger.error(f"[EnvironmentManager] Error stepping environment '{env_id}': {e}", exc_info=True)
                raise

        return step_results
