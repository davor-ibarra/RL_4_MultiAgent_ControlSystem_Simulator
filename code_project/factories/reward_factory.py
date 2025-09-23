import logging
from components.rewards.gaussian_reward import GaussianReward
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
# --- NUEVO: Importar el calculador simple ---
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
# --- FIN NUEVO ---
from interfaces.stability_calculator import StabilityCalculator as StabilityCalculatorInterface

class RewardFactory:
    @staticmethod
    def create_reward_function(reward_config):
        """
        Creates the main reward function and ensures an appropriate stability
        calculator is available if needed by the reward_mode.
        """
        reward_type = reward_config.get('type')
        reward_params = reward_config.get('params', {})
        reward_mode = reward_config.get('reward_mode', 'global') # Needed for logic
        stability_config = reward_config.get('stability_calculator', {})
        shadow_params = reward_config.get('shadow_baseline_params', {}) # Needed for fallback params
        stability_calculator_instance = None

        # --- Attempt 1: Create Configured Stability Calculator (IRA) if Enabled ---
        if stability_config.get('enabled', False):
            stab_type = stability_config.get('type')
            stab_params = stability_config.get('params', {})
            logging.info(f"Attempting to create configured stability calculator: {stab_type}")
            try:
                if stab_type == 'ira_instantaneous':
                    stability_calculator_instance = IRAStabilityCalculator(stab_params)
                    logging.info("IRAStabilityCalculator created successfully.")
                # Add other configured types here
                else:
                    logging.warning(f"Configured stability calculator type '{stab_type}' unknown. None created.")
            except Exception as e:
                 logging.error(f"Failed to create configured stability calculator '{stab_type}': {e}", exc_info=True)
                 stability_calculator_instance = None # Ensure it's None on error

        # --- Attempt 2: Create Fallback Stability Calculator if Needed and Configured ---
        # If no calculator was created above AND the mode requires w_stab (Shadow)
        if stability_calculator_instance is None and reward_mode == 'shadow-baseline':
            logging.info("Configured stability calculator disabled or failed, but Shadow mode requires w_stab. Attempting fallback...")
            fallback_calc_config = shadow_params.get('w_stab_calculator_params')
            if fallback_calc_config:
                fallback_type = fallback_calc_config.get('type')
                fallback_params = fallback_calc_config # Pass the whole sub-dict
                try:
                    if fallback_type == 'simple_exponential':
                         stability_calculator_instance = SimpleExponentialStabilityCalculator(fallback_params)
                         logging.info("Created SimpleExponentialStabilityCalculator as fallback for w_stab.")
                    # Add other fallback types here if needed
                    else:
                         logging.warning(f"Fallback stability calculator type '{fallback_type}' unknown. No fallback created.")
                except Exception as e:
                    logging.error(f"Failed to create fallback stability calculator '{fallback_type}': {e}", exc_info=True)
                    stability_calculator_instance = None # Ensure None on error
            else:
                logging.warning("Shadow mode active, main stability calculator disabled/failed, and no 'w_stab_calculator_params' found in 'shadow_baseline_params'. w_stab will default to 1.0.")

        # --- Log Final Calculator Status ---
        if stability_calculator_instance:
             logging.info(f"Using stability calculator: {type(stability_calculator_instance).__name__}")
        elif reward_mode == 'shadow-baseline':
             logging.warning(f"Proceeding in Shadow Baseline mode WITHOUT a stability calculator. w_stab will default to 1.0.")
        else:
             logging.info("No stability calculator will be used.")


        # --- Create Main Reward Function (Pass the created calculator, if any) ---
        logging.info(f"Attempting to create main reward function: {reward_type}")
        try:
            if reward_type == 'gaussian':
                # Pass calculator instance (can be IRA, SimpleExponential, or None)
                return GaussianReward(
                    reward_config=reward_config,
                    stability_calculator=stability_calculator_instance
                )
            # Add other reward function types here
            raise ValueError(f"Reward function type '{reward_type}' not recognized.")

        except Exception as e:
             logging.error(f"Failed to create reward function '{reward_type}': {e}", exc_info=True)
             raise ValueError(f"Error creating reward function: {e}") from e