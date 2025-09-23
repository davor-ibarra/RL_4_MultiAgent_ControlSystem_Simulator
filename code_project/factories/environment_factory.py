import logging
from typing import Dict, Any, Optional

# Import interfaces and base classes
from interfaces.environment import Environment
from interfaces.reward_function import RewardFunction
# No longer need RewardStrategy here directly

# Import specific environment type
from components.environments.pendulum_environment import PendulumEnvironment

# Import other factories needed to build components
from .system_factory import SystemFactory
from .controller_factory import ControllerFactory
from .agent_factory import AgentFactory # Use the agent factory

class EnvironmentFactory:
    """
    Factory class for creating simulation environment instances.
    Responsible for orchestrating the creation of the system, controller,
    and agent components required by the environment.
    """
    @staticmethod
    def create_environment(config: Dict[str, Any],
                           reward_function_instance: RewardFunction,
                           ) -> Environment:
        """
        Creates an environment instance based on the configuration.

        Args:
            config: The main configuration dictionary.
            reward_function_instance: The pre-created reward function instance.

        Returns:
            An instance of an Environment subclass.

        Raises:
            ValueError: If environment type is unknown or creation fails.
            RuntimeError: For unexpected errors during creation.
        """
        env_config = config.get('environment', {})
        env_type = env_config.get('type')
        logging.info(f"Attempting to create environment of type: {env_type}")

        if env_type == 'pendulum_environment':
            try:
                # --- Create Required Sub-Components ---
                system_cfg = env_config.get('system', {})
                controller_cfg = env_config.get('controller', {})
                agent_cfg = env_config.get('agent', {})
                reward_cfg = env_config.get('reward', {})
                # *** Get pid_adaptation config section ***
                pid_adapt_cfg = config.get('pid_adaptation', {})

                system = SystemFactory.create_system(system_cfg.get('type'), system_cfg.get('params', {}))
                controller = ControllerFactory.create_controller(controller_cfg.get('type'), controller_cfg.get('params', {}))

                # --- Prepare Agent Parameters ---
                agent_type = agent_cfg.get('type')
                # Use copy to avoid modifying original config dict during injection
                agent_params = agent_cfg.get('params', {}).copy()

                # Verify that reward_strategy_instance was injected by world_initializer
                if 'reward_strategy_instance' not in agent_params:
                    raise ValueError("'reward_strategy_instance' missing in agent_params. Check world_initializer.")

                # Inject shadow baseline params if needed by agent constructor
                reward_mode = reward_cfg.get('reward_mode')
                if reward_mode == 'shadow_baseline':
                    agent_params['shadow_baseline_params'] = reward_cfg.get('shadow_baseline_params')

                # *** Inject gain_step and variable_step from pid_adaptation config ***
                gain_step_value = pid_adapt_cfg.get('gain_step')
                variable_step_value = pid_adapt_cfg.get('variable_step')
                if gain_step_value is None or variable_step_value is None:
                     raise ValueError("Missing 'gain_step' or 'variable_step' in 'pid_adaptation' config section.")
                agent_params['gain_step'] = gain_step_value
                agent_params['variable_step'] = variable_step_value
                # ********************************************************************

                # --- Create Agent ---
                # Now agent_params contains everything PIDQLearningAgent expects (except reward_strategy)
                agent = AgentFactory.create_agent(agent_type, agent_params)

                # --- Get Environment Specific Parameters ---
                dt = env_config.get('dt')
                # gain_step and variable_step are now passed to the agent constructor,
                # but PendulumEnvironment *also* needs them directly.
                # This is slightly redundant but necessary based on current design.
                gain_step_for_env = pid_adapt_cfg.get('gain_step')
                variable_step_for_env = pid_adapt_cfg.get('variable_step')
                reset_gains = pid_adapt_cfg.get('reset_gains_each_episode')

                if dt is None or gain_step_for_env is None or variable_step_for_env is None or reset_gains is None:
                    raise ValueError("Missing required env parameters: dt, gain_step, variable_step, reset_gains")

                # --- Create Environment Instance ---
                environment = PendulumEnvironment(
                    system=system,
                    controller=controller,
                    agent=agent,
                    reward_function=reward_function_instance,
                    dt=dt,
                    reset_gains=reset_gains,
                    config=config # Pass full config
                )

            except KeyError as e:
                logging.error(f"Missing config key for '{env_type}': {e}", exc_info=True)
                raise ValueError(f"Missing configuration for env '{env_type}'") from e
            except ValueError as e:
                logging.error(f"Config/creation error for '{env_type}': {e}", exc_info=True)
                raise
            except Exception as e:
                logging.error(f"Failed to create env '{env_type}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to create env '{env_type}'") from e

        else:
            raise ValueError(f"Unknown environment type specified: {env_type}")

        logging.info(f"Successfully created environment: {type(environment).__name__}")
        return environment