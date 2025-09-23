# factories/environment_factory.py
import logging
from typing import Dict, Any, Optional
from interfaces.environment import Environment
from interfaces.reward_function import RewardFunction
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent
from components.environments.pendulum_environment import PendulumEnvironment

logger = logging.getLogger(__name__) # Logger específico del módulo

class EnvironmentFactory:
    def __init__(self):
        logger.info("[EnvironmentFactory] Instance created.")

    def create_environment(self,
                           config: Dict[str, Any], # Config completa
                           reward_function_instance: RewardFunction,
                           system_instance: DynamicSystem,
                           controller_instance: Controller,
                           agent_instance: RLAgent
                           ) -> Environment:
        logger.debug("[EnvironmentFactory:create_environment] Attempting to create environment...")

        env_config_section = config.get('environment', {})
        if not isinstance(env_config_section, dict):
            raise ValueError("Config section 'environment' missing or not a dictionary.")

        env_type = env_config_section.get('type')
        if not env_type or not isinstance(env_type, str):
            raise ValueError("Missing or invalid 'type' in 'environment' config section.")
        logger.info(f"[EnvironmentFactory:create_environment] Environment type requested: {env_type}")

        environment: Environment
        try:
            if env_type == 'pendulum_environment':
                # Extraer dt desde environment.simulation.dt
                dt_val = env_config_section.get('simulation', {}).get('dt')
                # Extraer reset_gains desde environment.controller.pid_adaptation.reset_gains_each_episode
                reset_gains_val = env_config_section.get('controller', {}).get('pid_adaptation', {}).get('reset_gains_each_episode')

                # PendulumEnvironment validará dt y reset_gains
                logger.debug(f"[EnvironmentFactory:create_environment] Creating PendulumEnvironment. dt={dt_val}, reset_gains={reset_gains_val}")
                environment = PendulumEnvironment(
                    system=system_instance,
                    controller=controller_instance,
                    agent=agent_instance,
                    reward_function=reward_function_instance,
                    dt=dt_val, # type: ignore # El constructor de PendulumEnv validará
                    reset_gains=reset_gains_val, # type: ignore # El constructor validará
                    config=config # Pasar config completa
                )
            # --- Añadir otros tipos de entorno aquí ---
            # elif env_type == 'other_env_type':
            #     # Extraer otros params específicos del entorno desde config
            #     environment = OtherEnvClass(...)
            else:
                raise ValueError(f"Unknown environment type specified: {env_type}")

            logger.info(f"[EnvironmentFactory:create_environment] Environment '{type(environment).__name__}' created.")
            return environment
        except (ValueError, TypeError, KeyError) as e_constr: # Errores del constructor del entorno o de acceso a config
            logger.error(f"[EnvironmentFactory:create_environment] Error constructing environment '{env_type}': {e_constr}", exc_info=True)
            raise
        except Exception as e_unexp:
            logger.error(f"[EnvironmentFactory:create_environment] Unexpected error creating environment '{env_type}': {e_unexp}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating environment '{env_type}'") from e_unexp