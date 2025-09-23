# factories/environment_factory.py
import logging
from typing import Dict, Any, Optional

# Import interfaces and base classes
from interfaces.environment import Environment
from interfaces.reward_function import RewardFunction
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent

# Import specific environment type
from components.environments.pendulum_environment import PendulumEnvironment
# Import other environment types here

# 12.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class EnvironmentFactory:
    """
    Factory service for creating simulation environment instances.
    Uses pre-resolved components (System, Controller, Agent, RewardFunction)
    provided by the DI container.
    """
    def __init__(self):
        logger.info("EnvironmentFactory instance created.")
        pass

    def create_environment(self,
                           config: Dict[str, Any], # Pasar config COMPLETA
                           reward_function_instance: RewardFunction, # Inyectada
                           system_instance: DynamicSystem,           # Inyectada
                           controller_instance: Controller,          # Inyectada
                           agent_instance: RLAgent                   # Inyectada
                           ) -> Environment:
        """
        Creates an environment instance using pre-created components.

        Args:
            config: The main configuration dictionary.
            reward_function_instance: Resolved RewardFunction instance.
            system_instance: Resolved DynamicSystem instance.
            controller_instance: Resolved Controller instance.
            agent_instance: Resolved RLAgent instance.

        Returns:
            An instance of an Environment subclass.

        Raises:
            ValueError: If environment type unknown or config is missing required environment params.
            TypeError: If injected instances are of unexpected types (should be caught by DI).
            RuntimeError: For unexpected errors.
        """
        logger.debug("Attempting to create environment...")
        try:
            # 12.2: Extraer config específica del entorno
            env_config = config.get('environment', {})
            if not env_config or not isinstance(env_config, dict):
                raise ValueError("Configuration missing 'environment' section or it's not a dictionary.")

            env_type = env_config.get('type')
            if not env_type:
                raise ValueError("Missing 'type' in 'environment' config section.")
            logger.info(f"Attempting to create environment of type: {env_type}")

            environment: Environment # Type hint

            if env_type == 'pendulum_environment':
                # 12.3: Validar y extraer parámetros específicos del *entorno* desde config
                dt = env_config.get('dt')
                if dt is None or not isinstance(dt, (float, int)) or dt <= 0:
                    # Fail-Fast si falta dt
                    raise ValueError("Missing or invalid required environment parameter in config: 'dt' must be a positive number.")

                # Obtener config de adaptación PID para `reset_gains`
                pid_adapt_cfg = config.get('pid_adaptation', {})
                reset_gains = pid_adapt_cfg.get('reset_gains_each_episode')
                if reset_gains is None or not isinstance(reset_gains, bool):
                    # Fail-Fast si falta
                    raise ValueError("Missing or invalid parameter in 'pid_adaptation' config: 'reset_gains_each_episode' must be boolean.")

                # 12.4: Crear instancia pasando dependencias inyectadas y params específicos
                logger.debug("Creating PendulumEnvironment instance...")
                environment = PendulumEnvironment(
                    system=system_instance,
                    controller=controller_instance,
                    agent=agent_instance,
                    reward_function=reward_function_instance,
                    dt=dt, # Parámetro específico del entorno
                    reset_gains=reset_gains, # Parámetro específico del entorno
                    config=config # Pasar config completa por si la necesita internamente
                )

            # --- Add other environment types here ---
            # elif env_type == 'other_environment':
            #     # Get specific params for OtherEnvironment
            #     # ... validation (Fail-Fast if missing) ...
            #     environment = OtherEnvironment(
            #           system=system_instance, ..., other_param=..., config=config)

            else:
                # Fail-Fast si el tipo es desconocido
                raise ValueError(f"Unknown environment type specified in config: {env_type}")

            logger.info(f"Successfully created environment: {type(environment).__name__}")
            return environment

        # 12.5: Capturar errores específicos y relanzar
        except KeyError as e: # Si alguna sección esperada (environment, pid_adaptation) falta
            logger.error(f"Missing configuration key required for environment '{env_type}': {e}", exc_info=True)
            raise ValueError(f"Invalid config for environment '{env_type}': Missing key {e}") from e
        except ValueError as e: # Captura errores de validación o tipo desconocido
            logger.error(f"Configuration error creating environment '{env_type}': {e}", exc_info=True)
            raise # Re-raise known config/value errors (Fail-Fast)
        except TypeError as e: # Si una instancia inyectada tiene tipo incorrecto
             logger.error(f"Type error with provided components for environment '{env_type}': {e}", exc_info=True)
             raise TypeError(f"Component type mismatch for environment '{env_type}'. DI Error?") from e
        except Exception as e:
            logger.error(f"Unexpected error creating environment '{env_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating environment '{env_type}'") from e