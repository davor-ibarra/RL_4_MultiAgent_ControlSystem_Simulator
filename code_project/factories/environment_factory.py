import logging
from typing import Dict, Any, Optional

# Import interfaces and base classes
from interfaces.environment import Environment
from interfaces.reward_function import RewardFunction
from interfaces.dynamic_system import DynamicSystem # Needed for type hints
from interfaces.controller import Controller     # Needed for type hints
from interfaces.rl_agent import RLAgent          # Needed for type hints

# Import specific environment type
from components.environments.pendulum_environment import PendulumEnvironment
# Import other environment types here
# from components.environments.other_environment import OtherEnvironment

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class EnvironmentFactory:
    """
    Factory class (as a service) for creating simulation environment instances.
    It receives pre-resolved components (System, Controller, Agent, RewardFunction)
    created via their respective interfaces and factories by the DI container.
    """
    def __init__(self):
        """Constructor (puede estar vacío si la fábrica no necesita dependencias)."""
        logger.info("EnvironmentFactory instance created.")
        pass

    def create_environment(self,
                           config: Dict[str, Any],
                           reward_function_instance: RewardFunction,
                           system_instance: DynamicSystem,    # Injected instance
                           controller_instance: Controller,   # Injected instance
                           agent_instance: RLAgent           # Injected instance
                           ) -> Environment:
        """
        Creates an environment instance based on the configuration, using
        pre-created components provided by the DI container.
        This is now an instance method.

        Args:
            config: The main configuration dictionary.
            reward_function_instance: The resolved RewardFunction instance.
            system_instance: The resolved DynamicSystem instance.
            controller_instance: The resolved Controller instance.
            agent_instance: The resolved RLAgent instance.

        Returns:
            An instance of an Environment subclass.

        Raises:
            ValueError: If environment type is unknown or config is invalid/missing required params.
            RuntimeError: For unexpected errors during creation.
        """
        try:
            env_config = config.get('environment', {})
            if not env_config or not isinstance(env_config, dict):
                 raise ValueError("Configuration missing 'environment' section or it's not a dictionary.")

            env_type = env_config.get('type')
            if not env_type:
                 raise ValueError("Missing 'type' in 'environment' config section.")
            logger.info(f"Attempting to create environment of type: {env_type}")

            environment: Environment # Type hint

            if env_type == 'pendulum_environment':
                # Get Environment Specific Parameters from config needed for constructor
                dt = env_config.get('dt')
                if dt is None or not isinstance(dt, (float, int)) or dt <= 0:
                    raise ValueError("Missing or invalid required environment parameter in config: 'dt' must be a positive number.")

                # Get PID adaptation settings needed by PendulumEnvironment constructor
                pid_adapt_cfg = config.get('pid_adaptation', {})
                reset_gains = pid_adapt_cfg.get('reset_gains_each_episode')
                if reset_gains is None or not isinstance(reset_gains, bool):
                     raise ValueError("Missing or invalid required parameter in 'pid_adaptation' config: 'reset_gains_each_episode' must be boolean.")

                # --- Create Environment Instance ---
                logger.debug("Creating PendulumEnvironment instance...")
                environment = PendulumEnvironment(
                    system=system_instance,
                    controller=controller_instance,
                    agent=agent_instance,
                    reward_function=reward_function_instance,
                    dt=dt,
                    reset_gains=reset_gains,
                    config=config # Pass full config for internal use (e.g., termination checks)
                )

            # --- Add other environment types here ---
            # elif env_type == 'other_environment':
            #     # Get specific params for OtherEnvironment
            #     # ... validation ...
            #     environment = OtherEnvironment(...)

            else:
                raise ValueError(f"Unknown environment type specified in config: {env_type}")

            logger.info(f"Successfully created environment: {type(environment).__name__}")
            return environment

        except KeyError as e:
            logger.error(f"Missing configuration key required for environment '{env_type}': {e}", exc_info=True)
            raise ValueError(f"Invalid configuration for environment '{env_type}': Missing key {e}") from e
        except ValueError as e: # Catch ValueErrors from checks or unknown type
            logger.error(f"Configuration error creating environment '{env_type}': {e}", exc_info=True)
            raise
        except TypeError as e:
             # If a received instance (system, controller, etc.) has the wrong type
             logger.error(f"Type error with provided components for environment '{env_type}': {e}", exc_info=True)
             raise TypeError(f"Component type mismatch for environment '{env_type}'.") from e
        except Exception as e:
            logger.error(f"Failed to create environment '{env_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating environment '{env_type}'") from e