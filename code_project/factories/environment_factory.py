from components.environments.pendulum_environment import PendulumEnvironment
from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory
import logging

class EnvironmentFactory:
    @staticmethod
    def create_environment(config): # Pass the whole config dictionary
        """Creates the simulation environment based on the configuration."""
        try:
            env_config = config['environment'] # Specific environment settings
            env_type = env_config.get('type')
            logging.info(f"Attempting to create environment of type: {env_type}")

            if env_type == 'pendulum_environment':
                # --- Create Components ---
                system = SystemFactory.create_system(env_config['system'])
                logging.info("System created.")

                # Pass dt explicitly to ControllerFactory
                controller_dt = env_config['dt']
                controller = ControllerFactory.create_controller(env_config['controller'], dt=controller_dt)
                logging.info("Controller created.")

                # AgentFactory needs the full config to access pid_adaptation etc.
                agent = AgentFactory.create_agent(env_config['agent'], config)
                logging.info("Agent created.")

                reward_function = RewardFactory.create_reward_function(env_config['reward'])
                logging.info("Reward function created.")

                # --- Extract Environment Parameters ---
                dt = env_config['dt']
                # Get adaptation params safely from top-level config
                pid_adapt_config = config.get('pid_adaptation', {})
                gain_step = pid_adapt_config.get('gain_step', 1.0) # Default if missing
                variable_step = pid_adapt_config.get('variable_step', False)
                reset_gains = pid_adapt_config.get('reset_gains_each_episode', True)
                logging.info(f"PID Adaptation params: gain_step={gain_step}, variable_step={variable_step}, reset_gains={reset_gains}")

                # --- Instantiate Environment ---
                # Pass the full config to the environment if it needs access to simulation/stabilization criteria later
                environment = PendulumEnvironment(
                    system=system,
                    controller=controller,
                    agent=agent,
                    reward_function=reward_function,
                    dt=dt,
                    gain_step=gain_step,
                    variable_step=variable_step,
                    reset_gains=reset_gains,
                    config=config # Pass full config
                )
                logging.info("Pendulum Environment successfully created.")
                return environment

            # Add other environment types here with 'elif'
            # elif env_type == 'other_environment':
            #     pass

            else:
                logging.error(f"Environment type '{env_type}' not recognized.")
                raise ValueError(f"Environment type '{env_type}' not recognized.")

        except KeyError as e:
            logging.error(f"Configuration Error: Missing key {e} in config structure.")
            raise ValueError(f"Configuration Error: Missing key {e}") from e
        except Exception as e:
            # Log the full traceback for debugging
            logging.exception(f"Unexpected error creating environment: {e}")
            raise ValueError(f"Error creating environment: {e}") from e