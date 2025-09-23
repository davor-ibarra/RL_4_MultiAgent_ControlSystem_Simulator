import logging
from components.agents.pid_qlearning_agent import PIDQLearningAgent

class AgentFactory:
    @staticmethod
    def create_agent(agent_config, config): # Pasamos config completo
        """
        Creates an agent instance based on configuration.

        Args:
            agent_config (dict): The 'agent' section from the config.
            config (dict): The full configuration dictionary.

        Returns:
            An instance of the specified agent.

        Raises:
            ValueError: If the agent_type is not recognized or params are missing.
        """
        agent_type = agent_config.get('type')
        params = agent_config.get('params', {})
        env_config = config.get('environment', {}) # Acceso a dt, etc.
        reward_config = env_config.get('reward', {}) # Acceso a config de recompensa
        pid_adapt_config = config.get('pid_adaptation', {}) # Acceso a gain_step

        logging.info(f"Attempting to create agent of type: {agent_type}")

        if agent_type == 'pid_qlearning':
            try:
                # Extract necessary params safely
                state_config = params['state_config']
                num_actions = params['num_actions']
                gain_step = pid_adapt_config['gain_step']
                variable_step = pid_adapt_config['variable_step']
                discount_factor = params['discount_factor']
                epsilon = params['epsilon']
                epsilon_min = params['epsilon_min']
                epsilon_decay = params['epsilon_decay']
                learning_rate = params['learning_rate']
                learning_rate_min = params['learning_rate_min']
                learning_rate_decay = params['learning_rate_decay']
                use_epsilon_decay = params['use_epsilon_decay']
                use_learning_rate_decay = params['use_learning_rate_decay']

                # --- NEW: Extract reward mode and shadow params ---
                reward_mode = reward_config.get('reward_mode', 'global')
                shadow_params = reward_config.get('shadow_baseline_params', {}) # Get shadow sub-dict

                return PIDQLearningAgent(
                    state_config=state_config,
                    num_actions=num_actions,
                    gain_step=gain_step,
                    variable_step=variable_step,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    learning_rate=learning_rate,
                    learning_rate_min=learning_rate_min,
                    learning_rate_decay=learning_rate_decay,
                    use_epsilon_decay=use_epsilon_decay,
                    use_learning_rate_decay=use_learning_rate_decay,
                    # --- Pass new params ---
                    reward_mode=reward_mode,
                    shadow_baseline_params=shadow_params
                )
            except KeyError as e:
                 logging.error(f"Missing required parameter for PIDQLearningAgent in config: {e}")
                 raise ValueError(f"Missing required parameter for PIDQLearningAgent: {e}") from e
            except Exception as e:
                 logging.error(f"Unexpected error creating PIDQLearningAgent: {e}", exc_info=True)
                 raise

        raise ValueError(f"Agent type '{agent_type}' not recognized.")