from components.agents.pid_qlearning_agent import PIDQLearningAgent

class AgentFactory:
    @staticmethod
    def create_agent(agent_config, env_config):
        agent_type = agent_config.get('type')

        if agent_type == 'pid_qlearning':
            return PIDQLearningAgent(
                state_config=agent_config['params']['state_config'],
                num_actions=agent_config['params']['num_actions'],
                gain_step=env_config['pid_adaptation']['gain_step'],
                variable_step=env_config['pid_adaptation']['variable_step'],
                discount_factor=agent_config['params']['discount_factor'],
                epsilon=agent_config['params']['epsilon'],
                epsilon_min=agent_config['params']['epsilon_min'],
                epsilon_decay=agent_config['params']['epsilon_decay'],
                learning_rate=agent_config['params']['learning_rate'],
                learning_rate_min=agent_config['params']['learning_rate_min'],
                learning_rate_decay=agent_config['params']['learning_rate_decay'],
                use_epsilon_decay=agent_config['params']['use_epsilon_decay'],
                use_learning_rate_decay=agent_config['params']['use_learning_rate_decay']
            )

        raise ValueError(f"Agent '{agent_type}' not recognized.")