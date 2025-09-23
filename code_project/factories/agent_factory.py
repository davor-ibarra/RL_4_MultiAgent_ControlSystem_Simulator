import logging
from typing import Dict, Any
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Import Strategy Interface

# Import specific agent classes
from components.agents.pid_qlearning_agent import PIDQLearningAgent
# Import other agent types here if you add them later
# from components.agents.other_agent import OtherAgent

class AgentFactory:
    """
    Factory class for creating Reinforcement Learning agent instances.
    """
    @staticmethod
    def create_agent(agent_type: str, agent_params: Dict[str, Any]) -> RLAgent:
        """
        Creates an agent instance based on the specified type and parameters.

        Args:
            agent_type (str): The type of agent to create (e.g., 'pid_qlearning').
            agent_params (Dict[str, Any]): A dictionary containing all necessary parameters
                                           for the agent's constructor, **including** the
                                           'reward_strategy_instance'.

        Returns:
            An instance of an RLAgent subclass.

        Raises:
            ValueError: If the agent type is unknown or required parameters are missing.
            AttributeError: If the parameters dict is missing crucial keys.
        """
        logging.info(f"Attempting to create agent of type: {agent_type}")

        # Extract the mandatory reward strategy instance
        try:
            reward_strategy = agent_params.pop('reward_strategy_instance')
            if not isinstance(reward_strategy, RewardStrategy):
                raise TypeError("Provided 'reward_strategy_instance' is not a valid RewardStrategy object.")
        except KeyError:
            logging.error("CRITICAL: 'reward_strategy_instance' not found in agent_params.")
            raise ValueError("Agent creation requires 'reward_strategy_instance' in parameters.") from None
        except TypeError as e:
            logging.error(f"CRITICAL: Invalid reward strategy instance provided: {e}")
            raise

        try:
            if agent_type == 'pid_qlearning':
                # Pass the extracted strategy and the rest of the params
                agent = PIDQLearningAgent(reward_strategy=reward_strategy, **agent_params)
            # Add other agent types here
            # elif agent_type == 'other_agent':
            #     agent = OtherAgent(reward_strategy=reward_strategy, **agent_params)
            else:
                raise ValueError(f"Unknown agent type specified: {agent_type}")

            logging.info(f"Successfully created agent: {type(agent).__name__}")
            return agent

        except TypeError as e:
             # Catches errors like missing arguments in the specific agent's __init__
             logging.error(f"Type error creating agent '{agent_type}'. Check configuration parameters against agent constructor: {e}", exc_info=True)
             raise ValueError(f"Parameter mismatch for agent '{agent_type}'.") from e
        except KeyError as e:
             logging.error(f"Missing parameter key expected by agent '{agent_type}': {e}", exc_info=True)
             raise ValueError(f"Missing parameter for agent '{agent_type}'.") from e
        except Exception as e:
            logging.error(f"Failed to create agent of type '{agent_type}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to create agent '{agent_type}'") from e

