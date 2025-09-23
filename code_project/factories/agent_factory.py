import logging
from typing import Dict, Any
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Import Strategy Interface

# Import specific agent classes
from components.agents.pid_qlearning_agent import PIDQLearningAgent
# Import other agent types here if you add them later
# from components.agents.other_agent import OtherAgent

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory class (as a service) for creating Reinforcement Learning agent instances.
    Instances of this factory will be registered in the DI container.
    """
    def __init__(self):
        """Constructor (puede estar vacío si la fábrica no necesita dependencias)."""
        logger.info("AgentFactory instance created.")
        pass

    def create_agent(self, agent_type: str, agent_params: Dict[str, Any]) -> RLAgent:
        """
        Creates an agent instance based on the specified type and parameters.
        This is now an instance method.

        Args:
            agent_type (str): The type of agent to create (e.g., 'pid_qlearning').
            agent_params (Dict[str, Any]): A dictionary containing all necessary parameters
                                           for the agent's constructor. **Crucially, this
                                           dictionary MUST include 'reward_strategy_instance'**,
                                           which should be injected by the DI container's
                                           lambda provider when resolving the RLAgent interface.
                                           It may also contain other injected params like
                                           'shadow_baseline_params' if needed.

        Returns:
            An instance of an RLAgent subclass.

        Raises:
            ValueError: If the agent type is unknown or required parameters are missing/invalid.
            AttributeError: If the parameters dict is missing crucial keys like 'reward_strategy_instance'.
            TypeError: If 'reward_strategy_instance' is not a valid RewardStrategy object.
            RuntimeError: For unexpected errors during agent creation.
        """
        logger.info(f"Attempting to create agent of type: {agent_type}")

        # Validate and extract the mandatory reward strategy instance
        if 'reward_strategy_instance' not in agent_params:
            logger.error("CRITICAL: 'reward_strategy_instance' not found in agent_params provided to AgentFactory.")
            raise AttributeError("Agent creation requires 'reward_strategy_instance' key in parameters dictionary.")

        reward_strategy = agent_params['reward_strategy_instance']
        if not isinstance(reward_strategy, RewardStrategy):
            logger.error(f"CRITICAL: Invalid type for 'reward_strategy_instance': {type(reward_strategy).__name__}. Expected RewardStrategy.")
            raise TypeError("Provided 'reward_strategy_instance' is not a valid RewardStrategy object.")

        # Create a copy of params to avoid modifying the original dict passed by the container lambda
        # Remove the strategy instance from the copy before passing to the agent constructor
        constructor_params = agent_params.copy()
        del constructor_params['reward_strategy_instance'] # Remove strategy instance itself

        try:
            agent: RLAgent # Type hint
            if agent_type == 'pid_qlearning':
                # Pass the extracted strategy object and the rest of the params using **
                # The 'shadow_baseline_params' would be inside constructor_params if added by the lambda
                logger.debug(f"Creating PIDQLearningAgent with params: {constructor_params.keys()}")
                agent = PIDQLearningAgent(reward_strategy=reward_strategy, **constructor_params)
            # Add other agent types here
            # elif agent_type == 'other_agent':
            #     agent = OtherAgent(reward_strategy=reward_strategy, **constructor_params)
            else:
                raise ValueError(f"Unknown agent type specified: {agent_type}")

            logger.info(f"Successfully created agent: {type(agent).__name__}")
            return agent

        except TypeError as e:
             # Catches errors like missing arguments in the specific agent's __init__
             logger.error(f"Type error creating agent '{agent_type}'. Check configuration parameters against agent constructor: {e}", exc_info=True)
             raise ValueError(f"Parameter mismatch for agent '{agent_type}'. Ensure all required agent parameters are in config.") from e
        except KeyError as e:
             # This might happen if the agent constructor expects a key not present in constructor_params
             logger.error(f"Missing parameter key expected by agent '{agent_type}' constructor: {e}", exc_info=True)
             raise ValueError(f"Missing parameter for agent '{agent_type}'. Check agent constructor and config.") from e
        except Exception as e:
            logger.error(f"Failed to create agent of type '{agent_type}': {e}", exc_info=True)
            # Wrap in a RuntimeError for clarity that it's a creation failure
            raise RuntimeError(f"Unexpected error creating agent '{agent_type}'") from e