import logging
from typing import Dict, Any
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Importar para validación

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
        # El logger se inyectará si la factoría lo necesitara, pero por ahora usamos el logger del módulo.
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
                                           It also includes parameters extracted from other
                                           config sections like 'pid_adaptation' (gain_step, etc.).

        Returns:
            An instance of an RLAgent subclass.

        Raises:
            ValueError: If the agent type is unknown or required parameters are missing/invalid.
            AttributeError: If the parameters dict is missing crucial keys like 'reward_strategy_instance'.
            TypeError: If 'reward_strategy_instance' is not a valid RewardStrategy object or other params have wrong types.
            RuntimeError: For unexpected errors during agent creation.
        """
        logger.info(f"Attempting to create agent of type: {agent_type}")
        logger.debug(f"Agent params received by factory: {list(agent_params.keys())}")

        # --- Validación y Extracción de Dependencias Clave ---
        if 'reward_strategy_instance' not in agent_params:
            msg = "CRITICAL: 'reward_strategy_instance' not found in agent_params provided to AgentFactory."
            logger.error(msg)
            raise AttributeError(msg)

        reward_strategy = agent_params['reward_strategy_instance']
        if not isinstance(reward_strategy, RewardStrategy):
             msg = f"CRITICAL: Invalid type for 'reward_strategy_instance': {type(reward_strategy).__name__}. Expected RewardStrategy."
             logger.error(msg)
             raise TypeError(msg)

        # Crear copia de params para no modificar el original y quitar strategy_instance
        constructor_params = agent_params.copy()
        # La instancia de la estrategia se pasará como argumento nombrado al constructor
        del constructor_params['reward_strategy_instance']

        # --- Instanciación Específica por Tipo ---
        agent: RLAgent # Type hint
        try:
            if agent_type == 'pid_qlearning':
                # Verificar presencia de parámetros específicos requeridos por PIDQLearningAgent
                required_keys = ['state_config', 'num_actions', 'gain_step', 'variable_step',
                                 'discount_factor', 'epsilon', 'epsilon_min', 'epsilon_decay',
                                 'learning_rate', 'learning_rate_min', 'learning_rate_decay',
                                 'use_epsilon_decay', 'use_learning_rate_decay']
                # 'q_init_value', 'visit_init_value', 'shadow_baseline_params' son opcionales en el constructor
                missing_keys = [key for key in required_keys if key not in constructor_params or constructor_params[key] is None] # Check for None too
                if missing_keys:
                    raise ValueError(f"Missing required parameters {missing_keys} for agent 'pid_qlearning' in constructor_params.")

                logger.debug(f"Creating PIDQLearningAgent with constructor params: {list(constructor_params.keys())}")
                # Pasar la estrategia como argumento nombrado y el resto con **
                agent = PIDQLearningAgent(reward_strategy=reward_strategy, **constructor_params)

            # --- Añadir otros tipos de agente ---
            # elif agent_type == 'other_agent':
            #     # Validar params para OtherAgent
            #     # ...
            #     agent = OtherAgent(reward_strategy=reward_strategy, **constructor_params)

            else:
                raise ValueError(f"Unknown agent type specified: {agent_type}")

            logger.info(f"Successfully created agent: {type(agent).__name__}")
            return agent

        except TypeError as e:
             # Captura errores de argumentos/tipos en el __init__ del agente específico
             logger.error(f"Type error creating agent '{agent_type}'. Check config parameters against agent constructor: {e}", exc_info=True)
             # Ser más específico sobre el posible problema
             if "required positional argument" in str(e) or "missing" in str(e):
                  raise ValueError(f"Parameter mismatch for agent '{agent_type}'. Ensure all required parameters are in config and DI lambda.") from e
             else:
                  raise TypeError(f"Parameter type mismatch for agent '{agent_type}'. Check config values.") from e
        except KeyError as e:
             # Si el constructor del agente accede a una clave que no está en constructor_params
             logger.error(f"Missing parameter key expected by agent '{agent_type}' constructor: {e}", exc_info=True)
             raise ValueError(f"Missing parameter for agent '{agent_type}'. Check agent constructor and config/DI lambda.") from e
        except ValueError as e: # Captura errores de validación (e.g., missing keys)
             logger.error(f"Configuration or parameter error for agent '{agent_type}': {e}", exc_info=True)
             raise # Re-raise known config/value errors
        except Exception as e:
            logger.error(f"Failed to create agent of type '{agent_type}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating agent '{agent_type}'") from e