# factories/agent_factory.py
import logging
from typing import Dict, Any, Optional # Optional añadido
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Importar para validación

# Import specific agent classes
from components.agents.pid_qlearning_agent import PIDQLearningAgent
# Import other agent types here if you add them later

# 10.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory service for creating Reinforcement Learning agent instances.
    Relies on the DI container to provide resolved dependencies within agent_params.
    """
    def __init__(self):
        # El logger se obtiene del módulo, no se inyecta en la factoría misma.
        logger.info("AgentFactory instance created.")
        pass

    def create_agent(self, agent_type: str, agent_params: Dict[str, Any]) -> RLAgent:
        """
        Creates an agent instance based on the specified type and parameters dictionary.

        Args:
            agent_type (str): Type of agent to create (e.g., 'pid_qlearning').
            agent_params (Dict[str, Any]): Dictionary containing all necessary parameters
                                           for the agent's constructor. MUST include
                                           'reward_strategy_instance' injected by DI.
                                           Should also include other necessary parameters like
                                           'state_config', 'num_actions', learning rates, etc.

        Returns:
            An instance of an RLAgent subclass.

        Raises:
            ValueError: If agent type is unknown or required parameters are missing/invalid in agent_params.
            TypeError: If 'reward_strategy_instance' is invalid or other parameters have wrong types.
            RuntimeError: For unexpected errors during agent creation.
        """
        logger.info(f"Attempting to create agent of type: {agent_type}")
        logger.debug(f"Agent params received by factory (keys): {list(agent_params.keys())}")

        # --- Validación y Extracción de Dependencias Clave ---
        # 10.2: Validar presencia y tipo de la estrategia inyectada (Fail-Fast)
        if 'reward_strategy_instance' not in agent_params:
            msg = "CRITICAL: 'reward_strategy_instance' not found in agent_params provided to AgentFactory."
            logger.error(msg)
            raise ValueError(msg) # Usar ValueError o KeyError/AttributeError

        reward_strategy = agent_params.get('reward_strategy_instance')
        if not isinstance(reward_strategy, RewardStrategy):
            msg = f"CRITICAL: Invalid type for 'reward_strategy_instance': {type(reward_strategy).__name__}. Expected RewardStrategy."
            logger.error(msg)
            raise TypeError(msg)

        # --- Crear copia de params y quitar la estrategia para pasarla por separado ---
        # Esto evita pasarla dos veces si está también en el **constructor_params
        constructor_params = agent_params.copy()
        del constructor_params['reward_strategy_instance']

        # --- Instanciación Específica por Tipo ---
        agent: RLAgent # Type hint
        try:
            if agent_type == 'pid_qlearning':
                # 10.3: Validar presencia de claves *esenciales* para PIDQLearningAgent en los params
                #       (No valida los valores, solo la presencia de las claves)
                required_keys = [
                    'state_config', 'num_actions', 'gain_step', 'variable_step',
                    'discount_factor', 'epsilon', 'epsilon_min', 'epsilon_decay',
                    'learning_rate', 'learning_rate_min', 'learning_rate_decay',
                    'use_epsilon_decay', 'use_learning_rate_decay'
                    # 'q_init_value', 'visit_init_value', 'shadow_baseline_params' son opcionales
                ]
                missing_keys = [key for key in required_keys if key not in constructor_params]
                if missing_keys:
                    # Fail-Fast si falta una clave esencial
                    raise ValueError(f"Missing required parameters {missing_keys} for agent '{agent_type}' in agent_params.")

                logger.debug(f"Creating PIDQLearningAgent with reward_strategy and params (keys): {list(constructor_params.keys())}")
                # 10.4: Pasar estrategia como argumento nombrado, resto con **
                agent = PIDQLearningAgent(reward_strategy=reward_strategy, **constructor_params)

            # --- Añadir otros tipos de agente ---
            # elif agent_type == 'other_agent':
            #     required_keys_other = [...]
            #     missing_keys = [...]
            #     if missing_keys: raise ValueError(...)
            #     agent = OtherAgent(reward_strategy=reward_strategy, **constructor_params)

            else:
                # Fail-Fast si el tipo es desconocido
                raise ValueError(f"Unknown agent type specified: {agent_type}")

            logger.info(f"Successfully created agent: {type(agent).__name__}")
            return agent

        # 10.5: Capturar errores específicos y relanzar con mensajes claros
        except TypeError as e:
            logger.error(f"Type error creating agent '{agent_type}'. Check config parameters vs agent constructor: {e}", exc_info=True)
            if "required positional argument" in str(e) or "missing" in str(e) or "unexpected keyword" in str(e):
                 raise ValueError(f"Parameter mismatch for agent '{agent_type}'. Ensure config/DI provides correct args: {e}") from e
            else:
                 raise TypeError(f"Parameter type mismatch for agent '{agent_type}'. Check config values: {e}") from e
        except KeyError as e:
             logger.error(f"Missing parameter key expected by agent '{agent_type}' constructor: {e}", exc_info=True)
             raise ValueError(f"Missing parameter for agent '{agent_type}'. Check agent constructor and config/DI: {e}") from e
        except ValueError as e: # Captura errores de validación o tipo desconocido
             logger.error(f"Configuration or parameter error for agent '{agent_type}': {e}", exc_info=True)
             raise # Re-raise known config/value errors (Fail-Fast)
        except Exception as e:
            logger.error(f"Unexpected error creating agent '{agent_type}': {e}", exc_info=True)
            # Envolver en RuntimeError para indicar fallo inesperado en creación
            raise RuntimeError(f"Unexpected error creating agent '{agent_type}'") from e