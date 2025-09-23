# factories/agent_factory.py
import logging
from typing import Dict, Any, Optional
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Para type check
from components.agents.pid_qlearning_agent import PIDQLearningAgent

logger = logging.getLogger(__name__) # Logger específico del módulo

class AgentFactory:
    def __init__(self):
        logger.info("[AgentFactory] Instance created.")

    def create_agent(self, agent_type: str, agent_params: Dict[str, Any]) -> RLAgent:
        # agent_params ya contiene reward_strategy_instance, state_config, gain_step,
        # variable_step, shadow_baseline_params, etc., inyectados por DI Container.
        logger.info(f"[AgentFactory:create_agent] Attempting to create agent type: {agent_type}")
        logger.debug(f"[AgentFactory:create_agent] Received agent_params keys: {list(agent_params.keys())}")

        # Validar presencia de reward_strategy_instance (esencial)
        reward_strategy_instance = agent_params.get('reward_strategy_instance') # Extraerla primero
        if not isinstance(reward_strategy_instance, RewardStrategy):
            msg = f"CRITICAL: 'reward_strategy_instance' in agent_params is missing or not a RewardStrategy (Type: {type(reward_strategy_instance).__name__})."
            logger.critical(f"[AgentFactory:create_agent] {msg}")
            raise TypeError(msg) # Fail-Fast

        # Crear una copia de agent_params para pasarla como **kwargs
        # y ELIMINAR 'reward_strategy_instance' de esta copia para evitar el TypeError.
        constructor_kwargs = agent_params.copy()
        if 'reward_strategy_instance' in constructor_kwargs: del constructor_kwargs['reward_strategy_instance']
        if 'early_termination_config' in constructor_kwargs and 'early_termination' in constructor_kwargs: del constructor_kwargs['early_termination']

        agent: RLAgent
        try:
            if agent_type == 'pid_qlearning':
                logger.debug(f"[AgentFactory:create_agent] Creating PIDQLearningAgent. Passing reward_strategy explicitly and other params via **kwargs (keys: {list(constructor_kwargs.keys())}).")
                agent = PIDQLearningAgent(
                    reward_strategy=reward_strategy_instance, # Argumento nombrado explícito
                    **constructor_kwargs # El resto de params desempaquetados
                )
            # --- Añadir otros tipos de agente aquí ---
            # elif agent_type == 'other_agent_type':
            #     agent = OtherAgentClass(reward_strategy=reward_strategy_instance, **constructor_kwargs)
            else:
                raise ValueError(f"Unknown agent type specified: {agent_type}")

            logger.info(f"[AgentFactory:create_agent] Agent '{type(agent).__name__}' created successfully.")
            return agent
        except (ValueError, TypeError) as e_constr:
            logger.error(f"[AgentFactory:create_agent] Error constructing agent '{agent_type}': {e_constr}", exc_info=True)
            raise
        except Exception as e_unexp:
            logger.error(f"[AgentFactory:create_agent] Unexpected error creating agent '{agent_type}': {e_unexp}", exc_info=True)
            raise RuntimeError(f"Unexpected error creating agent '{agent_type}'") from e_unexp