# factories/agent_factory.py
import logging
from typing import Dict, Any, Callable
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Para validación de tipo

# No se importa PIDQLearningAgent aquí directamente.

logger = logging.getLogger(__name__)

class AgentFactory:
    def __init__(self):
        self._creators: Dict[str, Callable[..., RLAgent]] = {}
        logger.info("[AgentFactory] Instance created. Ready to register agent creators.")

    def register_agent_type(self, agent_type_name: str, creator_func: Callable[..., RLAgent]):
        if agent_type_name in self._creators:
            logger.warning(f"[AgentFactory:register] Overwriting creator for agent type: {agent_type_name}")
        self._creators[agent_type_name] = creator_func
        logger.info(f"[AgentFactory:register] Agent type '{agent_type_name}' registered with creator: {getattr(creator_func, '__name__', str(creator_func))}")

    def create_agent(self, agent_type: str, agent_constructor_params: Dict[str, Any]) -> RLAgent:
        """
        Crea una instancia de RLAgent.
        agent_constructor_params es un diccionario que contiene todos los argumentos
        nombrados que el constructor del agente espera (incluyendo 'reward_strategy').
        """
        logger.info(f"[AgentFactory:create_agent] Attempting agent type: '{agent_type}'")
        # logger.debug(f"[AgentFactory:create_agent] With constructor_params keys: {list(agent_constructor_params.keys())}")

        creator = self._creators.get(agent_type)
        if not creator:
            error_msg = f"Unknown agent type specified: '{agent_type}'. Available types: {list(self._creators.keys())}"
            logger.critical(f"[AgentFactory:create_agent] {error_msg}")
            raise ValueError(error_msg)
            
        return creator(**agent_constructor_params)