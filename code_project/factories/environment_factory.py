# factories/environment_factory.py
import logging
from typing import Dict, Any, Callable
from interfaces.environment import Environment
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator

logger = logging.getLogger(__name__)

class EnvironmentFactory:
    def __init__(self):
        self._creators: Dict[str, Callable[..., Environment]] = {}
        logger.info("[EnvironmentFactory] Instance created. Ready to register environment creators.")

    def register_environment_type(self, env_type_name: str, creator_func: Callable[..., Environment]):
        if env_type_name in self._creators:
            logger.warning(f"[EnvironmentFactory:register] Overwriting creator for environment type: {env_type_name}")
        self._creators[env_type_name] = creator_func
        logger.info(f"[EnvironmentFactory:register] Environment type '{env_type_name}' registered with creator: "
                    f"{getattr(creator_func, '__name__', str(creator_func))}")

    def create_environment(self,
                           env_type: str,
                           system: DynamicSystem,
                           controllers: Dict[str, Controller],
                           agent: RLAgent,
                           reward_function: RewardFunction,
                           stability_calculator: BaseStabilityCalculator,
                           config: Dict[str, Any]
                           ) -> Environment:
        logger.info(f"[EnvironmentFactory:create_environment] Attempting environment type: '{env_type}'")
        
        creator = self._creators.get(env_type)
        if not creator:
            error_msg = f"Unknown environment type specified: '{env_type}'. Available types: {list(self._creators.keys())}"
            logger.critical(f"[EnvironmentFactory:create_environment] {error_msg}")
            raise ValueError(error_msg)

        return creator(system=system,
                       controllers=controllers,
                       agent=agent,
                       reward_function=reward_function,
                       stability_calculator=stability_calculator,
                       config=config)