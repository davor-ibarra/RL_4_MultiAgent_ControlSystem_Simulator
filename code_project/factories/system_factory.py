# factories/system_factory.py
import logging
from typing import Dict, Any, Callable
from interfaces.dynamic_system import DynamicSystem

logger = logging.getLogger(__name__)

class SystemFactory:
    def __init__(self):
        self._creators: Dict[str, Callable[..., DynamicSystem]] = {}
        logger.info("[SystemFactory] Instance created. Ready to register system creators.")

    def register_system_type(self, system_type_name: str, creator_func: Callable[..., DynamicSystem]):
        """Registra una función creadora para un tipo de sistema dinámico específico."""
        if system_type_name in self._creators:
            logger.warning(f"[SystemFactory:register] Overwriting creator for system type: {system_type_name}")
        self._creators[system_type_name] = creator_func
        logger.info(f"[SystemFactory:register] System type '{system_type_name}' registered with creator: "
                    f"{getattr(creator_func, '__name__', str(creator_func))}")

    def create_system(self, system_type: str, system_params: Dict[str, Any]) -> DynamicSystem:
        """
        Crea una instancia de DynamicSystem.
        system_params son los contenidos bajo config.environment.system.params.
        """
        logger.info(f"[SystemFactory:create_system] Attempting to create system of type: '{system_type}'")
        creator = self._creators.get(system_type)
        if not creator:
            error_msg = f"Unknown system type specified: '{system_type}'. Available types: {list(self._creators.keys())}"
            logger.critical(f"[SystemFactory:create_system] {error_msg}")
            raise ValueError(error_msg)
        return creator(**system_params)