# factories/controller_factory.py
import logging
from typing import Dict, Any, Callable
from interfaces.controller import Controller

# No se importa PIDController aquí directamente.

logger = logging.getLogger(__name__)

class ControllerFactory:
    def __init__(self):
        self._creators: Dict[str, Callable[..., Controller]] = {}
        logger.info("[ControllerFactory] Instance created. Ready to register controller creators.")

    def register_controller_type(self, controller_type_name: str, creator_func: Callable[..., Controller]):
        if controller_type_name in self._creators:
            logger.warning(f"[ControllerFactory:register] Overwriting creator for controller type: {controller_type_name}")
        self._creators[controller_type_name] = creator_func
        logger.info(f"[ControllerFactory:register] Controller type '{controller_type_name}' registered with creator: {getattr(creator_func, '__name__', str(creator_func))}")

    def create_controller(self, controller_type: str, controller_params: Dict[str, Any]) -> Controller:
        """
        Crea una instancia de Controller.
        controller_params incluye los de config.environment.controller.params y 'dt_sec'.
        """
        logger.info(f"[ControllerFactory:create_controller] Attempting controller type: '{controller_type}'")
        # logger.debug(f"[ControllerFactory:create_controller] With params: {controller_params}")

        creator = self._creators.get(controller_type)
        if not creator:
            error_msg = f"Unknown controller type specified: '{controller_type}'. Available types: {list(self._creators.keys())}"
            logger.critical(f"[ControllerFactory:create_controller] {error_msg}")
            raise ValueError(error_msg)
        
        # El constructor del controlador concreto valida sus params.
        return creator(**controller_params)