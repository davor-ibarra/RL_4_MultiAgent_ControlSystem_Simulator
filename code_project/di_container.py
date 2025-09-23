import threading
import logging
from typing import Any, Callable, Dict, Type, Optional, cast, List, Set, TYPE_CHECKING, Union

# Interfaces and Factories (Sin cambios en importaciones)
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.plot_generator import PlotGenerator # Asegurar que está importada

from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory
from factories.environment_factory import EnvironmentFactory

# Importar Servicios y Componentes concretos
from utils.data.result_handler import ResultHandler
# (4.1) Importar HeatmapGenerator
from utils.data.heatmap_generator import HeatmapGenerator
from components.plotting.matplotlib_plot_generator import MatplotlibPlotGenerator
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector

# Mover importación de SimulationManager y VisualizationManager a TYPE_CHECKING o build
if TYPE_CHECKING:
    from simulation_manager import SimulationManager
    # (4.2) Mover VisualizationManager a TYPE_CHECKING también si se importa arriba
    # from utils.plotting.visualization_manager import VisualizationManager

# Definir un token específico para vis_config
VIS_CONFIG_TOKEN = "visualization_config_dict"

class Container:
    """ Contenedor simple DI. """
    def __init__(self):
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._resolving: threading.local = threading.local()
        self._container_logger = logging.getLogger(f'{__name__}[{id(self)}]')
        self._container_logger.info("DI Container instance created.")

    def register(self, token: Any, provider: Callable[['Container'], Any], singleton: bool = False):
        """Registra un proveedor para un token."""
        with self._lock:
            # Lógica de advertencia/debug (sin cambios)
            if token in self._providers and not self._providers[token][1] and not singleton:
                 self._container_logger.warning(f"Overwriting existing provider for token: {token}")
            elif token in self._providers and self._providers[token][1] and not singleton:
                 self._container_logger.warning(f"Registering a transient provider over existing singleton for token: {token}")

            self._providers[token] = (provider, singleton)
            if token in self._singletons:
                 self._container_logger.debug(f"Removing existing singleton instance for re-registered token: {token}")
                 del self._singletons[token]

    def resolve(self, token: Any) -> Any:
        """Resuelve una dependencia registrada."""
        pid = threading.get_ident()
        is_optional = False; origin_token = token
        if getattr(token, '__origin__', None) is Union:
            args = getattr(token, '__args__', ())
            if len(args) == 2 and type(None) in args:
                 is_optional = True
                 token = next(arg for arg in args if arg is not type(None))
        elif getattr(token, '__origin__', None) is Optional:
            is_optional = True; args = getattr(token, '__args__', None)
            if args and len(args) == 1: token = args[0]
            else: raise ValueError(f"No se pudo desempaquetar Optional: {origin_token}")

        with self._lock:
            if token not in self._providers:
                if is_optional: return None
                self._container_logger.error(f"[{pid}] No provider registered for REQUIRED token: {origin_token}")
                raise ValueError(f"No provider registered for required token: {origin_token}")
            provider, singleton_flag = self._providers[token]
            if singleton_flag and token in self._singletons:
                return self._singletons[token]

            if not hasattr(self._resolving, 'tokens'): self._resolving.tokens = set()
            if token in self._resolving.tokens:
                cycle = list(self._resolving.tokens) + [token]
                cycle_str = ' -> '.join(map(str, cycle))
                self._container_logger.error(f"[{pid}] Dependency cycle detected: {cycle_str}")
                raise RecursionError(f"Dependency cycle detected: {cycle_str}")

            self._resolving.tokens.add(token)

        instance = None
        try:
            instance = provider(self)
        except Exception as e:
            self._container_logger.error(f"[{pid}] Error executing provider for token {token}: {e}", exc_info=True)
            if hasattr(self._resolving, 'tokens') and token in self._resolving.tokens:
                self._resolving.tokens.remove(token)
                if not self._resolving.tokens: delattr(self._resolving, 'tokens')
            raise
        finally:
            if hasattr(self._resolving, 'tokens'):
                if token in self._resolving.tokens: self._resolving.tokens.remove(token)
                if not self._resolving.tokens: delattr(self._resolving, 'tokens')

        if singleton_flag:
            with self._lock: self._singletons[token] = instance

        return instance

    def get_registered_tokens(self) -> list[Any]:
        """Devuelve una lista de todos los tokens registrados."""
        with self._lock: return list(self._providers.keys())

# --- Funciones Helper para creación de componentes complejos ---
def _create_stability_calculator(c: Container) -> Optional[BaseStabilityCalculator]:
    """
    Crea una instancia de StabilityCalculator basada en la configuración.
    Devuelve None si está deshabilitado o la sección falta/es inválida.
    Lanza RuntimeError si está habilitado pero mal configurado o la creación falla.
    """
    # --- START MOD: Versión Limpia _create_stability_calculator ---
    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)
    factory = c.resolve(RewardFactory)

    # 1. Extraer la configuración específica de forma segura
    stability_config = config.get('environment', {}).get('reward_setup', {}).get('calculation', {}).get('stability_calculator')

    # 2. Verificar si debe crearse
    #    No se crea si falta la sección, no es un dict, o 'enabled' es explícitamente False
    is_enabled = isinstance(stability_config, dict) and stability_config.get('enabled', False)

    #    Validar tipo de 'enabled' si existe y no es booleano (advertir pero tratar como False)
    if isinstance(stability_config, dict) and 'enabled' in stability_config and not isinstance(stability_config['enabled'], bool):
        logger_instance.warning(f"Valor de 'enabled' en stability_calculator config no es booleano ({type(stability_config['enabled']).__name__}). Asumiendo deshabilitado.")
        is_enabled = False

    if not is_enabled:
        logger_instance.info("Stability calculator no habilitado o sección inválida. No se crea instancia.")
        return None

    # 3. Intentar crear la instancia (si está habilitado)
    logger_instance.info(f"Stability calculator HABILITADO (type: {stability_config.get('type', '??')}). Intentando crear vía RewardFactory...")
    try:
        # La factoría es responsable de validar 'type', 'params' y crear la instancia.
        # Si la factoría lanza ValueError/TypeError por config inválida, o
        # si el constructor del calculator falla, la excepción se captura abajo.
        instance = factory.create_stability_calculator(stability_config)

        # Seguridad adicional: la factoría no debería devolver None si enabled=True y sin errores
        if instance is None:
            msg = "Error Interno: RewardFactory devolvió None para stability calculator habilitado."
            logger_instance.critical(msg)
            raise RuntimeError(msg)

        logger_instance.info(f"StabilityCalculator '{type(instance).__name__}' creado exitosamente.")
        return instance

    except (ValueError, TypeError) as e_config:
        # Errores de configuración/parámetros detectados por la factoría o constructor
        msg = f"Error Crítico de Configuración para Stability Calculator HABILITADO: {e_config}"
        logger_instance.critical(msg, exc_info=True) # Incluir traceback para contexto
        raise RuntimeError(msg) from e_config # Detener DI (Fail-Fast)
    except Exception as e_unexpected:
        # Otros errores inesperados durante la creación
        msg = f"Error Inesperado creando Stability Calculator HABILITADO: {e_unexpected}"
        logger_instance.critical(msg, exc_info=True)
        raise RuntimeError(msg) from e_unexpected
    # --- END MOD: Versión Limpia _create_stability_calculator ---


def _create_reward_function_instance(c: Container) -> RewardFunction:
    # --- START MOD: Versión Limpia _create_reward_function_instance ---
    try:
        config = c.resolve(dict)
        logger_instance = c.resolve(logging.Logger)
        factory = c.resolve(RewardFactory)
        reward_setup_config = config.get('environment', {}).get('reward_setup', {})
        if not reward_setup_config:
            raise ValueError("Config 'environment.reward_setup' no encontrada.")

        # Resolver dependencia (el proveedor _create_stability_calculator devuelve Optional)
        # Se resuelve usando el tipo base, el proveedor maneja la creación o el None.
        stability_calculator_instance = c.resolve(BaseStabilityCalculator)
        #logger_instance.debug(f"[DI Helper] _create_reward_function_instance: Resolved BaseStabilityCalculator as: {type(stability_calculator_instance).__name__ if stability_calculator_instance else 'None'}")

        # Pasar la instancia resuelta (puede ser None) a la factoría
        instance = factory.create_reward_function(reward_setup_config, stability_calculator_instance)
        #logger_instance.debug(f"RewardFunction '{type(instance).__name__}' creada.")
        return instance
    except Exception as e:
        # El error original (ej. del calculator o la factoría) ya se habrá logueado como CRITICAL
        # Solo añadimos un mensaje genérico de que la creación de RewardFunction falló.
        logger_instance.critical(f"Fallo crítico durante la creación de RewardFunction (ver error previo): {e}", exc_info=False) # No duplicar traceback
        raise RuntimeError("Fallo crítico al crear RewardFunction") from e
    # --- END MOD: Versión Limpia _create_reward_function_instance ---

def _create_reward_strategy(c: Container) -> RewardStrategy:
    from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
    from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
    from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy
    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)
    try:
        strategy_config = config.get('environment', {}).get('reward_setup', {}).get('learning_strategy', {})
        if not strategy_config: raise ValueError("Config 'environment.reward_setup.learning_strategy' no encontrada.")
        mode = strategy_config.get('type')
        if not mode: raise ValueError("Falta 'type' en config de learning_strategy.")
        params = strategy_config.get('strategy_params', {})
        if not isinstance(params, dict): raise TypeError("'strategy_params' debe ser un diccionario.")
        logger_instance.info(f"Creando RewardStrategy: {mode}")
        strategy: RewardStrategy
        if mode == 'global': 
            strategy = GlobalRewardStrategy()
        elif mode == 'shadow_baseline': 
            shadow_params = params.get('shadow_baseline', {})
            beta_value = shadow_params.get('beta')
            strategy = ShadowBaselineRewardStrategy(beta_value)
        elif mode == 'echo_baseline':
            logger_instance.debug("Creating EchoBaselineRewardStrategy instance.")
            strategy = EchoBaselineRewardStrategy()
        else: raise ValueError(f"learning_strategy.type desconocido: '{mode}'")
        logger_instance.debug(f"RewardStrategy '{type(strategy).__name__}' creada.")
        return strategy
    except (ValueError, TypeError, KeyError) as e:
        logger_instance.critical(f"Error crítico creando RewardStrategy: {e}", exc_info=True)
        raise RuntimeError(f"Fallo crítico creando RewardStrategy: {e}") from e
    except Exception as e:
         logger_instance.critical(f"Error inesperado creando RewardStrategy: {e}", exc_info=True)
         raise RuntimeError("Fallo inesperado creando RewardStrategy") from e

def _create_virtual_simulator(c: Container) -> Optional[VirtualSimulator]:
    from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator
    logger_instance = c.resolve(logging.Logger) # Obtener logger primero
    #logger_instance.debug("Attempting to create VirtualSimulator...")

    config = c.resolve(dict); logger_instance = c.resolve(logging.Logger)
    if config is None: # Check config early
        logger_instance.error("VirtualSimulator creation failed: Main config is None.")
        return None
    try:
        learn_strategy_type = config.get('environment', {}).get('reward_setup', {}).get('learning_strategy', {}).get('type')
        logger_instance.info(f"Inside _create_virtual_simulator: learn_strategy_type read from config is '{learn_strategy_type}' (Type: {type(learn_strategy_type).__name__})")
        #logger_instance.debug(f"_create_virtual_simulator: Learning strategy type = {learn_strategy_type}") # Log strategy
        if learn_strategy_type != 'echo_baseline':
            logger_instance.debug("VirtualSimulator not created: 'echo_baseline' strategy not selected.")
            return None
        logger_instance.info("Creando VirtualSimulator para 'echo_baseline'...")
        system = c.resolve(DynamicSystem)
        controller_template = c.resolve(Controller)
        reward_function = c.resolve(RewardFunction)
        dt_val = config.get('environment', {}).get('dt')
        
        if system is None:
            logger_instance.error("VirtualSimulator creation failed: Could not resolve DynamicSystem.")
            return None # Devuelve None tras log error
        if controller_template is None:
            logger_instance.error("VirtualSimulator creation failed: Could not resolve Controller template.")
            return None # Devuelve None tras log error
        if reward_function is None:
            logger_instance.error("VirtualSimulator creation failed: Could not resolve RewardFunction.")
            return None # Devuelve None tras log error
        if dt_val is None:
            logger_instance.error("VirtualSimulator creation failed: 'environment.dt' not found in config.")
            return None # Devuelve None tras log error
        if not isinstance(dt_val, (float, int)) or dt_val <= 0:
            logger_instance.error(f"VirtualSimulator creation failed: Invalid 'dt' ({dt_val}). Must be a positive number.")
            return None # Devuelve None tras log error
        
        logger_instance.info("Dependencies resolved. Instantiating PendulumVirtualSimulator...")
        simulator = PendulumVirtualSimulator(
            system=system,
            controller=controller_template,
            reward_function=reward_function,
            dt=dt_val # Usar el valor validado
        )
        logger_instance.info("PendulumVirtualSimulator creado.")
        return simulator
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger_instance.error(f"Error configurando VirtualSimulator: {e}", exc_info=True)
        raise RuntimeError(f"Fallo al crear VirtualSimulator requerido: {e}") from e
    except Exception as e:
         logger_instance.error(f"Error inesperado creando VirtualSimulator: {e}", exc_info=True)
         raise RuntimeError("Fallo inesperado creando VirtualSimulator") from e

# --- Función build_container (Ajustada para VisualizationManager) ---
def build_container(config: Dict[str, Any], vis_config: Optional[Dict[str, Any]]) -> Container:
    """ Construye y configura el contenedor DI. """
    # Importar aquí para evitar ciclos globales
    from simulation_manager import SimulationManager
    from visualization_manager import VisualizationManager

    container = Container()
    logger = container._container_logger

    # --- Registro Fundamental ---
    #logger.debug("Registering fundamental components...")
    container.register(Container, lambda c: container)
    container.register(logging.Logger, lambda c: logging.getLogger(), singleton=True)
    container.register(dict, lambda c: config, singleton=True)
    container.register(VIS_CONFIG_TOKEN, lambda c: vis_config, singleton=True) # vis_config puede ser None

    # --- Factorías ---
    #logger.debug("Registering factories...")
    container.register(SystemFactory, lambda c: SystemFactory(), singleton=True)
    container.register(ControllerFactory, lambda c: ControllerFactory(), singleton=True)
    container.register(AgentFactory, lambda c: AgentFactory(), singleton=True)
    container.register(RewardFactory, lambda c: RewardFactory(), singleton=True)
    container.register(EnvironmentFactory, lambda c: EnvironmentFactory(), singleton=True)

    # --- Componentes Principales (vía helpers) ---
    # Registrar los proveedores usando la INTERFAZ BASE como token.
    # Los helpers _create_* se encargan de devolver la instancia o None (o lanzar error).
    #logger.debug("Registering core component providers (using base interface tokens)...")
    container.register(BaseStabilityCalculator, _create_stability_calculator, singleton=True)
    container.register(RewardFunction, _create_reward_function_instance, singleton=True)
    container.register(RewardStrategy, _create_reward_strategy, singleton=True)
    container.register(VirtualSimulator, _create_virtual_simulator, singleton=True)

    # --- Componentes con dependencias (Lambdas directas usando factorías) ---
    # (Asumiendo que las factorías se modificarán en el futuro para tomar config específica)
    #logger.debug("Registering components with dependencies...")
    container.register(DynamicSystem, lambda c: c.resolve(SystemFactory).create_system(
        system_type=c.resolve(dict).get('environment', {}).get('system', {}).get('type', 'unknown'),
        system_params=c.resolve(dict).get('environment', {}).get('system', {}).get('params', {})),
        singleton=True
    )
    container.register(Controller, lambda c: c.resolve(ControllerFactory).create_controller(
        controller_type=c.resolve(dict).get('environment', {}).get('controller', {}).get('type', 'unknown'),
        controller_params={
            **c.resolve(dict).get('environment', {}).get('controller', {}).get('params', {}),
            'dt': c.resolve(dict).get('environment', {}).get('dt')
        }),
        singleton=True
    )
    container.register(RLAgent, lambda c: c.resolve(AgentFactory).create_agent(
        agent_type=c.resolve(dict).get('environment', {}).get('agent', {}).get('type', 'unknown'),
        agent_params={
            **c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}),
            'gain_step': c.resolve(dict).get('pid_adaptation', {}).get('gain_step'),
            'variable_step': c.resolve(dict).get('pid_adaptation', {}).get('variable_step'),
            'reward_strategy_instance': c.resolve(RewardStrategy),
             'shadow_baseline_params': c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('learning_strategy', {}).get('strategy_params', {}).get('shadow_baseline')
                if c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('learning_strategy', {}).get('type') == 'shadow_baseline' else None,
        }),
        singleton=True
    )
    container.register(Environment, lambda c: c.resolve(EnvironmentFactory).create_environment(
        config=c.resolve(dict),
        reward_function_instance=c.resolve(RewardFunction),
        system_instance=c.resolve(DynamicSystem),
        controller_instance=c.resolve(Controller),
        agent_instance=c.resolve(RLAgent)),
        singleton=True
    )

    # --- Servicios de Soporte ---
    logger.debug("Registering support services...")
    # (4.3) Registrar HeatmapGenerator (singleton)
    container.register(HeatmapGenerator, lambda c: HeatmapGenerator(c.resolve(logging.Logger)), singleton=True)
    # Plot Generator (interfaz -> implementación)
    container.register(PlotGenerator, lambda c: MatplotlibPlotGenerator(), singleton=True)
    # ResultHandler necesita logger (ya no heatmap_generator)
    container.register(ResultHandler, lambda c: ResultHandler(logger=c.resolve(logging.Logger)), singleton=True)
    # SimulationManager
    container.register('simulation_manager.SimulationManager', lambda c: SimulationManager(
        logger=c.resolve(logging.Logger),
        result_handler=c.resolve(ResultHandler),
        container=c),
        singleton=False # Transient
    )
    # MetricsCollector
    container.register(MetricsCollector, lambda c: ExtendedMetricsCollector(), singleton=False) # Transient

    # (4.4) Actualizar registro de VisualizationManager para inyectar HeatmapGenerator
    container.register(VisualizationManager, lambda c: VisualizationManager(
        logger_instance=c.resolve(logging.Logger),
        plot_generator=c.resolve(PlotGenerator),
        # (4.5) Inyectar HeatmapGenerator resuelto
        heatmap_generator=c.resolve(HeatmapGenerator),
        vis_config_data=c.resolve(VIS_CONFIG_TOKEN), # Resuelve config de vis (puede ser None)
        results_folder=c.resolve(str)), # Resuelve carpeta de resultados (registrada en main)
        singleton=False # Transient
    )

    logger.info("DI Container built and providers registered.")
    return container