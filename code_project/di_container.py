# di_container.py
import threading
import logging
from typing import Any, Callable, Dict, Type, Optional, cast, List, Set, TYPE_CHECKING, Union

# Interfaces and Factories
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.plot_generator import PlotGenerator

from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory
from factories.environment_factory import EnvironmentFactory

# Importar Servicios y Componentes concretos
from utils.data.result_handler import ResultHandler
from utils.data.heatmap_generator import HeatmapGenerator
from components.plotting.matplotlib_plot_generator import MatplotlibPlotGenerator
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector

if TYPE_CHECKING:
    from simulation_manager import SimulationManager
    from visualization_manager import VisualizationManager


VIS_CONFIG_TOKEN = "visualization_config_dict"

class Container:
    """ Contenedor simple DI. """
    def __init__(self):
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._resolving: threading.local = threading.local()
        # Usar un logger específico para el contenedor, fácil de identificar
        self._container_logger = logging.getLogger(f'{__name__}.DIContainer[{id(self)}]')
        self._container_logger.info("DI Container instance created.")

    def register(self, token: Any, provider: Callable[['Container'], Any], singleton: bool = False):
        with self._lock:
            if token in self._providers:
                 log_level = logging.WARNING if self._providers[token][1] or not singleton else logging.DEBUG
                 self._container_logger.log(log_level, f"Overwriting provider for token: {token} (Singleton: {singleton})")
            self._providers[token] = (provider, singleton)
            if token in self._singletons:
                 self._container_logger.debug(f"Removing existing singleton for re-registered token: {token}")
                 del self._singletons[token]

    def resolve(self, token: Any) -> Any:
        pid = threading.get_ident()
        is_optional = False; origin_token = token
        if getattr(token, '__origin__', None) is Union: # Handles Optional[T] as Union[T, NoneType]
            args = getattr(token, '__args__', ())
            if len(args) == 2 and type(None) in args:
                 is_optional = True
                 token = next(arg for arg in args if arg is not type(None))

        with self._lock:
            if token not in self._providers:
                if is_optional:
                    #self._container_logger.debug(f"[{pid}] No provider for OPTIONAL token: {origin_token}. Returning None.")
                    return None
                self._container_logger.error(f"[{pid}] No provider registered for REQUIRED token: {origin_token}")
                raise ValueError(f"No provider registered for required token: {origin_token}")
            provider, singleton_flag = self._providers[token]
            if singleton_flag and token in self._singletons:
                #self._container_logger.debug(f"[{pid}] Resolving singleton for token: {token}")
                return self._singletons[token]

            if not hasattr(self._resolving, 'tokens'): self._resolving.tokens = set()
            if token in self._resolving.tokens:
                cycle = list(self._resolving.tokens) + [token]
                cycle_str = ' -> '.join(map(str, cycle))
                self._container_logger.error(f"[{pid}] Dependency cycle detected: {cycle_str}")
                raise RecursionError(f"Dependency cycle detected: {cycle_str}")
            self._resolving.tokens.add(token)
            #self._container_logger.debug(f"[{pid}] Resolving token: {token} (Singleton: {singleton_flag})")

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
        #self._container_logger.debug(f"[{pid}] Resolved instance for token {token}: {type(instance).__name__}")
        return instance

    def get_registered_tokens(self) -> list[Any]:
        with self._lock: return list(self._providers.keys())


# --- Funciones Helper para creación de componentes complejos ---
def _create_stability_calculator(c: Container) -> Optional[BaseStabilityCalculator]:
    config = c.resolve(dict) # Main config
    logger_instance = c.resolve(logging.Logger)
    factory = c.resolve(RewardFactory)

    # Navegar a la sección stability_calculator
    stability_config = config.get('environment', {}).get('reward_setup', {}).get('calculation', {}).get('stability_calculator')

    if not isinstance(stability_config, dict) or not stability_config:
        logger_instance.info("[DIHelper:_create_stability_calculator] Config section absent/invalid. No instance created.")
        return None

    # Inferir 'enabled' por la presencia y validez de 'type'
    calculator_type = stability_config.get('type')
    if not calculator_type or not isinstance(calculator_type, str):
        logger_instance.info(f"[DIHelper:_create_stability_calculator] 'type' missing or invalid. No instance created (Type: {calculator_type}).")
        return None

    logger_instance.info(f"[DIHelper:_create_stability_calculator] Stability calculator TYPE '{calculator_type}' configured. Attempting creation...")
    try:
        # La factoría es responsable de extraer los params correctos (e.g., 'ira_params')
        # y pasarlos al constructor del calculador.
        instance = factory.create_stability_calculator(stability_config) # Pasar la sub-sección completa
        if instance is None: # La factoría puede devolver None si internamente decide no crear
            # Esto no debería ocurrir si type está presente, la factoría debería lanzar error.
            msg = "RewardFactory returned None for stability calculator with a valid 'type'."
            logger_instance.critical(f"[DIHelper:_create_stability_calculator] {msg}")
            raise RuntimeError(msg)
        logger_instance.info(f"[DIHelper:_create_stability_calculator] StabilityCalculator '{type(instance).__name__}' created successfully.")
        return instance
    except (ValueError, TypeError) as e_config:
        msg = f"Config error for Stability Calculator (type: {calculator_type}): {e_config}"
        logger_instance.critical(f"[DIHelper:_create_stability_calculator] {msg}", exc_info=True)
        raise RuntimeError(msg) from e_config
    except Exception as e_unexpected:
        msg = f"Unexpected error creating Stability Calculator (type: {calculator_type}): {e_unexpected}"
        logger_instance.critical(f"[DIHelper:_create_stability_calculator] {msg}", exc_info=True)
        raise RuntimeError(msg) from e_unexpected

def _create_reward_function_instance(c: Container) -> RewardFunction:
    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)
    factory = c.resolve(RewardFactory)

    reward_setup_config = config.get('environment', {}).get('reward_setup')
    if not isinstance(reward_setup_config, dict):
        raise ValueError("Config 'environment.reward_setup' not found or not a dictionary.")

    # Resolver dependencia BaseStabilityCalculator (puede ser None)
    stability_calculator_instance = c.resolve(Optional[BaseStabilityCalculator]) # Correctly resolve Optional
    #logger_instance.debug(f"[DIHelper:_create_reward_function] Resolved BaseStabilityCalculator: {type(stability_calculator_instance).__name__ if stability_calculator_instance else 'None'}")

    try:
        instance = factory.create_reward_function(reward_setup_config, stability_calculator_instance)
        logger_instance.info(f"[DIHelper:_create_reward_function] RewardFunction '{type(instance).__name__}' created.")
        return instance
    except Exception as e:
        logger_instance.critical(f"[DIHelper:_create_reward_function] Critical failure creating RewardFunction: {e}", exc_info=True)
        raise RuntimeError("Failed to create RewardFunction") from e

def _create_reward_strategy(c: Container) -> RewardStrategy:
    # Importar estrategias aquí para evitar ciclos si ellas importan Container (raro, pero posible)
    from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
    from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
    from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)

    strategy_config = config.get('environment', {}).get('reward_setup', {}).get('reward_strategy')
    if not isinstance(strategy_config, dict):
        raise ValueError("Config 'environment.reward_setup.reward_strategy' not found or not a dictionary.")

    strategy_type = strategy_config.get('type')
    if not strategy_type: raise ValueError("Missing 'type' in 'reward_strategy' config.")

    # Obtener el bloque de parámetros específico para la estrategia elegida
    # Ejemplo: strategy_params = {'global': {}, 'shadow_baseline': {'beta': 0.2, ...}, ...}
    all_strategy_params = strategy_config.get('strategy_params', {})
    if not isinstance(all_strategy_params, dict):
        raise TypeError("'reward_strategy.strategy_params' must be a dictionary.")

    specific_params = all_strategy_params.get(strategy_type, {}) # Obtener params para el tipo, o {}
    if not isinstance(specific_params, dict):
        logger_instance.warning(f"[DIHelper:_create_reward_strategy] Params for strategy '{strategy_type}' not a dict. Using {{}}.")
        specific_params = {}

    logger_instance.info(f"[DIHelper:_create_reward_strategy] Creating RewardStrategy: {strategy_type} with params: {specific_params.keys()}")
    strategy: RewardStrategy
    try:
        if strategy_type == 'global':
            strategy = GlobalRewardStrategy(**specific_params) # Pasar params, aunque global no use
        elif strategy_type == 'shadow_baseline':
            strategy = ShadowBaselineRewardStrategy(**specific_params) # beta y baseline_init_value
        elif strategy_type == 'echo_baseline':
            strategy = EchoBaselineRewardStrategy(**specific_params) # Pasar params, aunque echo no use
        else:
            raise ValueError(f"Unknown reward_strategy.type: '{strategy_type}'")
        logger_instance.info(f"[DIHelper:_create_reward_strategy] RewardStrategy '{type(strategy).__name__}' created.")
        return strategy
    except (ValueError, TypeError) as e: # Errores del constructor de la estrategia
        logger_instance.critical(f"[DIHelper:_create_reward_strategy] Error creating RewardStrategy '{strategy_type}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to create RewardStrategy '{strategy_type}': {e}") from e
    except Exception as e_unexp:
        logger_instance.critical(f"[DIHelper:_create_reward_strategy] Unexpected error creating RewardStrategy '{strategy_type}': {e_unexp}", exc_info=True)
        raise RuntimeError(f"Unexpected error creating RewardStrategy '{strategy_type}'") from e_unexp

def _create_virtual_simulator(c: Container) -> Optional[VirtualSimulator]:
    from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator # Import local
    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)

    # Verificar si la estrategia requiere un simulador virtual
    learn_strategy_type = config.get('environment', {}).get('reward_setup', {}).get('reward_strategy', {}).get('type')
    if learn_strategy_type != 'echo_baseline': # Actualmente solo Echo lo necesita
        logger_instance.info(f"[DIHelper:_create_virtual_simulator] Strategy '{learn_strategy_type}' does not require VirtualSimulator. None created.")
        return None

    logger_instance.info(f"[DIHelper:_create_virtual_simulator] Creating VirtualSimulator for '{learn_strategy_type}' strategy...")
    try:
        system = c.resolve(DynamicSystem)
        controller_template = c.resolve(Controller) # Plantilla para deepcopy
        reward_function = c.resolve(RewardFunction)
        dt_val = config.get('environment', {}).get('simulation', {}).get('dt')

        if dt_val is None or not isinstance(dt_val, (float, int)) or dt_val <= 0:
            raise ValueError(f"Invalid 'dt' ({dt_val}) from config 'environment.simulation.dt'. Must be positive number.")

        simulator = PendulumVirtualSimulator(
            system=system,
            controller=controller_template,
            reward_function=reward_function,
            dt=dt_val
        )
        logger_instance.info(f"[DIHelper:_create_virtual_simulator] PendulumVirtualSimulator created for '{learn_strategy_type}'.")
        return simulator
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger_instance.error(f"[DIHelper:_create_virtual_simulator] Error configuring VirtualSimulator for '{learn_strategy_type}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to create VirtualSimulator required for '{learn_strategy_type}': {e}") from e
    except Exception as e_unexp:
         logger_instance.error(f"[DIHelper:_create_virtual_simulator] Unexpected error creating VirtualSimulator: {e_unexp}", exc_info=True)
         raise RuntimeError("Unexpected error creating VirtualSimulator") from e_unexp


# --- Función build_container ---
def build_container(config: Dict[str, Any], vis_config: Optional[Dict[str, Any]]) -> Container:
    from simulation_manager import SimulationManager # Importar aquí
    from visualization_manager import VisualizationManager # Importar aquí

    container = Container()
    logger = container._container_logger # Usar el logger interno del contenedor

    # --- Registro Fundamental ---
    container.register(Container, lambda c: container) # Registrarse a sí mismo
    container.register(logging.Logger, lambda c: logging.getLogger(), singleton=True) # Logger raíz
    container.register(dict, lambda c: config, singleton=True) # Config principal
    container.register(VIS_CONFIG_TOKEN, lambda c: vis_config, singleton=True) # Vis_config (puede ser None)

    # --- Factorías ---
    container.register(SystemFactory, lambda c: SystemFactory(), singleton=True)
    container.register(ControllerFactory, lambda c: ControllerFactory(), singleton=True)
    container.register(AgentFactory, lambda c: AgentFactory(), singleton=True)
    container.register(RewardFactory, lambda c: RewardFactory(), singleton=True)
    container.register(EnvironmentFactory, lambda c: EnvironmentFactory(), singleton=True)

    # --- Componentes Principales (vía helpers) ---
    container.register(BaseStabilityCalculator, _create_stability_calculator, singleton=True) # Puede devolver None
    container.register(RewardFunction, _create_reward_function_instance, singleton=True)
    container.register(RewardStrategy, _create_reward_strategy, singleton=True)
    container.register(VirtualSimulator, _create_virtual_simulator, singleton=True) # Puede devolver None

    # --- Componentes con dependencias (Lambdas directas usando factorías y config) ---
    container.register(DynamicSystem, lambda c: c.resolve(SystemFactory).create_system(
        system_type=c.resolve(dict).get('environment', {}).get('system', {}).get('type', 'unknown_system'),
        system_params=c.resolve(dict).get('environment', {}).get('system', {}).get('params', {})),
        singleton=True
    )
    container.register(Controller, lambda c: c.resolve(ControllerFactory).create_controller(
        controller_type=c.resolve(dict).get('environment', {}).get('controller', {}).get('type', 'unknown_controller'),
        controller_params={ # Pasar todos los params del controller + dt
            **c.resolve(dict).get('environment', {}).get('controller', {}).get('params', {}),
            'dt': c.resolve(dict).get('environment', {}).get('simulation', {}).get('dt')
        }),
        singleton=True
    )
    container.register(RLAgent, lambda c: c.resolve(AgentFactory).create_agent(
        agent_type=c.resolve(dict).get('environment', {}).get('agent', {}).get('type', 'unknown_agent'),
        agent_params={ # Pasar todos los params del agente + pid_adaptation + params específicos de estrategia
            **c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}),
            'gain_step': c.resolve(dict).get('environment', {}).get('controller', {}).get('pid_adaptation', {}).get('gain_step'),
            'variable_step': c.resolve(dict).get('environment', {}).get('controller', {}).get('pid_adaptation', {}).get('variable_step'),
            # Otras configuraciones 
            'early_termination_config': c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}).get('early_termination', {}),
            'reward_strategy_instance': c.resolve(RewardStrategy), # Inyectar estrategia
            # Inyectar params de la estrategia de recompensa específica si es shadow_baseline
            'shadow_baseline_params': c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('reward_strategy', {}).get('strategy_params', {}).get('shadow_baseline')
                if c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('reward_strategy', {}).get('type') == 'shadow_baseline' else None,
        }),
        singleton=True
    )
    container.register(Environment, lambda c: c.resolve(EnvironmentFactory).create_environment(
        config=c.resolve(dict), # Pasar config completa
        reward_function_instance=c.resolve(RewardFunction),
        system_instance=c.resolve(DynamicSystem),
        controller_instance=c.resolve(Controller),
        agent_instance=c.resolve(RLAgent)),
        singleton=True
    )

    # --- Servicios de Soporte ---
    container.register(HeatmapGenerator, lambda c: HeatmapGenerator(c.resolve(logging.Logger)), singleton=True)
    container.register(PlotGenerator, lambda c: MatplotlibPlotGenerator(), singleton=True)
    container.register(ResultHandler, lambda c: ResultHandler(logger=c.resolve(logging.Logger)), singleton=True)
    container.register('simulation_manager.SimulationManager', lambda c: SimulationManager(
        logger=c.resolve(logging.Logger),
        result_handler=c.resolve(ResultHandler),
        container=c), # Inyectar el propio contenedor
        singleton=False # SimulationManager es transient (uno por `main` run)
    )
    container.register(MetricsCollector, lambda c: ExtendedMetricsCollector(), singleton=False) # Transient

    container.register(VisualizationManager, lambda c: VisualizationManager(
        logger_instance=c.resolve(logging.Logger),
        plot_generator=c.resolve(PlotGenerator),
        heatmap_generator=c.resolve(HeatmapGenerator),
        vis_config_data=c.resolve(VIS_CONFIG_TOKEN), # Puede ser None
        results_folder=c.resolve(str)), # Resuelve str (registrado en main)
        singleton=False # Transient
    )

    logger.info("DI Container built and all providers registered.")
    return container