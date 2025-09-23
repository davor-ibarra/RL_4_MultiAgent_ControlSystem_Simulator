import threading
import logging
from typing import Any, Callable, Dict, Type, Optional, cast, List, Set, TYPE_CHECKING

# Interfaces and Factories
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector # Interfaz

from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory
from factories.environment_factory import EnvironmentFactory

# Importar Servicios y Componentes concretos
from result_handler import ResultHandler
# Evitar importación directa de SimulationManager aquí para romper ciclo
# from simulation_manager import SimulationManager # <-- Causa ciclo
from heatmap_generator import HeatmapGenerator
# Placeholder para PlotGenerator
from utils.plot_generator import PlotGenerator

# Mover importación de SimulationManager a TYPE_CHECKING o función build
if TYPE_CHECKING:
    from simulation_manager import SimulationManager


class ExtendedMetricsCollector(MetricsCollector):
     _instance_count = 0
     def __init__(self):
         ExtendedMetricsCollector._instance_count += 1; self._metrics: Dict[str, List[Any]] = {};
         self._logger = logging.getLogger(f"{__name__}.Instance_{ExtendedMetricsCollector._instance_count}")
     def log(self, metric_name, metric_value): self._metrics.setdefault(metric_name, []).append(metric_value)
     def get_metrics(self): return self._metrics.copy()
     def reset(self, episode_id: int = -1): self._metrics = {}


class Container:
    """ Contenedor simple DI (sin cambios respecto a la versión anterior) """
    # ... (Código de la clase Container sin cambios) ...
    def __init__(self):
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}
        self._lock = threading.Lock() # Lock para proteger acceso concurrente a providers/singletons
        self._resolving: threading.local = threading.local() # Thread-local storage para detección de ciclos
        self._container_logger = logging.getLogger('DI_Container') # Logger específico del contenedor
        if not self._container_logger.hasHandlers():
            handler = logging.StreamHandler(); formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'); handler.setFormatter(formatter); self._container_logger.addHandler(handler)
            self._container_logger.setLevel(logging.INFO); self._container_logger.propagate = False

    def register(self, token: Any, provider: Callable[['Container'], Any], singleton: bool = False):
        with self._lock:
            self._providers[token] = (provider, singleton)
            if singleton and token in self._singletons: del self._singletons[token]

    def resolve(self, token: Any) -> Any:
        pid = threading.get_ident()
        is_optional = False; origin_token = token
        if getattr(token, '__origin__', None) is Optional:
             is_optional = True; args = getattr(token, '__args__', None)
             if args and len(args) == 1: token = args[0]
             else: raise ValueError(f"No se pudo desempaquetar Optional: {origin_token}")
        with self._lock:
            if token not in self._providers:
                if is_optional: return None
                raise ValueError(f"No provider registered for required token: {origin_token}")
            provider, singleton_flag = self._providers[token]
            if singleton_flag and token in self._singletons: return self._singletons[token]
            if not hasattr(self._resolving, 'tokens'): self._resolving.tokens = set()
            if token in self._resolving.tokens:
                 cycle = list(self._resolving.tokens) + [token]; cycle_str = ' -> '.join(map(str, cycle))
                 raise RecursionError(f"Dependency cycle detected: {cycle_str}")
            self._resolving.tokens.add(token)
        instance = None
        try: instance = provider(self)
        except Exception as e:
             self._container_logger.error(f"[{pid}] Error executing provider for token {token}: {e}", exc_info=True)
             if hasattr(self._resolving, 'tokens') and token in self._resolving.tokens:
                  self._resolving.tokens.remove(token)
                  if not self._resolving.tokens: del self._resolving.tokens
             raise
        finally:
             if hasattr(self._resolving, 'tokens'):
                  if token in self._resolving.tokens: self._resolving.tokens.remove(token)
                  if not self._resolving.tokens: del self._resolving.tokens
        if singleton_flag:
            with self._lock: self._singletons[token] = instance
        return instance

    def get_registered_tokens(self) -> list[Any]:
         with self._lock: return list(self._providers.keys())


# --- Función build_container (CON LAMBDA RLAgent CORREGIDA) ---
def build_container(config: Dict[str, Any]) -> Container:
    """ Construye y configura el contenedor DI. """
    # Importar SimulationManager aquí para usarlo en el registro
    from simulation_manager import SimulationManager

    container = Container()
    # container._container_logger.setLevel(logging.DEBUG) # Habilitar DEBUG

    # --- Registro Fundamental ---
    container.register(Container, lambda c: container)
    container.register(logging.Logger, lambda c: logging.getLogger(), singleton=True)
    container.register(dict, lambda c: config, singleton=True)

    # --- Factorías ---
    container.register(SystemFactory, lambda c: SystemFactory(), singleton=True)
    container.register(ControllerFactory, lambda c: ControllerFactory(), singleton=True)
    container.register(AgentFactory, lambda c: AgentFactory(), singleton=True)
    container.register(RewardFactory, lambda c: RewardFactory(), singleton=True)
    container.register(EnvironmentFactory, lambda c: EnvironmentFactory(), singleton=True)

    # --- Componentes Principales ---
    container.register( Optional[BaseStabilityCalculator], lambda c: c.resolve(RewardFactory).create_stability_calculator( c.resolve(dict).get('environment', {}).get('reward_setup', {})), singleton=True )
    container.register( RewardFunction, lambda c: c.resolve(RewardFactory).create_reward_function( c.resolve(dict).get('environment', {}).get('reward_setup', {}), c.resolve(Optional[BaseStabilityCalculator])), singleton=True )
    container.register(RewardStrategy, lambda c: _create_reward_strategy(c), singleton=True)
    container.register( DynamicSystem, lambda c: c.resolve(SystemFactory).create_system( c.resolve(dict).get('environment', {}).get('system', {}).get('type', 'unknown'), c.resolve(dict).get('environment', {}).get('system', {}).get('params', {})), singleton=True )
    # --- CORRECCIÓN LAMBDA CONTROLLER: Añadir dt explícitamente ---
    container.register(
        Controller,
        lambda c: c.resolve(ControllerFactory).create_controller(
            controller_type=c.resolve(dict).get('environment', {}).get('controller', {}).get('type', 'unknown'),
            controller_params={
                 # Parámetros específicos del controlador
                 **c.resolve(dict).get('environment', {}).get('controller', {}).get('params', {}),
                 # Añadir dt desde la sección environment
                 'dt': c.resolve(dict).get('environment', {}).get('dt')
            }
        ),
        singleton=True
    )
    # --- CORRECCIÓN LAMBDA RLAGENT: Incluir gain_step y variable_step ---
    container.register(
        RLAgent,
        lambda c: c.resolve(AgentFactory).create_agent(
            agent_type=c.resolve(dict).get('environment', {}).get('agent', {}).get('type', 'unknown'),
            agent_params={
                # Parámetros de environment.agent.params
                **c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}),
                # Parámetros de pid_adaptation <-- ¡AÑADIDOS!
                'gain_step': c.resolve(dict).get('pid_adaptation', {}).get('gain_step'),
                'variable_step': c.resolve(dict).get('pid_adaptation', {}).get('variable_step'),
                # Instancia de RewardStrategy resuelta
                'reward_strategy_instance': c.resolve(RewardStrategy),
                # Parámetros específicos de la estrategia
                'shadow_baseline_params': c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('strategy_params', {}).get('shadow_baseline')
                                          if c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('learning_strategy') == 'shadow_baseline'
                                          else None,
            }
        ),
        singleton=True
    )
    container.register( Optional[VirtualSimulator], lambda c: _create_virtual_simulator(c), singleton=True )
    container.register( Environment, lambda c: c.resolve(EnvironmentFactory).create_environment( config=c.resolve(dict), reward_function_instance=c.resolve(RewardFunction), system_instance=c.resolve(DynamicSystem), controller_instance=c.resolve(Controller), agent_instance=c.resolve(RLAgent)), singleton=True )

    # --- Servicios de Soporte ---
    container.register(HeatmapGenerator, lambda c: HeatmapGenerator(c.resolve(logging.Logger)), singleton=True)
    container.register( ResultHandler, lambda c: ResultHandler( logger=c.resolve(logging.Logger), heatmap_generator=c.resolve(HeatmapGenerator)), singleton=True )
    # Usar la clase SimulationManager importada localmente
    container.register( SimulationManager, lambda c: SimulationManager( logger=c.resolve(logging.Logger), result_handler=c.resolve(ResultHandler), container=c), singleton=False )
    container.register(PlotGenerator, lambda c: PlotGenerator(c.resolve(logging.Logger)), singleton=True)
    container.register( MetricsCollector, lambda c: ExtendedMetricsCollector(), singleton=False )

    container._container_logger.info("DI Container built and providers registered.")
    return container

# --- Funciones Helper (_create_reward_strategy, _create_virtual_simulator) ---
# (Sin cambios respecto a la versión anterior)
def _create_reward_strategy(c: Container) -> RewardStrategy:
    from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
    from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
    from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy
    config = c.resolve(dict); logger_instance = c.resolve(logging.Logger)
    try:
        reward_setup_cfg = config.get('environment', {}).get('reward_setup', {})
        if not reward_setup_cfg: raise KeyError("Config 'environment.reward_setup' no encontrada.")
        mode = reward_setup_cfg.get('learning_strategy', 'global')
        params = reward_setup_cfg.get('strategy_params', {})
        logger_instance.info(f"Creating RewardStrategy: {mode}")
        if mode == 'global': return GlobalRewardStrategy()
        elif mode == 'shadow_baseline':
            shadow_params = params.get('shadow_baseline', {}); beta = shadow_params.get('beta', 0.1)
            logger_instance.debug(f"ShadowBaseline strategy using beta={beta}")
            return ShadowBaselineRewardStrategy(beta=beta)
        elif mode == 'echo_baseline':
            echo_params_value = params.get('echo_baseline') # Obtener el valor (podría ser None)
            # Asegurar que sea un diccionario antes de desempaquetar
            if not isinstance(echo_params_value, dict):
                echo_params_dict = {} # Usar dict vacío si es None o no es dict
                if echo_params_value is not None: # Advertir si no era None pero tampoco dict
                     logger_instance.warning(f"echo_baseline_params no es dict (valor: {echo_params_value}). Usando {{}}.")
            else:
                echo_params_dict = echo_params_value # Usar el dict si es válido
            # Ahora desempaquetar el diccionario garantizado
            return EchoBaselineRewardStrategy(**echo_params_dict)
        else: raise ValueError(f"learning_strategy desconocido: '{mode}'")
    except KeyError as e: logger_instance.error(f"Error creando RewardStrategy: Falta {e}", exc_info=True); raise ValueError(f"Config inválida RewardStrategy: falta {e}") from e
    except Exception as e: logger_instance.error(f"Error inesperado creando RewardStrategy: {e}", exc_info=True); raise RuntimeError(f"Error creando RewardStrategy: {e}") from e

def _create_virtual_simulator(c: Container) -> Optional[VirtualSimulator]:
    from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator
    config = c.resolve(dict); logger_instance = c.resolve(logging.Logger)
    try:
        reward_setup_cfg = config.get('environment', {}).get('reward_setup', {})
        mode = reward_setup_cfg.get('learning_strategy')
        if mode != 'echo_baseline': logger_instance.debug("VirtualSimulator no requerido ({mode})."); return None
        logger_instance.info("Estrategia 'echo_baseline'. Creando VirtualSimulator...")
        system = c.resolve(DynamicSystem); controller_template = c.resolve(Controller)
        reward_function = c.resolve(RewardFunction); dt = config.get('environment', {}).get('dt')
        if dt is None: raise KeyError("'environment.dt' no encontrado.")
        logger_instance.debug("Creando PendulumVirtualSimulator instance...")
        simulator = PendulumVirtualSimulator( system=system, controller=controller_template, reward_function=reward_function, dt=dt )
        logger_instance.info("PendulumVirtualSimulator creado.")
        return simulator
    except KeyError as e: logger_instance.error(f"Falta config creando VirtualSimulator: {e}", exc_info=True); return None
    except ValueError as e: logger_instance.error(f"Error resolviendo dependencia para VirtualSimulator: {e}", exc_info=True); return None
    except Exception as e: logger_instance.error(f"Error inesperado creando VirtualSimulator: {e}", exc_info=True); return None