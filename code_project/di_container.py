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
from interfaces.metrics_collector import MetricsCollector

from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory
from factories.environment_factory import EnvironmentFactory

# Importar Servicios y Componentes concretos
from result_handler import ResultHandler
# from simulation_manager import SimulationManager # Causa ciclo, importar localmente
from heatmap_generator import HeatmapGenerator
from utils.visualization_generator import VisualizationGenerator # Mantener hasta renombrar en Paso 4
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector # Ejemplo de MetricsCollector concreto

# Mover importación de SimulationManager a TYPE_CHECKING o función build
if TYPE_CHECKING:
    from simulation_manager import SimulationManager


class Container:
    """ Contenedor simple DI (sin cambios respecto a la versión anterior) """
    def __init__(self):
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._resolving: threading.local = threading.local()
        self._container_logger = logging.getLogger('DI_Container')
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


# --- Funciones Helper para creación de componentes complejos ---

def _create_stability_calculator(c: Container) -> Optional[BaseStabilityCalculator]:
    """Crea el StabilityCalculator basado en la config 'reward_setup.stability_calculator'."""
    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)
    factory = c.resolve(RewardFactory)
    try:
        # Pasar la sección específica a la factory
        stability_config = config.get('environment', {}).get('reward_setup', {}).get('stability_calculator', {})
        return factory.create_stability_calculator(stability_config)
    except Exception as e:
        logger_instance.error(f"Error crítico creando StabilityCalculator vía factory: {e}", exc_info=True)
        # No relanzar aquí, la factory ya loguea y devuelve None si falla
        return None # Asegurar que devuelve None en error

def _create_reward_function_instance(c: Container) -> RewardFunction:
    """Crea la RewardFunction basada en la config 'reward_setup.calculation'."""
    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)
    factory = c.resolve(RewardFactory)
    try:
        # Resolver el StabilityCalculator primero (puede ser None)
        stability_calculator_instance = c.resolve(Optional[BaseStabilityCalculator])
        # Pasar la sección 'calculation' y el calculator resuelto a la factory
        calculation_config = config.get('environment', {}).get('reward_setup', {}).get('calculation', {})
        reward_setup_config_for_factory = config.get('environment', {}).get('reward_setup', {}) # Pasar reward_setup completo? RewardFactory lo usa
        # return factory.create_reward_function(calculation_config, stability_calculator_instance)
        return factory.create_reward_function(reward_setup_config_for_factory, stability_calculator_instance)

    except Exception as e:
        logger_instance.critical(f"Error crítico creando RewardFunction vía factory: {e}", exc_info=True)
        raise RuntimeError("Fallo al crear RewardFunction") from e # Relanzar error crítico

def _create_reward_strategy(c: Container) -> RewardStrategy:
    """Crea la RewardStrategy basada en la config 'reward_setup.learning_strategy'."""
    # Importar estrategias concretas aquí para evitar ciclos globales
    from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
    from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
    from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

    config = c.resolve(dict)
    logger_instance = c.resolve(logging.Logger)
    try:
        strategy_config = config.get('environment', {}).get('reward_setup', {}).get('learning_strategy', {})
        if not strategy_config: raise KeyError("Config 'environment.reward_setup.learning_strategy' no encontrada.")

        mode = strategy_config.get('type', 'global') # Obtener tipo de estrategia
        params = strategy_config.get('strategy_params', {}) # Obtener params específicos

        logger_instance.info(f"Creating RewardStrategy: {mode}")
        if mode == 'global':
            return GlobalRewardStrategy()
        elif mode == 'shadow_baseline':
            shadow_params = params.get('shadow_baseline', {}); beta = shadow_params.get('beta', 0.1)
            logger_instance.debug(f"ShadowBaseline strategy using beta={beta}")
            return ShadowBaselineRewardStrategy(beta=beta)
        elif mode == 'echo_baseline':
            echo_params_value = params.get('echo_baseline')
            # Validar y asegurar dict para desempaquetar (si hubiera params)
            echo_params_dict = echo_params_value if isinstance(echo_params_value, dict) else {}
            if echo_params_value is not None and not isinstance(echo_params_value, dict):
                 logger_instance.warning(f"echo_baseline params no es dict. Usando {{}}.")
            return EchoBaselineRewardStrategy(**echo_params_dict) # Desempaquetar
        else:
            raise ValueError(f"learning_strategy.type desconocido: '{mode}'")
    except KeyError as e: logger_instance.error(f"Error creando RewardStrategy: Falta {e}"); raise ValueError(f"Config inválida RewardStrategy: falta {e}") from e
    except ValueError as e: logger_instance.error(f"Error creando RewardStrategy: {e}"); raise
    except Exception as e: logger_instance.error(f"Error inesperado creando RewardStrategy: {e}", exc_info=True); raise RuntimeError(f"Error creando RewardStrategy: {e}") from e


def _create_virtual_simulator(c: Container) -> Optional[VisualizationGenerator]:
    """Crea el VirtualSimulator si la estrategia es 'echo_baseline'."""
    # Importar simulador concreto aquí
    from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator

    config = c.resolve(dict); logger_instance = c.resolve(logging.Logger)
    try:
        # Comprobar estrategia de aprendizaje
        learn_strategy_type = config.get('environment', {}).get('reward_setup', {}).get('learning_strategy', {}).get('type')
        if learn_strategy_type != 'echo_baseline':
            logger_instance.debug(f"VirtualSimulator no requerido (estrategia: {learn_strategy_type}).")
            return None

        logger_instance.info("Estrategia 'echo_baseline'. Creando VirtualSimulator...")
        # Resolver dependencias necesarias para el simulador
        system = c.resolve(DynamicSystem); controller_template = c.resolve(Controller)
        reward_function = c.resolve(RewardFunction); dt = config.get('environment', {}).get('dt')
        if dt is None: raise KeyError("'environment.dt' no encontrado.")

        logger_instance.debug("Creando PendulumVirtualSimulator instance...")
        simulator = PendulumVirtualSimulator( system=system, controller=controller_template, reward_function=reward_function, dt=dt )
        logger_instance.info("PendulumVirtualSimulator creado.")
        return simulator
    except KeyError as e: logger_instance.error(f"Falta config creando VirtualSimulator: {e}"); return None
    except ValueError as e: logger_instance.error(f"Error resolviendo dependencia para VirtualSimulator: {e}"); return None
    except Exception as e: logger_instance.error(f"Error inesperado creando VirtualSimulator: {e}"); return None


# --- Función build_container (Ajustada para usar funciones helper) ---
def build_container(config: Dict[str, Any]) -> Container:
    """ Construye y configura el contenedor DI. """
    # Importar SimulationManager aquí para usarlo en el registro
    from simulation_manager import SimulationManager

    container = Container()

    # --- Registro Fundamental ---
    container.register(Container, lambda c: container)
    container.register(logging.Logger, lambda c: logging.getLogger(), singleton=True)
    container.register(dict, lambda c: config, singleton=True)
    # Nota: results_folder NO se registra aquí, se hace desde main.py después de crearlo.

    # --- Factorías ---
    container.register(SystemFactory, lambda c: SystemFactory(), singleton=True)
    container.register(ControllerFactory, lambda c: ControllerFactory(), singleton=True)
    container.register(AgentFactory, lambda c: AgentFactory(), singleton=True)
    container.register(RewardFactory, lambda c: RewardFactory(), singleton=True)
    container.register(EnvironmentFactory, lambda c: EnvironmentFactory(), singleton=True)

    # --- Componentes Principales (Usando funciones helper) ---
    container.register(Optional[BaseStabilityCalculator], _create_stability_calculator, singleton=True)
    container.register(RewardFunction, _create_reward_function_instance, singleton=True)
    container.register(RewardStrategy, _create_reward_strategy, singleton=True)
    container.register(Optional[VirtualSimulator], _create_virtual_simulator, singleton=True)

    # --- Componentes con dependencias (Lambdas directas) ---
    container.register(DynamicSystem, lambda c: c.resolve(SystemFactory).create_system(
        system_type=c.resolve(dict).get('environment', {}).get('system', {}).get('type', 'unknown'),
        system_params=c.resolve(dict).get('environment', {}).get('system', {}).get('params', {})),
        singleton=True
    )
    container.register(Controller, lambda c: c.resolve(ControllerFactory).create_controller(
        controller_type=c.resolve(dict).get('environment', {}).get('controller', {}).get('type', 'unknown'),
        controller_params={
            **c.resolve(dict).get('environment', {}).get('controller', {}).get('params', {}),
            'dt': c.resolve(dict).get('environment', {}).get('dt') # dt inyectado aquí
        }),
        singleton=True
    )
    container.register(RLAgent, lambda c: c.resolve(AgentFactory).create_agent(
        agent_type=c.resolve(dict).get('environment', {}).get('agent', {}).get('type', 'unknown'),
        agent_params={
            **c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}),
            'gain_step': c.resolve(dict).get('pid_adaptation', {}).get('gain_step'),
            'variable_step': c.resolve(dict).get('pid_adaptation', {}).get('variable_step'),
            'reward_strategy_instance': c.resolve(RewardStrategy), # Estrategia resuelta
            # Pasar params de estrategia específicos si se necesitan directamente en el agente (ej Shadow)
            'shadow_baseline_params': c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('learning_strategy', {}).get('strategy_params', {}).get('shadow_baseline')
                                      if c.resolve(dict).get('environment', {}).get('reward_setup', {}).get('learning_strategy', {}).get('type') == 'shadow_baseline'
                                      else None,
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
    container.register(HeatmapGenerator, lambda c: HeatmapGenerator(c.resolve(logging.Logger)), singleton=True)
    # Se usa PlotGenerator hasta renombrar en Paso 4
    container.register(VisualizationGenerator, lambda c: VisualizationGenerator(c.resolve(logging.Logger)), singleton=True)
    # ResultHandler necesita el logger y ahora potencialmente PlotGenerator si se quiere integrar
    container.register(ResultHandler, lambda c: ResultHandler(
        logger=c.resolve(logging.Logger),
        heatmap_generator=c.resolve(HeatmapGenerator)), # Pasar heatmap generator
        singleton=True
    )
    container.register(SimulationManager, lambda c: SimulationManager(
        logger=c.resolve(logging.Logger),
        result_handler=c.resolve(ResultHandler),
        container=c), # Pasar el propio contenedor
        singleton=False # Transient: nueva instancia por ejecución
    )
    container.register(MetricsCollector, lambda c: ExtendedMetricsCollector(), singleton=False) # Transient

    container._container_logger.info("DI Container built and providers registered.")
    return container