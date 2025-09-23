import threading
import logging
from typing import Any, Callable, Dict, Type, Optional

# Interfaces and Factories
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.reward_function import RewardFunction
from interfaces.reward_strategy import RewardStrategy
from interfaces.stability_calculator import BaseStabilityCalculator
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector # Añadir interfaz

from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory
from factories.environment_factory import EnvironmentFactory

# Importar Servicios y Componentes concretos que se registrarán directamente o cuyas factorías se usarán
from result_handler import ResultHandler
from simulation_manager import SimulationManager
from heatmap_generator import HeatmapGenerator
# Asumimos que visualization.py se refactorizará a PlotGenerator en Paso 5
from utils.plot_generator import PlotGenerator 
# Asumimos que se creará/refactorizará ExtendedMetricsCollector en Paso 3
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector # <-- Descomentar y ajustar en Paso 3

# Placeholder classes hasta que se definan en pasos posteriores
class PlotGenerator: # Placeholder
    def __init__(self, logger: logging.Logger): pass
    def generate(self, *args, **kwargs): pass

class ExtendedMetricsCollector(MetricsCollector): # Placeholder
     def __init__(self): self._metrics = {}
     def log(self, metric_name, metric_value): self._metrics[metric_name] = metric_value # Simplificado
     def get_metrics(self): return self._metrics.copy()
     def reset(self, episode_id: int = -1): self._metrics = {}


class Container:
    """
    Contenedor simple para inyección de dependencias.
    Permite registro y resolución de servicios, con soporte opcional de singleton.
    """
    def __init__(self):
        # Mapa token -> (provider_fn, singleton_flag)
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        # Instancias singleton cache
        self._singletons: Dict[Any, Any] = {}
        # Lock para hilos
        self._lock = threading.Lock()

    def register(self, token: Any, provider: Callable[['Container'], Any], singleton: bool = False):
        """
        Registra un proveedor bajo un token.
        :param token: Clave de registro (puede ser clase o string).
        :param provider: Función provider(container) que devuelve instancia.
        :param singleton: Si True, se creará una única instancia.
        """
        with self._lock:
            self._providers[token] = (provider, singleton)
            # Si se re-registra un singleton, eliminar la instancia cacheada
            if singleton and token in self._singletons:
                del self._singletons[token]

    def resolve(self, token: Any, singleton: bool = False) -> Any:
        """
        Resuelve instancia para el token registrado.
        Si fue marcado como singleton, devuelve siempre la misma.
        Maneja Optional[Token] resolviendo Token o devolviendo None si no está registrado.
        """
         # Manejo especial para Optional[T]
        is_optional = False
        origin_token = token
        if hasattr(token, '__origin__') and token.__origin__ is Optional:
             is_optional = True
             # El token real es el primer argumento de Optional (e.g., Optional[VirtualSimulator] -> VirtualSimulator)
             token = token.__args__[0]

        with self._lock:
            if singleton and token in self._singletons:
                return self._singletons[token]

            if token not in self._providers:
                 # Si se pidió Optional y no está, devolver None
                 if is_optional:
                      return None
                 # Si no era Optional y no está, lanzar error
                 raise ValueError(f"No hay proveedor registrado para el token: {origin_token}")

            provider, singleton_flag = self._providers[token]

            # Verificar si ya se está resolviendo este token para evitar recursión infinita
            # (Esto es una protección básica, contenedores más complejos tienen manejo de ciclos)
            if hasattr(threading.current_thread(), '_resolving'):
                if token in threading.current_thread()._resolving: # type: ignore
                    raise RecursionError(f"Dependencia circular detectada resolviendo {token}")
                threading.current_thread()._resolving.add(token) # type: ignore
            else:
                threading.current_thread()._resolving = {token} # type: ignore

            try:
                instance = provider(self)
            finally:
                 # Asegurarse de limpiar el estado de resolución incluso si hay error
                 if hasattr(threading.current_thread(), '_resolving'):
                     threading.current_thread()._resolving.remove(token) # type: ignore
                     if not threading.current_thread()._resolving: # type: ignore
                          del threading.current_thread()._resolving # type: ignore


            if singleton_flag:
                self._singletons[token] = instance
            return instance

    def get_registered_tokens(self) -> list[Any]:
         """Devuelve una lista de todos los tokens registrados."""
         with self._lock:
              return list(self._providers.keys())


def build_container(config: Dict[str, Any]) -> Container:
    """
    Construye y devuelve un contenedor con todos los servicios y fábricas registrados.
    """
    container = Container()
    container.register(Container, lambda c: container) # Registrarse a sí mismo

    # --- Servicios Básicos ---
    container.register(logging.Logger, lambda c: logging.getLogger(), singleton=True)
    container.register(dict, lambda c: config, singleton=True)
    # results_folder se registra en main.py después de crear la carpeta

    # --- Fábricas (registradas como servicios singleton) ---
    container.register(SystemFactory, lambda c: SystemFactory(), singleton=True)
    container.register(ControllerFactory, lambda c: ControllerFactory(), singleton=True)
    container.register(AgentFactory, lambda c: AgentFactory(), singleton=True)
    container.register(RewardFactory, lambda c: RewardFactory(), singleton=True)
    container.register(EnvironmentFactory, lambda c: EnvironmentFactory(), singleton=True)

    # --- Componentes Principales (resueltos via interfaces y factorías) ---

    # Calculador de Estabilidad (puede ser None)
    # Usar Optional[BaseStabilityCalculator] permite resolver None si está deshabilitado
    container.register(
        Optional[BaseStabilityCalculator],
        lambda c: c.resolve(RewardFactory).create_stability_calculator(
            c.resolve(dict)['environment']['reward_setup']
        ),
        singleton=True # El calculador en sí mismo suele ser singleton
    )
    # Registrar también BaseStabilityCalculator directamente para casos donde se sabe que existe
    # Esto fallará si está deshabilitado y se resuelve BaseStabilityCalculator en lugar de Optional[...]
    container.register(
        BaseStabilityCalculator,
        lambda c: c.resolve(Optional[BaseStabilityCalculator]) or (_raise_if_none("StabilityCalculator (required but disabled/not found)")),
        singleton=True
    )


    # Función de Recompensa
    container.register(
        RewardFunction,
        lambda c: c.resolve(RewardFactory).create_reward_function(
            c.resolve(dict)['environment']['reward_setup'],
            # Pasa la instancia resuelta de Optional[BaseStabilityCalculator]
            c.resolve(Optional[BaseStabilityCalculator])
        ),
        singleton=True # La función de recompensa suele ser singleton
    )

    # Estrategia de Recompensa (depende de config, puede ser singleton o no)
    container.register(
        RewardStrategy,
        lambda c: _create_reward_strategy(c),
        singleton=True # Las estrategias suelen ser stateless, singleton es razonable
    )

    # Sistema Dinámico
    container.register(
        DynamicSystem,
        lambda c: c.resolve(SystemFactory).create_system(
            c.resolve(dict)['environment']['system']['type'],
            c.resolve(dict)['environment']['system']['params']
        ),
        singleton=True # El sistema suele ser singleton
    )

    # Controlador
    container.register(
        Controller,
        lambda c: c.resolve(ControllerFactory).create_controller(
            c.resolve(dict)['environment']['controller']['type'],
            c.resolve(dict)['environment']['controller']['params']
        ),
        singleton=True # El controlador suele ser singleton
    )

    # Agente RL
    container.register(
        RLAgent,
        lambda c: c.resolve(AgentFactory).create_agent(
            c.resolve(dict)['environment']['agent']['type'],
            # Pasar parámetros y la estrategia resuelta
            {
                **c.resolve(dict)['environment']['agent']['params'], # Copia params
                'reward_strategy_instance': c.resolve(RewardStrategy), # Inyectar estrategia
                 # Inyectar baseline_params si existen y la estrategia es shadow (esto puede ser mejorado)
                'shadow_baseline_params': c.resolve(dict)['environment']['reward_setup'].get('strategy_params', {}).get('shadow_baseline') if c.resolve(dict)['environment']['reward_setup'].get('learning_strategy') == 'shadow_baseline' else None,
            }
        ),
        singleton=True # El agente es stateful, debe ser singleton
    )

    # Entorno
    container.register(
        Environment,
        lambda c: c.resolve(EnvironmentFactory).create_environment(
            c.resolve(dict), # Pasar config completo
            c.resolve(RewardFunction) # Pasar RewardFunction
            # EnvironmentFactory resolverá System, Controller, Agent internamente
        ),
        singleton=True # El entorno principal suele ser singleton
    )

    # Simulador Virtual (Opcional)
    # Resolverá None si _create_virtual_simulator devuelve None
    container.register(
        Optional[VirtualSimulator],
        lambda c: _create_virtual_simulator(c),
        singleton=True # Si existe, suele ser singleton
    )
    # Registrar también VirtualSimulator directamente (fallará si no aplica o se resuelve None)
    container.register(
        VirtualSimulator,
        lambda c: c.resolve(Optional[VirtualSimulator]) or (_raise_if_none("VirtualSimulator (required but mode != echo_baseline)")),
        singleton=True
    )

    # --- Servicios de Soporte ---
    container.register(
         ResultHandler,
         # ResultHandler necesita el logger
         lambda c: ResultHandler(c.resolve(logging.Logger)),
         singleton=True
         )
    container.register(
        SimulationManager,
        # SimulationManager necesita logger, result_handler, y resolverá env, agent, etc.
        # Simplificamos su __init__ en Paso 3, por ahora solo logger y result_handler
        lambda c: SimulationManager(
             c.resolve(logging.Logger),
             c.resolve(ResultHandler)
             # En Paso 3, podríamos añadirle más dependencias o hacer que las resuelva internamente
         ),
         singleton=False # Nueva instancia por cada llamada a main (si fuera necesario)
    )
    container.register(
        HeatmapGenerator,
        lambda c: HeatmapGenerator(c.resolve(logging.Logger)),
        singleton=True
    )
    container.register( # PlotGenerator (refactorizado de visualization.py en Paso 5)
        PlotGenerator,
        lambda c: PlotGenerator(c.resolve(logging.Logger)),
        singleton=True
    )
    container.register( # Métricas (refactorizado de extended_metrics_collector.py en Paso 3)
         MetricsCollector, # Registrar contra la interfaz
         lambda c: ExtendedMetricsCollector(), # Crear instancia concreta
         singleton=False # Probablemente no singleton, se resetea por episodio
    )


    logging.info(f"Contenedor construido. Tokens registrados: {container.get_registered_tokens()}")
    return container


# --- Funciones Helper (Modificadas para usar DI) ---

def _create_reward_strategy(c: Container) -> RewardStrategy:
    # Importar aquí para evitar dependencias circulares a nivel de módulo
    from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
    from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
    from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

    # Obtener config y logger del contenedor
    config = c.resolve(dict)
    logger = c.resolve(logging.Logger)
    logger.debug("Creando RewardStrategy...")

    try:
        reward_setup_cfg = config['environment']['reward_setup']
        mode = reward_setup_cfg.get('learning_strategy', 'global')
        params = reward_setup_cfg.get('strategy_params', {})
        logger.info(f"Reward Strategy Mode seleccionado: {mode}")

        if mode == 'global':
            return GlobalRewardStrategy()
        elif mode == 'shadow_baseline':
            sbp = params.get('shadow_baseline', {})
            beta = sbp.get('beta', 0.1) # Obtener beta de los params
            logger.info(f"Creando ShadowBaselineRewardStrategy con beta={beta}")
            return ShadowBaselineRewardStrategy(beta=beta) # Pasar beta al constructor
        elif mode == 'echo_baseline':
            return EchoBaselineRewardStrategy()
        else:
            raise ValueError(f"learning_strategy desconocido: {mode}")
    except KeyError as e:
         logger.error(f"Falta clave en config al crear RewardStrategy: {e}. Config: {config.get('environment', {}).get('reward_setup', {})}")
         raise ValueError(f"Configuración inválida para RewardStrategy: falta {e}") from e
    except Exception as e:
         logger.error(f"Error inesperado creando RewardStrategy: {e}", exc_info=True)
         raise

def _create_virtual_simulator(c: Container) -> Optional[VirtualSimulator]:
    # Importar aquí para evitar dependencias circulares
    from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator

    # Obtener config y logger del contenedor
    config = c.resolve(dict)
    logger = c.resolve(logging.Logger)
    logger.debug("Intentando crear VirtualSimulator...")

    try:
        mode = config['environment']['reward_setup'].get('learning_strategy')
        if mode != 'echo_baseline':
            logger.info(f"VirtualSimulator no necesario (learning_strategy='{mode}').")
            return None

        logger.info(f"Creando PendulumVirtualSimulator (learning_strategy='{mode}').")
        # Resolver dependencias necesarias para el simulador virtual
        system = c.resolve(DynamicSystem)
        # Pasar el *controlador resuelto* como plantilla
        controller_template = c.resolve(Controller)
        reward_function = c.resolve(RewardFunction)
        dt = config['environment']['dt']

        # Crear instancia
        return PendulumVirtualSimulator(
            system=system,
            controller=controller_template, # Pasar el controlador real como plantilla
            reward_function=reward_function,
            dt=dt
        )
    except KeyError as e:
         logger.error(f"Falta clave en config al crear VirtualSimulator: {e}. Config: {config.get('environment', {})}")
         # Devolver None si la configuración es incorrecta para crearlo
         return None
    except ValueError as e: # Capturar errores de resolve si falta una dependencia
         logger.error(f"Error resolviendo dependencias para VirtualSimulator: {e}", exc_info=True)
         return None
    except Exception as e:
         logger.error(f"Error inesperado creando VirtualSimulator: {e}", exc_info=True)
         return None

def _raise_if_none(error_message: str):
     """Helper para lanzar error si un Optional resuelto es None cuando no debería serlo."""
     raise ValueError(error_message)