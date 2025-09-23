# di_container.py
import threading
import logging
from typing import Any, Callable, Dict, Type, Optional, Union # Quitamos cast, List, Set, TYPE_CHECKING si no se usan directamente

# Interfaces (solo para type hints si son necesarias, pero los tokens pueden ser clases)
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

# Factorías
from factories.system_factory import SystemFactory
from factories.controller_factory import ControllerFactory
from factories.agent_factory import AgentFactory
from factories.reward_factory import RewardFactory, NullStabilityCalculator # Importar Null aquí si se registra aquí
from factories.environment_factory import EnvironmentFactory

# Componentes Concretos (para registro en factorías y algunos helpers)
from utils.data.result_handler import ResultHandler
from utils.data.heatmap_generator import HeatmapGenerator
from components.plotting.matplotlib_plot_generator import MatplotlibPlotGenerator
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector
from simulation_manager import SimulationManager # Para token de string
from visualization_manager import VisualizationManager # Para token de string

# Componentes específicos que se registran en las factorías
from components.agents.pid_qlearning_agent import PIDQLearningAgent
from components.environments.pendulum_environment import PendulumEnvironment
from components.controllers.pid_controller import PIDController
from components.systems.inverted_pendulum_system import InvertedPendulumSystem
from components.rewards.instantaneous_reward_calculator import InstantaneousRewardCalculator
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy
from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator


# Esto es aceptable si son pocos y bien conocidos.
VIS_CONFIG_TOKEN_STR = "visualization_config_dict_token"
PROCESSED_DATA_DIRECTIVES_TOKEN_STR = "processed_data_directives_dict_token"
OUTPUT_DIR_TOKEN_STR = "output_dir_path_token"
SIMULATION_MANAGER_TOKEN_STR = "simulation_manager_service_token"
# VISUALIZATION_MANAGER_TOKEN_STR = "visualization_manager_service_token" # Se usa la clase como token

class Container:
    def __init__(self):
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._resolving_tracker: threading.local = threading.local()
        self._container_logger = logging.getLogger(f'{__name__}.DIContainer[{id(self)}]')
        self._container_logger.info("DI Container instance created.")

    def register(self, token: Any, provider: Callable[['Container'], Any], singleton: bool = False):
        with self._lock:
            if token in self._providers:
                log_level = logging.WARNING if self._providers[token][1] or not singleton else logging.DEBUG
                self._container_logger.log(log_level, f"Overwriting provider for token: {token} (New Singleton: {singleton})")
            self._providers[token] = (provider, singleton)
            if token in self._singletons: # Si se re-registra, eliminar singleton viejo
                del self._singletons[token]

    def resolve(self, token: Any) -> Any:
        # Manejo de Optional[Type] como token
        is_optional = False
        actual_token = token
        if getattr(token, '__origin__', None) is Union:
            args = getattr(token, '__args__', ())
            if len(args) == 2 and type(None) in args:
                is_optional = True
                actual_token = next(arg for arg in args if arg is not type(None))
        
        with self._lock:
            if actual_token not in self._providers:
                if is_optional:
                    # self._container_logger.debug(f"No provider for OPTIONAL token: {token}. Returning None.")
                    return None
                self._container_logger.error(f"No provider registered for REQUIRED token: {token} (resolved as {actual_token})")
                raise ValueError(f"No provider for required token: {token}")
            
            provider_func, is_singleton = self._providers[actual_token]
            if is_singleton and actual_token in self._singletons:
                return self._singletons[actual_token]

            if not hasattr(self._resolving_tracker, 'current_set'):
                self._resolving_tracker.current_set = set()
            
            if actual_token in self._resolving_tracker.current_set:
                cycle_path = list(self._resolving_tracker.current_set) + [actual_token]
                self._container_logger.error(f"Dependency cycle detected: {' -> '.join(map(str, cycle_path))}")
                raise RecursionError(f"Dependency cycle: {' -> '.join(map(str, cycle_path))}")
            
            self._resolving_tracker.current_set.add(actual_token)
            # self._container_logger.debug(f"Resolving token: {actual_token} (Singleton: {is_singleton})")

        instance = None
        try:
            instance = provider_func(self)
        except Exception as e:
            self._container_logger.error(f"Error executing provider for token {actual_token}: {e}", exc_info=True)
            # Limpiar tracker antes de re-lanzar
            if hasattr(self._resolving_tracker, 'current_set') and actual_token in self._resolving_tracker.current_set:
                self._resolving_tracker.current_set.remove(actual_token)
                if not self._resolving_tracker.current_set: delattr(self._resolving_tracker, 'current_set')
            raise
        finally:
            if hasattr(self._resolving_tracker, 'current_set'):
                if actual_token in self._resolving_tracker.current_set: self._resolving_tracker.current_set.remove(actual_token)
                if not self._resolving_tracker.current_set: delattr(self._resolving_tracker, 'current_set')

        if is_singleton:
            with self._lock: self._singletons[actual_token] = instance
        
        # self._container_logger.debug(f"Resolved instance for token {actual_token}: {type(instance).__name__}")
        return instance

# --- Helpers de Creación (para simplificar lambdas en build_container) ---

def _create_stability_calculator_helper(c: Container) -> BaseStabilityCalculator:
    """Helper para crear BaseStabilityCalculator usando RewardFactory."""
    # RewardFactory ya no crea StabilityCalculator. StabilityCalculator se crea directamente.
    # Se resuelve la config y se decide qué tipo de StabilityCalculator crear.
    config = c.resolve(dict)
    # logger = c.resolve(logging.Logger) # No es necesario aquí si el constructor del calc loguea

    stability_measure_cfg = config.get('environment', {}).get('reward_setup', {}).get('calculation', {}).get('stability_measure')
    
    if not isinstance(stability_measure_cfg, dict) or not stability_measure_cfg:
        # logger.info("[DIHelper:_create_stability_calc] Stability measure config absent/invalid. Creating NullStabilityCalculator.")
        return NullStabilityCalculator() # Sin params

    calc_type = stability_measure_cfg.get('type')
    # Registrar tipos concretos directamente en DI o usar una mini-factoría aquí si crece mucho
    if calc_type == 'ira_zscore_metric':
        params = stability_measure_cfg.get('ira_zscore_metric_params', {})
        return IRAStabilityCalculator(params)
    elif calc_type == 'exp_decay_metric':
        params = stability_measure_cfg.get('exp_decay_metric_params', {})
        return SimpleExponentialStabilityCalculator(params)
    else:
        # logger.warning(f"[DIHelper:_create_stability_calc] Unknown stability_measure type '{calc_type}'. Using NullStabilityCalculator.")
        return NullStabilityCalculator()

def _create_reward_function_helper(c: Container) -> RewardFunction:
    """Helper para crear RewardFunction usando RewardFactory."""
    logger_instance = c.resolve(logging.Logger)
    config = c.resolve(dict)
    reward_factory = c.resolve(RewardFactory)
    stability_calculator = c.resolve(BaseStabilityCalculator) # Resuelta

    # Obtener la sección 'calculation' de la configuración
    reward_setup_cfg_section = config.get('environment', {}).get('reward_setup', {})
    if not isinstance(reward_setup_cfg_section, dict) or not reward_setup_cfg_section:
        raise ValueError("DIHelper: Config 'environment.reward_setup' section must be a valid dictionary.")
    
    logger_instance.debug(f"[DIHelper:_create_RF_helper] Passing all 'reward_setup' section to RewardFactory. Keys: {list(reward_setup_cfg_section.keys())}")
    return reward_factory.create_reward_function(reward_setup_cfg_section, stability_calculator)

def _create_reward_strategy_helper(c: Container) -> RewardStrategy:
    """Helper para crear RewardStrategy usando RewardFactory."""
    config = c.resolve(dict)
    reward_factory = c.resolve(RewardFactory)
    
    reward_strategy_cfg_section = config.get('environment', {}).get('reward_setup', {}).get('reward_strategy')
    if not isinstance(reward_strategy_cfg_section, dict):
        raise ValueError("DIHelper: Config 'environment.reward_setup.reward_strategy' missing or not a dict for RewardStrategy.")
        
    return reward_factory.create_reward_strategy(reward_strategy_cfg_section)

def _create_virtual_simulator_helper(c: Container) -> Optional[VirtualSimulator]:
    """Helper para crear VirtualSimulator si es necesario."""
    config = c.resolve(dict)
    reward_strategy = c.resolve(RewardStrategy) # Resolver la estrategia para chequear 'needs_virtual_simulation'

    if not reward_strategy.needs_virtual_simulation:
        # c.resolve(logging.Logger).info("[DIHelper:_create_virtual_sim] Strategy does not require VirtualSimulator. None created.")
        return None

    # logger = c.resolve(logging.Logger)
    # logger.info("[DIHelper:_create_virtual_sim] Creating VirtualSimulator...")
    
    system_tpl = c.resolve(DynamicSystem)
    controller_tpl = c.resolve(Controller)
    reward_func_tpl = c.resolve(RewardFunction)
    stability_calc_tpl = c.resolve(BaseStabilityCalculator) # <<< NECESARIO para el nuevo PendulumVirtualSimulator
    dt_val = config.get('environment', {}).get('simulation', {}).get('dt_sec')

    # Asumimos que PendulumVirtualSimulator es el único tipo por ahora.
    # Si hubiera más, se necesitaría una VirtualSimulatorFactory.
    return PendulumVirtualSimulator(
        system_template=system_tpl,
        controller_template=controller_tpl,
        reward_function_template=reward_func_tpl,
        stability_calculator_template=stability_calc_tpl, # <<< PASARLO
        dt_sec_value=dt_val
    )

# --- Función Principal de Construcción del Contenedor ---
def build_container(main_config: Dict[str, Any],
                    vis_config: Optional[Dict[str, Any]],
                    processed_data_directives: Optional[Dict[str, Any]],
                    output_dir: str # Recibir output_dir directamente
                   ) -> Container:
    
    from simulation_manager import SimulationManager
    from visualization_manager import VisualizationManager

    container_instance = Container()
    # container_logger = container_instance._container_logger # Para logging interno del build

    # 1. Registro Fundamental (singletons)
    container_instance.register(Container, lambda c_self: container_instance, singleton=True)
    container_instance.register(logging.Logger, lambda c_log: logging.getLogger(), singleton=True) # Logger raíz o específico
    container_instance.register(dict, lambda c_cfg: main_config, singleton=True) # Config principal
    container_instance.register(VIS_CONFIG_TOKEN_STR, lambda c_vis_cfg: vis_config, singleton=True)
    container_instance.register(PROCESSED_DATA_DIRECTIVES_TOKEN_STR, lambda c_pdd_cfg: processed_data_directives if processed_data_directives else {}, singleton=True)
    container_instance.register(OUTPUT_DIR_TOKEN_STR, lambda c_out_dir: output_dir, singleton=True)


    # 2. Factorías (singletons)
    agent_f = AgentFactory()
    agent_f.register_agent_type('pid_qlearning', PIDQLearningAgent)
    container_instance.register(AgentFactory, lambda c: agent_f, singleton=True)

    env_f = EnvironmentFactory()
    env_f.register_environment_type('pendulum_environment', PendulumEnvironment)
    container_instance.register(EnvironmentFactory, lambda c: env_f, singleton=True)

    ctrl_f = ControllerFactory()
    ctrl_f.register_controller_type('pid', PIDController)
    container_instance.register(ControllerFactory, lambda c: ctrl_f, singleton=True)

    sys_f = SystemFactory()
    sys_f.register_system_type('inverted_pendulum', InvertedPendulumSystem)
    container_instance.register(SystemFactory, lambda c: sys_f, singleton=True)
    
    reward_f = RewardFactory()
    # RewardFactory ya no maneja StabilityCalculators. Estos se registran directamente o vía helper.
    reward_f.register_reward_function_type('default_instantaneous_reward', InstantaneousRewardCalculator)
    reward_f.register_reward_strategy_type('weighted_sum_features', GlobalRewardStrategy)
    reward_f.register_reward_strategy_type('shadow_baseline_delta', ShadowBaselineRewardStrategy)
    reward_f.register_reward_strategy_type('echo_virtual_baseline_delta', EchoBaselineRewardStrategy)
    container_instance.register(RewardFactory, lambda c: reward_f, singleton=True)

    # 3. Componentes Principales (singletons, usando helpers o factorías)
    container_instance.register(BaseStabilityCalculator, _create_stability_calculator_helper, singleton=True)
    container_instance.register(RewardFunction, _create_reward_function_helper, singleton=True)
    container_instance.register(RewardStrategy, _create_reward_strategy_helper, singleton=True)
    container_instance.register(Optional[VirtualSimulator], _create_virtual_simulator_helper, singleton=True)

    container_instance.register(DynamicSystem, lambda c: c.resolve(SystemFactory).create_system(
        system_type=c.resolve(dict).get('environment', {}).get('system', {}).get('type', 'unknown_system'), # Fallback
        system_params=c.resolve(dict).get('environment', {}).get('system', {}).get('params', {})
    ), singleton=True)

    container_instance.register(Controller, lambda c: c.resolve(ControllerFactory).create_controller(
        controller_type=c.resolve(dict).get('environment', {}).get('controller', {}).get('type', 'unknown_controller'),
        controller_params={
            **c.resolve(dict).get('environment', {}).get('controller', {}).get('params', {}), # Incluye anti_windup
            'dt_sec': c.resolve(dict).get('environment', {}).get('simulation', {}).get('dt_sec')
        }
    ), singleton=True)

    container_instance.register(RLAgent, lambda c: c.resolve(AgentFactory).create_agent(
        agent_type=c.resolve(dict).get('environment', {}).get('agent', {}).get('type', 'unknown_agent'),
        agent_constructor_params={ # Todos los params que el constructor del agente espera
            **c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}), # Params de config.agent.params
            'reward_strategy': c.resolve(RewardStrategy), # Instancia resuelta
            # Extraer gain_delta y per_gain_delta de la sección controller.pid_adaptation
            'gain_delta': c.resolve(dict).get('environment', {}).get('controller', {}).get('pid_adaptation', {}).get('gain_delta'),
            'per_gain_delta': c.resolve(dict).get('environment', {}).get('controller', {}).get('pid_adaptation', {}).get('per_gain_delta'),
        }
    ), singleton=True)

    container_instance.register(Environment, lambda c: c.resolve(EnvironmentFactory).create_environment(
        env_type=c.resolve(dict).get('environment', {}).get('type', 'unknown_environment'),
        system=c.resolve(DynamicSystem),
        controller=c.resolve(Controller),
        agent=c.resolve(RLAgent),
        reward_function=c.resolve(RewardFunction),
        stability_calculator=c.resolve(BaseStabilityCalculator), # <<< INYECTAR BaseStabilityCalculator
        config=c.resolve(dict) # Config completa
    ), singleton=True)

    # 4. Servicios de Soporte (singletons)
    container_instance.register(ResultHandler, lambda c: ResultHandler(logger=c.resolve(logging.Logger)), singleton=True)
    container_instance.register(HeatmapGenerator, lambda c: HeatmapGenerator(injected_logger=c.resolve(logging.Logger)), singleton=True)
    container_instance.register(PlotGenerator, lambda c: MatplotlibPlotGenerator(), singleton=True) # Asumiendo que no tiene deps complejas

    # 5. Managers y Colectores (transitorios, creados por petición)
    container_instance.register(MetricsCollector, lambda c: ExtendedMetricsCollector(
        allowed_metrics_to_log=c.resolve(PROCESSED_DATA_DIRECTIVES_TOKEN_STR).get('allowed_json_metrics', [])
        if c.resolve(PROCESSED_DATA_DIRECTIVES_TOKEN_STR).get('json_history_enabled', False) else []
    ), singleton=False) # Transient

    # Usar tokens string para evitar importaciones circulares si SimulationManager/VisualizationManager están en otros módulos
    container_instance.register("simulation_manager.SimulationManager", lambda c: SimulationManager(
        logger=c.resolve(logging.Logger),
        result_handler=c.resolve(ResultHandler),
        container=c # Pasa el propio contenedor
    ), singleton=False) # Transient

    container_instance.register(VisualizationManager, lambda c: VisualizationManager( # Usar la clase directamente como token
        logger_instance=c.resolve(logging.Logger),
        plot_generator=c.resolve(PlotGenerator),
        heatmap_generator=c.resolve(HeatmapGenerator),
        vis_config_data=c.resolve(VIS_CONFIG_TOKEN_STR),
        results_folder_path=c.resolve(OUTPUT_DIR_TOKEN_STR) # Resolver ruta de salida
    ), singleton=False) # Transient
    
    container_instance._container_logger.info("DI Container built and all providers registered.")
    return container_instance