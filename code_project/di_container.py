# di_container.py
import threading
import logging
from typing import Any, Callable, Dict, Optional, Union
import importlib

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
from factories.reward_factory import RewardFactory, NullStabilityCalculator
from factories.environment_factory import EnvironmentFactory

# Componentes Concretos (para registro en factorías y algunos helpers)
from utils.data.result_handler import ResultHandler
from utils.data.heatmap_generator import HeatmapGenerator
from components.plotting.matplotlib_plot_generator import MatplotlibPlotGenerator
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector
from simulation_manager import SimulationManager # Para token de string
from visualization_manager import VisualizationManager # Para token de string

# Componentes específicos que se registran en las factorías
from components.rewards.omni_reward_calculator import OmniRewardCalculator
from components.rewards.lagrangian_reward_calculator import LagrangianRewardCalculator
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.reward_strategies.base_reward_strategy import BaseRewardStrategy
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy
from components.simulators.virtual_simulator import DynamicVirtualSimulator


# Esto es aceptable solo si son pocos y bien conocidos.
VIS_CONFIG_TOKEN_STR = "visualization_config_dict_token"
PROCESSED_DATA_DIRECTIVES_TOKEN_STR = "processed_data_directives_dict_token"
OUTPUT_DIR_TOKEN_STR = "output_dir_path_token"
CONTROLLERS_DICT_TOKEN_STR = "controllers_dict_token"
SYSTEMS_DICT_TOKEN_STR = "systems_dict_token"
AGENTS_DICT_TOKEN_STR = "agents_dict_token"
ENVIRONMENTS_DICT_TOKEN_STR = "environments_dict_token"
REWARD_FUNCTIONS_DICT_TOKEN_STR = "reward_functions_dict_token"
REWARD_STRATEGIES_DICT_TOKEN_STR = "reward_strategies_dict_token"
SIMULATION_MANAGER_TOKEN_STR = "simulation_manager.SimulationManager"
# VISUALIZATION_MANAGER_TOKEN_STR = "visualization_manager_service_token" # Se usa la clase como token

class Container:
    def __init__(self):
        self._providers: Dict[Any, tuple[Callable[['Container'], Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._resolving_tracker: threading.local = threading.local()
        self._container_logger = logging.getLogger(f"{__name__}.DIContainer[{id(self)}]")
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
        is_optional = False
        actual_token = token
        if getattr(token, "__origin__", None) is Union:
            args = getattr(token, "__args__", ())
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
        finally:
            if hasattr(self._resolving_tracker, 'current_set'):
                if actual_token in self._resolving_tracker.current_set: 
                    self._resolving_tracker.current_set.remove(actual_token)
                if not self._resolving_tracker.current_set: 
                    delattr(self._resolving_tracker, 'current_set')

        if is_singleton:
            with self._lock: 
                self._singletons[actual_token] = instance
        # self._container_logger.debug(f"Resolved instance for token {actual_token}: {type(instance).__name__}")
        return instance

# --- Helpers de Creación (para simplificar lambdas en build_container) ---
def _import_class(module_path: str, class_name: str) -> type:
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def _import_from_config(component_config: Dict[str, Any], component_name: str) -> tuple[str, type]:
    if not isinstance(component_config, dict):
        raise ValueError(f"DI Builder: Config for '{component_name}' must be a dict.")
    module_name = component_config.get('module_name')
    module_path = component_config.get('module_path')
    class_name = component_config.get('class_name')
    if not module_name or not module_path or not class_name:
        raise ValueError(f"DI Builder: Config for '{component_name}' is missing 'module_name', 'module_path', or 'class_name'.")
    component_class = _import_class(module_path, class_name)
    return module_name, component_class

def _create_stability_calculator_helper(c: Container) -> BaseStabilityCalculator:               # REQUIERE QUE SEA AGNÓSTICO
    """Helper para crear BaseStabilityCalculator, inyectando la config necesaria."""
    # Se resuelve la config y se decide qué tipo de StabilityCalculator crear.
    config = c.resolve(dict)
    stability_cfg  = config.get('metrics_setup', {}).get('stability_measure', {})
    if not isinstance(stability_cfg, dict) or not stability_cfg:
        # logger.info("[DIHelper:_create_stability_calc] Stability measure config absent/invalid. Creating NullStabilityCalculator.")
        return NullStabilityCalculator() # Crear NullStabilityCalculator con una config de agente dummy, ya que no se usará.

    calc_type = stability_cfg.get('type')
    # Prioridad 1: Weighted Exponential (antes simple_exponential)
    if stability_cfg.get('weighted_exponential_params', {}).get('enabled', False):
        return SimpleExponentialStabilityCalculator(config=config)
    # Prioridad 2: IRA Z-Score
    if stability_cfg.get('ira_zscore_metric_params', {}).get('enabled', False):
        return IRAStabilityCalculator(config=config)
    return NullStabilityCalculator(config)

def _create_reward_function_helper(c: Container) -> RewardFunction:
    """Helper para crear RewardFunction usando RewardFactory."""
    config = c.resolve(dict)
    reward_factory = c.resolve(RewardFactory)
    stability_calculator = c.resolve(BaseStabilityCalculator)
    return reward_factory.create_reward_function(config, stability_calculator)

def _create_reward_strategy_helper(c: Container) -> RewardStrategy:
    """Helper para crear RewardStrategy usando RewardFactory."""
    config = c.resolve(dict)
    reward_factory = c.resolve(RewardFactory)
    agent_instance_for_vars = c.resolve(RLAgent)
    
    reward_strategy_cfg = config.get('environment', {}).get('reward_setup', {}).get('reward_strategy')
    if not isinstance(reward_strategy_cfg, dict):
        raise ValueError("DIHelper: Config 'environment.reward_setup.reward_strategy' missing or not a dict for RewardStrategy.")

    active_strategy_type = 'base' # Default
    strategy_params = {}
    
    for key, cfg in reward_strategy_cfg.items():
        if isinstance(cfg, dict) and cfg.get('enabled', False):
            active_strategy_type = key
            # Extraer parámetros específicos (ej. shadow_baseline_delta_params)
            strategy_params = cfg.get(f'{key}_params', {})
            break
    
    # Inyectar variables del agente y configuración global de recompensas
    strategy_params['agent_defining_vars'] = agent_instance_for_vars.get_agent_defining_vars()
    # Inyectar 'reward_config' completo para acceso a reward_assign, compose, etc.
    strategy_params['global_reward_config'] = config.get('environment', {}).get('reward_setup', {}).get('reward_config', {})
    return reward_factory.create_reward_strategy(active_strategy_type, strategy_params)

def _create_virtual_simulator_helper(c: Container) -> Optional[VirtualSimulator]:
    """Helper para crear VirtualSimulator si es necesario."""
    config = c.resolve(dict)
    reward_strategy = c.resolve(RewardStrategy)
    if not reward_strategy.needs_virtual_simulation:
        return None
    system_tpl = c.resolve(DynamicSystem)
    controller_tpl = c.resolve(Controller)
    reward_func_tpl = c.resolve(RewardFunction)
    stability_calc_tpl = c.resolve(BaseStabilityCalculator)
    dt_val = config.get('environment', {}).get('environment_single', {}).get('simulation', {}).get('dt_sec')
    return DynamicVirtualSimulator(system_template=system_tpl,
                                   controller_template=controller_tpl,
                                   reward_function_template=reward_func_tpl,
                                   stability_calculator_template=stability_calc_tpl,
                                   dt_sec_value=dt_val)

def _extract_entries(component_config: Dict[str, Any],
                     creator_key: str,
                     single_key: str,
                     multi_single_key: str,
                     multi_equal_key: str,
                     entry_prefix: str,
                     ) -> list[tuple[str, Dict[str, Any]]]:
    creator_mode = component_config.get(creator_key)
    if creator_mode == 'single':
        single_cfg = component_config.get(single_key, {})
        return [(single_key, single_cfg)] if single_cfg else []
    if creator_mode in {'multi_single', 'mix'}:
        multi_cfg = component_config.get(multi_single_key, {})
        return [
            (key, cfg)
            for key, cfg in multi_cfg.items()
            if key.startswith(entry_prefix) and isinstance(cfg, dict)
        ]
    if creator_mode == 'multi_equal':
        equal_cfg = component_config.get(multi_equal_key, {})
        return [(multi_equal_key, equal_cfg)] if equal_cfg else []
    return []

def _extract_controller_entries(controller_config: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
    return _extract_entries(controller_config,
                            creator_key='controller_creator',
                            single_key='controller_single',
                            multi_single_key='controller_multi_single',
                            multi_equal_key='controller_multi_equal',
                            entry_prefix='controller',)

def _extract_agent_entries(agent_config: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
    return _extract_entries(agent_config,
                            creator_key='agent_creator',
                            single_key='agent_single',
                            multi_single_key='agent_multi_single',
                            multi_equal_key='agent_multi_equal',
                            entry_prefix='agent',)

def _extract_system_entries(environment_config: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
    dynamics_cfg = environment_config.get('dynamicsystem', {})
    return _extract_entries(dynamics_cfg,
                            creator_key='dynamicsystem_creator',
                            single_key='dynamicsystem_single',
                            multi_single_key='dynamicsystem_multi_single',
                            multi_equal_key='dynamicsystem_multi_equal',
                            entry_prefix='dynamicsystem',)

def _extract_environment_entries(environment_config: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
    return _extract_entries(environment_config,
                            creator_key='environment_creator',
                            single_key='environment_single',
                            multi_single_key='environment_multi_single',
                            multi_equal_key='environment_multi_equal',
                            entry_prefix='environment',)

# --- Función Principal de Construcción del Contenedor ---
def build_container(main_config: Dict[str, Any],
                    vis_config: Optional[Dict[str, Any]],
                    processed_data_directives: Optional[Dict[str, Any]],
                    output_dir: str # Recibir output_dir directamente
                   ) -> Container:

    container_instance = Container()
    container_logger = container_instance._container_logger # Para logging interno del build

    # 1. Registro Fundamental (singletons)
    container_instance.register(Container, lambda c_self: container_instance, singleton=True)
    container_instance.register(logging.Logger, lambda c_log: logging.getLogger(), singleton=True)
    container_instance.register(dict, lambda c_cfg: main_config, singleton=True) # Config principal
    container_instance.register(VIS_CONFIG_TOKEN_STR, lambda c_vis_cfg: vis_config, singleton=True)
    container_instance.register(PROCESSED_DATA_DIRECTIVES_TOKEN_STR, lambda c_pdd_cfg: processed_data_directives or {}, singleton=True)
    container_instance.register(OUTPUT_DIR_TOKEN_STR, lambda c_out_dir: output_dir, singleton=True)

    # 2. Factorías (singletons)
    agent_factory = AgentFactory()
    container_instance.register(AgentFactory, lambda c: agent_factory, singleton=True)
    environment_factory = EnvironmentFactory()
    container_instance.register(EnvironmentFactory, lambda c: environment_factory, singleton=True)
    controller_factory = ControllerFactory()
    container_instance.register(ControllerFactory, lambda c: controller_factory, singleton=True)
    system_factory = SystemFactory()
    container_instance.register(SystemFactory, lambda c: system_factory, singleton=True)
    reward_factory = RewardFactory()
    container_instance.register(RewardFactory, lambda c: reward_factory, singleton=True)
    
    # Registrar por tipo de componentes
    env_config = main_config.get('environment', {})

    system_entries = _extract_system_entries(env_config)
    if not system_entries:
        raise ValueError("DI Builder: No dynamicsystem entries found in configuration.")
    for _, sys_cfg in system_entries:
        sys_type_name, sys_cls = _import_from_config(sys_cfg, 'System')
        system_factory.register_system_type(sys_type_name, sys_cls)
    primary_system_name, primary_system_cfg = system_entries[0]
    primary_system_type = _import_from_config(primary_system_cfg, 'System')[0]

    agent_entries = _extract_agent_entries(env_config.get('agent', {}))
    if not agent_entries:
        raise ValueError("DI Builder: No agent entries found in configuration.")
    for _, ag_cfg in agent_entries:
        ag_type_name, ag_cls = _import_from_config(ag_cfg, 'Agent')
        agent_factory.register_agent_type(ag_type_name, ag_cls)
    primary_agent_name, primary_agent_cfg = agent_entries[0]
    primary_agent_type = _import_from_config(primary_agent_cfg, 'Agent')[0]

    environment_entries = _extract_environment_entries(env_config)
    if not environment_entries:
        raise ValueError("DI Builder: No environment entries found in configuration.")
    for _, env_cfg in environment_entries:
        env_type_name, env_cls = _import_from_config(env_cfg, 'Environment')
        environment_factory.register_environment_type(env_type_name, env_cls)
    primary_env_name, primary_env_cfg = environment_entries[0]
    primary_env_type = _import_from_config(primary_env_cfg, 'Environment')[0]

    controller_entries = _extract_controller_entries(env_config.get('controller', {}))
    for ctrl_name, ctrl_cfg in controller_entries:
        ctrl_type_name, ctrl_cls = _import_from_config(ctrl_cfg, f"Controller({ctrl_cfg.get('type', ctrl_name)})")
        controller_factory.register_controller_type(ctrl_type_name, ctrl_cls)
    
    reward_factory.register_reward_function_type('omni_reward', OmniRewardCalculator)
    reward_factory.register_reward_function_type('lagrangian_reward', LagrangianRewardCalculator)
    
    reward_factory.register_reward_strategy_type('base', BaseRewardStrategy)
    #reward_factory.register_reward_strategy_type('weighted_sum_features', BaseRewardStrategy)
    reward_factory.register_reward_strategy_type('shadow_baseline_delta', ShadowBaselineRewardStrategy)
    reward_factory.register_reward_strategy_type('echo_virtual_baseline_delta', EchoBaselineRewardStrategy)

    # 3. Componentes Principales (singletons, usando helpers o factorías)
    container_instance.register(BaseStabilityCalculator, _create_stability_calculator_helper, singleton=True)
    container_instance.register(RewardFunction, _create_reward_function_helper, singleton=True)
    container_instance.register(RewardStrategy, _create_reward_strategy_helper, singleton=True)
    container_instance.register(Optional[VirtualSimulator], _create_virtual_simulator_helper, singleton=True)

    container_instance.register(REWARD_FUNCTIONS_DICT_TOKEN_STR,
                                lambda c: {
                                    primary_env_name: c.resolve(RewardFunction)
                                    },
                                singleton=True,)

    container_instance.register(REWARD_STRATEGIES_DICT_TOKEN_STR,
                                lambda c: {
                                    primary_agent_name: c.resolve(RewardStrategy)
                                    },
                                singleton=True,)

    container_instance.register(SYSTEMS_DICT_TOKEN_STR,
                                lambda c: {
                                    name: c.resolve(SystemFactory).create_system(system_type=_import_from_config(cfg, 'System')[0],
                                                                                 system_params=cfg.get('params', {}),)
                                    for name, cfg in system_entries
                                    },
                                singleton=True,)

    container_instance.register(DynamicSystem,
                                lambda c: c.resolve(SystemFactory).create_system(system_type=primary_system_type,
                                                                                 system_params=primary_system_cfg.get('params', {}),),
                                singleton=True,)

    container_instance.register(AGENTS_DICT_TOKEN_STR,
                                lambda c: {
                                    name: c.resolve(AgentFactory).create_agent(agent_type=_import_from_config(cfg, 'Agent')[0],
                                                                               agent_constructor_params={**cfg.get('learning_params', {}),
                                                                                                         **cfg.get('agent_config', {}),
                                                                                                         'reward_strategy': None,
                                                                                                         'main_config': c.resolve(dict),},)
                                    for name, cfg in agent_entries
                                    },
                                singleton=True,)

    container_instance.register(CONTROLLERS_DICT_TOKEN_STR,
                                lambda c: {
                                    f"controller_{ctrl_cfg.get('params', {}).get('name_objective_var')}": 
                                    c.resolve(ControllerFactory).create_controller(controller_type=_import_from_config(ctrl_cfg, 'Controller')[0],
                                                                                   controller_params={**ctrl_cfg.get('params', {}), 
                                                                                                      'dt_sec': env_config.get('simulation', {}).get('dt_sec')},)
                                for _, ctrl_cfg in controller_entries if isinstance(ctrl_cfg, dict)
                                },
                                singleton=True,)

    container_instance.register(RLAgent,
                                lambda c: c.resolve(AgentFactory).create_agent(agent_type=primary_agent_type,
                                                                               agent_constructor_params={**primary_agent_cfg.get('learning_params', {}),
                                                                                                         **primary_agent_cfg.get('agent_config', {}),
                                                                                                         'reward_strategy': None,
                                                                                                         'main_config': c.resolve(dict),},),
                                singleton=True,)

    # ENLACE DE DEPENDENCIAS CIRCULARES (Two-Phase Construction)
    # 1. Se resuelven las instancias que ya fueron construidas (pero no enlazadas).
    agent_instance = container_instance.resolve(RLAgent)
    strategy_instance = container_instance.resolve(RewardStrategy)
    
    # 2. Se realiza el enlace llamando al método setter.
    if hasattr(agent_instance, 'set_reward_strategy'):
        agent_instance.set_reward_strategy(strategy_instance)

    container_instance.register(Environment,
                                lambda c: c.resolve(EnvironmentFactory).create_environment(env_type=primary_env_type,
                                                                                           system=c.resolve(DynamicSystem),
                                                                                           controllers=c.resolve(CONTROLLERS_DICT_TOKEN_STR),
                                                                                           agent=c.resolve(RLAgent),
                                                                                           reward_function=c.resolve(RewardFunction),
                                                                                           stability_calculator=c.resolve(BaseStabilityCalculator),
                                                                                           config=c.resolve(dict),),
                                singleton=True,)

    container_instance.register(ENVIRONMENTS_DICT_TOKEN_STR,
                                lambda c: {
                                    name: c.resolve(EnvironmentFactory).create_environment(env_type=_import_from_config(cfg, 'Environment')[0],
                                                                                           system=c.resolve(SYSTEMS_DICT_TOKEN_STR).get(primary_system_name),
                                                                                           controllers=c.resolve(CONTROLLERS_DICT_TOKEN_STR),
                                                                                           agent=c.resolve(AGENTS_DICT_TOKEN_STR).get(primary_agent_name),
                                                                                           reward_function=c.resolve(RewardFunction),
                                                                                           stability_calculator=c.resolve(BaseStabilityCalculator),
                                                                                           config=c.resolve(dict),)
                                    for name, cfg in environment_entries
                                    },
                                singleton=True,)

    # 3. Servicios de Soporte (singletons)
    container_instance.register(ResultHandler, lambda c: ResultHandler(logger=c.resolve(logging.Logger)), singleton=True)
    container_instance.register(HeatmapGenerator, lambda c: HeatmapGenerator(injected_logger=c.resolve(logging.Logger)), singleton=True)
    container_instance.register(PlotGenerator, lambda c: MatplotlibPlotGenerator(), singleton=True)

    # 4. Managers y Colectores (transitorios)
    container_instance.register(MetricsCollector, 
                                lambda c: ExtendedMetricsCollector(data_save_config=c.resolve(PROCESSED_DATA_DIRECTIVES_TOKEN_STR)), 
                                singleton=False) # Transient

    # Usar tokens string para evitar importaciones circulares si SimulationManager/VisualizationManager están en otros módulos
    container_instance.register(SIMULATION_MANAGER_TOKEN_STR, 
                                lambda c: SimulationManager(logger=c.resolve(logging.Logger),
                                                            result_handler=c.resolve(ResultHandler),
                                                            container=c), 
                                singleton=False) # Transient

    container_instance.register(VisualizationManager, 
                                lambda c: VisualizationManager(logger_instance=c.resolve(logging.Logger),
                                                               plot_generator=c.resolve(PlotGenerator),
                                                               heatmap_generator=c.resolve(HeatmapGenerator),
                                                               is_config_data=c.resolve(VIS_CONFIG_TOKEN_STR),
                                                               results_folder_path=c.resolve(OUTPUT_DIR_TOKEN_STR)), singleton=False) # Transient
    
    container_logger.info("DI Container built and all providers registered.")
    return container_instance