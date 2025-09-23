# di_container.py
import threading
import logging
from typing import Any, Callable, Dict, Type, Optional, Union # Quitamos cast, List, Set, TYPE_CHECKING si no se usan directamente
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
from components.rewards.instantaneous_reward_calculator import InstantaneousRewardCalculator
from components.analysis.ira_stability_calculator import IRAStabilityCalculator
from components.analysis.simple_exponential_stability_calculator import SimpleExponentialStabilityCalculator
from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy
from components.simulators.virtual_simulator import DynamicVirtualSimulator


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
    """Helper para crear BaseStabilityCalculator, inyectando la config necesaria."""
    # RewardFactory ya no crea StabilityCalculator. StabilityCalculator se crea directamente.
    # Se resuelve la config y se decide qué tipo de StabilityCalculator crear.
    config = c.resolve(dict)

    stability_measure_cfg = config.get('environment', {}).get('reward_setup', {}).get('calculation', {}).get('stability_measure')
    agent_state_cfg = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})

    if not isinstance(stability_measure_cfg, dict) or not stability_measure_cfg:
        # logger.info("[DIHelper:_create_stability_calc] Stability measure config absent/invalid. Creating NullStabilityCalculator.")
        return NullStabilityCalculator() # Crear NullStabilityCalculator con una config de agente dummy, ya que no la usará.

    calc_type = stability_measure_cfg.get('type')
    # Registrar tipos concretos directamente en DI o usar una mini-factoría aquí si crece mucho
    if calc_type == 'ira_zscore_metric':
        params = stability_measure_cfg.get('ira_zscore_metric_params', {})
        return IRAStabilityCalculator(config=config)
    elif calc_type == 'exp_decay_metric':
        params = stability_measure_cfg.get('exp_decay_metric_params', {})
        return SimpleExponentialStabilityCalculator(config=config)
    else:
        # logger.warning(f"[DIHelper:_create_stability_calc] Unknown stability_measure type '{calc_type}'. Using NullStabilityCalculator.")
        return NullStabilityCalculator(config)

def _create_reward_function_helper(c: Container) -> RewardFunction:
    """Helper para crear RewardFunction usando RewardFactory."""
    logger_instance = c.resolve(logging.Logger)
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

    strategy_type = reward_strategy_cfg.get('type')
    strategy_params = reward_strategy_cfg.get('strategy_params', {}).get(strategy_type, {})
    strategy_params['agent_defining_vars'] = agent_instance_for_vars.get_agent_defining_vars()

    return reward_factory.create_reward_strategy(strategy_type, strategy_params)

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
    return DynamicVirtualSimulator(
        system_template=system_tpl,
        controller_template=controller_tpl,
        reward_function_template=reward_func_tpl,
        stability_calculator_template=stability_calc_tpl, # <<< PASARLO
        dt_sec_value=dt_val
    )

def _load_and_register_component_from_config(factory_instance: Any, register_method_name: str, component_config: Dict, component_log_name: str):
        container_instance = Container()
        container_logger = container_instance._container_logger # Para logging interno del build
        type_key = component_config.get('type')
        module_path = component_config.get('module_path')
        class_name = component_config.get('class_name')

        if not all([type_key, module_path, class_name]):
            raise ValueError(f"DI Builder: Config for '{component_log_name}' is missing 'type', 'module_path', or 'class_name'.")

        try:
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            register_method = getattr(factory_instance, register_method_name)
            register_method(type_key, component_class)
            container_logger.info(f"Dynamically loaded and registered {component_log_name} '{type_key}' from {module_path}.{class_name}")
        except ImportError:
            container_logger.critical(f"DI Builder: Failed to import module '{module_path}' for {component_log_name}.")
            raise
        except AttributeError:
            container_logger.critical(f"DI Builder: Failed to find class '{class_name}' in module '{module_path}' for {component_log_name}.")
            raise

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
    container_instance.register(logging.Logger, lambda c_log: logging.getLogger(), singleton=True) # Logger raíz o específico
    container_instance.register(dict, lambda c_cfg: main_config, singleton=True) # Config principal
    container_instance.register(VIS_CONFIG_TOKEN_STR, lambda c_vis_cfg: vis_config, singleton=True)
    container_instance.register(PROCESSED_DATA_DIRECTIVES_TOKEN_STR, lambda c_pdd_cfg: processed_data_directives if processed_data_directives else {}, singleton=True)
    container_instance.register(OUTPUT_DIR_TOKEN_STR, lambda c_out_dir: output_dir, singleton=True)

    # 2. Factorías (singletons)
    agent_f = AgentFactory()
    container_instance.register(AgentFactory, lambda c: agent_f, singleton=True)
    env_f = EnvironmentFactory()
    container_instance.register(EnvironmentFactory, lambda c: env_f, singleton=True)
    ctrl_f = ControllerFactory()
    container_instance.register(ControllerFactory, lambda c: ctrl_f, singleton=True)
    sys_f = SystemFactory()
    container_instance.register(SystemFactory, lambda c: sys_f, singleton=True)
    
    env_config_section = main_config.get('environment', {})
    _load_and_register_component_from_config(sys_f, 'register_system_type', env_config_section.get('system', {}), 'System')
    _load_and_register_component_from_config(agent_f, 'register_agent_type', env_config_section.get('agent', {}), 'Agent')
    _load_and_register_component_from_config(env_f, 'register_environment_type', env_config_section, 'Environment')

    reward_f = RewardFactory()
    # Registros para RewardFactory (no dinámicos por ahora)
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

    # NUEVA LÓGICA DE REGISTRO PARA MÚLTIPLES CONTROLADORES CON CARGA DINÁMICA
    controllers_dict_token = "controllers_dict_token"
    container_instance.register(controllers_dict_token, lambda c: {
        # La clave del diccionario será "controller_" + name_objective_var
        f"controller_{ctrl_config.get('params', {}).get('name_objective_var')}": 
        (
            # Paso 1: Cargar y registrar dinámicamente la clase del controlador
            _load_and_register_component_from_config(
                c.resolve(ControllerFactory),
                'register_controller_type',
                ctrl_config,
                f"Controller({ctrl_config.get('type')})"
            ),
            # Paso 2: Crear la instancia, pasando TODOS sus parámetros específicos
            c.resolve(ControllerFactory).create_controller(
                controller_type=ctrl_config.get('type'),
                controller_params={
                    # Desempaquetar los parámetros de la sección 'params'
                    **ctrl_config.get('params', {}),
                    # Añadir el 'dt_sec' global
                    'dt_sec': c.resolve(dict).get('environment', {}).get('simulation', {}).get('dt_sec')
                }
            )
        )[1] # Devolvemos solo la instancia creada en el paso 2
        # Iterar sobre las secciones de controlador definidas en el YAML
        for key, ctrl_config in c.resolve(dict).get('environment', {}).get('controller', {}).items()
        if key.startswith('controller')
    }, singleton=True)

    container_instance.register(RLAgent, lambda c: c.resolve(AgentFactory).create_agent(
        agent_type=c.resolve(dict).get('environment', {}).get('agent', {}).get('type', 'unknown_agent'),
        agent_constructor_params={
            **c.resolve(dict).get('environment', {}).get('agent', {}).get('params', {}),
            'reward_strategy': None,
            # Se pasa la configuración completa para que el agente pueda extraer los parámetros
            # de los controladores que le conciernen.
            'main_config': c.resolve(dict) 
        }
    ), singleton=True)

    # ENLACE DE DEPENDENCIAS CIRCULARES (Two-Phase Construction)
    # 1. Se resuelven las instancias que ya fueron construidas (pero no enlazadas).
    agent_instance = container_instance.resolve(RLAgent)
    strategy_instance = container_instance.resolve(RewardStrategy)
    
    # 2. Se realiza el enlace llamando al método setter.
    if hasattr(agent_instance, 'set_reward_strategy'):
        agent_instance.set_reward_strategy(strategy_instance)

    container_instance.register(Environment, lambda c: c.resolve(EnvironmentFactory).create_environment(
        env_type=c.resolve(dict).get('environment', {}).get('type', 'unknown_environment'),
        system=c.resolve(DynamicSystem),
        controllers=c.resolve("controllers_dict_token"),
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
        data_save_config=c.resolve(PROCESSED_DATA_DIRECTIVES_TOKEN_STR)
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