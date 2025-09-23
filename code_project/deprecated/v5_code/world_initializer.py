import logging
from typing import Dict, Any, Optional

# --- Factories ---
from factories.environment_factory import EnvironmentFactory
from factories.reward_factory import RewardFactory

# --- Components & Interfaces ---
from components.analysis.extended_metrics_collector import ExtendedMetricsCollector
from interfaces.reward_strategy import RewardStrategy

# --- Reward Strategies ---
from components.reward_strategies.global_reward_strategy import GlobalRewardStrategy
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy

# --- Optional: Virtual Simulator ---
try:
    from components.simulators.pendulum_virtual_simulator import PendulumVirtualSimulator
except ImportError:
    PendulumVirtualSimulator = None
    logging.info("PendulumVirtualSimulator not found, Echo Baseline mode will not work.")

def initialize_simulation_components(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Initializes all necessary components for the simulation based on the configuration.
    Orchestrates the creation using dedicated factories and handles dependency injection.

    Args:
        config: The main configuration dictionary (will be modified temporarily).

    Returns:
        A dictionary containing the initialized components
        ('env', 'metrics_collector', 'virtual_simulator', 'agent', 'controller',
         'reward_function', 'stability_calculator', 'reward_strategy'),
        or None if initialization fails critically.
    """
    logging.info("--- Initializing Simulation Components ---")
    components = {}

    # --- [1] Create Metrics Collector ---
    try:
        metrics_collector = ExtendedMetricsCollector()
        components['metrics_collector'] = metrics_collector
        logging.info("ExtendedMetricsCollector created.")
    except Exception as e:
        logging.error(f"CRITICAL: Error creating Metrics Collector: {e}. Exiting.", exc_info=True)
        return None

    # --- [2] Create Stability Calculator and Reward Function ---
    env_cfg = config.get('environment', {})
    reward_setup_cfg = env_cfg.get('reward_setup', {})
    stability_calculator = None # Initialize as None
    reward_function = None # Initialize as None
    try:
        # Create stability calculator first using the factory
        stability_calculator = RewardFactory.create_stability_calculator(reward_setup_cfg)
        components['stability_calculator'] = stability_calculator # Store even if None

        # Create reward function, passing the calculator instance
        reward_function = RewardFactory.create_reward_function(reward_setup_cfg, stability_calculator)
        components['reward_function'] = reward_function # Store the function instance
        logging.info(f"Reward Function ({type(reward_function).__name__}) and Stability Calculator ({type(stability_calculator).__name__ if stability_calculator else 'None'}) created.")

    except Exception as e:
        # Catch errors from RewardFactory (ValueError, RuntimeError)
        logging.error(f"CRITICAL: Error creating reward components via RewardFactory: {e}. Exiting.", exc_info=True)
        return None

    # --- [3] Create Reward Strategy ---
    reward_strategy: Optional[RewardStrategy] = None # Initialize as None
    try:
        # --- Leer learning_strategy y strategy_params ---
        reward_mode = reward_setup_cfg.get('learning_strategy', 'global') # Leer de la nueva clave
        strategy_params = reward_setup_cfg.get('strategy_params', {})

        if reward_mode == 'global':
            reward_strategy = GlobalRewardStrategy()
        elif reward_mode == 'shadow_baseline':
            # Obtener params específicos de shadow
            shadow_params = strategy_params.get('shadow_baseline', {})
            if not shadow_params:
                 logging.warning("Shadow baseline selected, but no 'shadow_baseline' params found under 'strategy_params'. Using defaults.")
            reward_strategy = ShadowBaselineRewardStrategy(beta=shadow_params.get('beta', 0.1))
        elif reward_mode == 'echo_baseline':
            # Obtener params específicos de echo (si los hubiera en el futuro)
            echo_params = strategy_params.get('echo_baseline', {})
            reward_strategy = EchoBaselineRewardStrategy() # Pasar echo_params si __init__ los necesita
        else:
            raise ValueError(f"Unsupported learning_strategy specified in config: {reward_mode}")

        components['reward_strategy'] = reward_strategy
        logging.info(f"Reward Strategy ({type(reward_strategy).__name__}) created for mode '{reward_mode}'.")
    except ValueError as e:
         logging.error(f"CRITICAL: Invalid config for reward strategy: {e}. Exiting.")
         return None
    except Exception as e:
        logging.error(f"CRITICAL: Error creating reward strategy: {e}. Exiting.", exc_info=True)
        return None

    # --- [4] Create Environment (using Factory) ---
    # Temporarily modify config to inject necessary instances for agent creation
    agent_params_config_path = env_cfg.get('agent', {}).get('params', {}) # Acceso a agent.params
    original_shadow_params_in_agent  = agent_params_config_path.get('shadow_baseline_params') # Store original value if exists

    try:
        # Inject the created reward_strategy instance into the agent parameters within the config dict.
        # This allows AgentFactory (called by EnvironmentFactory) to access it.
        agent_params_config_path['reward_strategy_instance'] = reward_strategy

        # --- Inyectar shadow_baseline_params DESDE reward_setup si es necesario ---
        # El Agente puede necesitar baseline_init_value durante su __init__
        if reward_mode == 'shadow_baseline':
             shadow_params_from_strategy = strategy_params.get('shadow_baseline', {})
             # Inyectar el dict completo para que el agente pueda tomar lo que necesita (init_value)
             agent_params_config_path['shadow_baseline_params'] = shadow_params_from_strategy
             # Esto sobreescribe cualquier valor que estuviera en agent.params (lo cual es la intención)

        # Crear entorno (EnvironmentFactory NO necesita cambios aquí, usa el config modificado)
        env = EnvironmentFactory.create_environment(
            config=config, # Pasar config completo (contiene agent.params modificados temporalmente)
            reward_function_instance=reward_function
        )
        components['env'] = env
        logging.info("Environment (incl Agent, Controller, System) created via Factory.")

        components['agent'] = getattr(env, 'agent', None)
        components['controller'] = getattr(env, 'controller', None)
        if not components['agent'] or not components['controller']:
            raise RuntimeError("Env factory didn't set up agent/controller.")

    except (ValueError, RuntimeError, KeyError, AttributeError) as e:
        logging.error(f"CRITICAL: Error creating environment via Factory: {e}. Exiting.", exc_info=True)
        # Limpiar config antes de retornar (importante)
        agent_params_config_path.pop('reward_strategy_instance', None)
        if reward_mode == 'shadow_baseline':
             agent_params_config_path.pop('shadow_baseline_params', None) # Siempre quitar el inyectado
             if original_shadow_params_in_agent is not None: # Restaurar si existía ANTES
                  agent_params_config_path['shadow_baseline_params'] = original_shadow_params_in_agent
        return None
    finally:
        # --- Limpieza final de config ---
        # Quitar instancia inyectada
        agent_params_config_path.pop('reward_strategy_instance', None)
        # Quitar/Restaurar shadow_baseline_params
        if reward_mode == 'shadow_baseline':
             agent_params_config_path.pop('shadow_baseline_params', None)
             if original_shadow_params_in_agent is not None:
                  agent_params_config_path['shadow_baseline_params'] = original_shadow_params_in_agent


    # --- [5] Initialize Virtual Simulator (if needed) ---
    virtual_simulator = None
    if reward_mode == 'echo_baseline':
        if PendulumVirtualSimulator is None:
            logging.error("CRITICAL: Echo Baseline mode selected, but PendulumVirtualSimulator could not be imported. Exiting.")
            return None
        try:
            env_system = getattr(env, 'system', None)
            env_controller = getattr(env, 'controller', None)
            dt = env_cfg.get('dt', None)

            if not all([env_system, env_controller, reward_function, dt]):
                 # Check components needed by the simulator's constructor
                raise ValueError("Missing required components (system, controller, reward_function, dt) for VirtualSimulator.")

            virtual_simulator = PendulumVirtualSimulator(
                system=env_system,
                controller=env_controller, # Pass the real controller instance
                reward_function=reward_function, # Pass the real reward function instance
                dt=dt
            )
            logging.info("PendulumVirtualSimulator created for Echo Baseline mode.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to create VirtualSimulator: {e}. Exiting.", exc_info=True)
            return None

    components['virtual_simulator'] = virtual_simulator # Store None if not created

    logging.info("--- Simulation Components Initialized Successfully ---")
    return components