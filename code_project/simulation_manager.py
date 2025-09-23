# simulation_manager.py
import logging
import time
import numpy as np
import pandas as pd
import os
# 1.1: Eliminar import 'copy', ya no se usa aquí
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING

# Interfaces (Type Hinting - Sin cambios)
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.controller import Controller
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.reward_strategy import RewardStrategy

# Importar estrategia concreta solo para check de tipo (si es necesario)
from components.reward_strategies.echo_baseline_reward_strategy import EchoBaselineRewardStrategy # Mantener para check

# Servicios Auxiliares
from utils.data.result_handler import ResultHandler
from utils.data.data_processing import summarize_episode

# Romper ciclo de importación para type hints
if TYPE_CHECKING:
    from di_container import Container # Importar Container para type hint

# Obtener logger (se inyecta en __init__)
# logger = logging.getLogger(__name__) # No obtener aquí

class SimulationManager:
    """
    Orquesta la ejecución de la simulación completa, episodio por episodio.
    Utiliza componentes resueltos vía DI y gestiona el flujo principal.
    Interactúa con los componentes a través de sus interfaces.
    """

    def __init__(self,
                 logger: logging.Logger, # 1.2: Inyectar logger
                 result_handler: ResultHandler,
                 container: 'Container' # 1.3: Inyectar el contenedor DI
                 ):
        """ Inicializa el SimulationManager con dependencias clave. """
        self.logger = logger # Usar logger inyectado
        self.result_handler = result_handler
        self.container = container # Guardar referencia al contenedor
        self.logger.info("SimulationManager instance created.")
        # 1.4: Validar dependencias esenciales inyectadas
        if container is None:
             msg = "SimulationManager requiere una instancia Container válida."
             self.logger.critical(msg); raise ValueError(msg) # Fail-Fast
        if logger is None or result_handler is None:
             msg = "SimulationManager requiere instancias Logger y ResultHandler válidas."
             self.logger.critical(msg); raise ValueError(msg) # Fail-Fast
        #self.logger.debug("SimulationManager -> __init__ -> OK") # [DEBUG ADDED]

    # --- Métodos Privados de Orquestación ---

    def _resolve_dependencies(self) -> Tuple[Environment, RLAgent, Controller, MetricsCollector, RewardStrategy, Optional[VirtualSimulator], Dict[str, Any], str]:
        """ Resuelve las dependencias principales desde el contenedor DI. """
        #self.logger.debug("Resolviendo dependencias para la simulación...")
        try:
            # 1.5: Resolver componentes usando el contenedor inyectado
            environment = self.container.resolve(Environment)
            agent = self.container.resolve(RLAgent)
            controller = self.container.resolve(Controller)
            metrics_collector = self.container.resolve(MetricsCollector) # Resuelve como Transient
            reward_strategy = self.container.resolve(RewardStrategy)
            # Resolver Optional[VirtualSimulator]
            virtual_simulator = self.container.resolve(Optional[VirtualSimulator])
            config = self.container.resolve(dict) # Config general
            results_folder = self.container.resolve(str) # Carpeta de resultados (registrada en main)
            #self.logger.debug("Dependencias resueltas.")

            # [DEBUG ADDED] - Log tipos resueltos
            #self.logger.debug(f"SimulationManager -> _resolve_dependencies -> Resolved: Env={type(environment).__name__}, Agt={type(agent).__name__}, Ctrl={type(controller).__name__}, RwdStr={type(reward_strategy).__name__}, VirtSim={type(virtual_simulator).__name__ if virtual_simulator else 'None'}")

            # 1.6: Validar dependencias requeridas (Fail-Fast)
            required_components = {
                "Environment": environment, "RLAgent": agent, "Controller": controller,
                "MetricsCollector": metrics_collector, "RewardStrategy": reward_strategy,
                "dict (config)": config, "str (results_folder)": results_folder
            }
            missing = [name for name, var in required_components.items() if var is None]
            if missing:
                raise ValueError(f"Fallo al resolver dependencias DI clave: {missing}")

            # Validar tipo de MetricsCollector (debe ser transient)
            # mc1 = self.container.resolve(MetricsCollector)
            # mc2 = self.container.resolve(MetricsCollector)
            # if mc1 is mc2:
            #     self.logger.warning("MetricsCollector parece estar registrado como Singleton, debería ser Transient.")
            # del mc1, mc2

            #self.logger.debug("SimulationManager -> _resolve_dependencies -> End OK") # [DEBUG ADDED]
            return environment, agent, controller, metrics_collector, reward_strategy, virtual_simulator, config, results_folder

        except (ValueError, RecursionError) as e: # Capturar errores de DI
            self.logger.critical(f"Error fatal resolviendo dependencias DI: {e}", exc_info=True)
            raise # Relanzar para detener la simulación (Fail-Fast)
        except Exception as e:
            self.logger.critical(f"Error inesperado resolviendo dependencias DI: {e}", exc_info=True)
            raise # Relanzar (Fail-Fast)

    def _initialize_episode(self,
                            episode_id: int,
                            environment: Environment,
                            agent: RLAgent,
                            controller: Controller,
                            metrics_collector: MetricsCollector, # Recibir instancia transient
                            config: Dict
                           ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inicializa el entorno, colector, agente y estado para un nuevo episodio."""
        self.logger.info(f"--- [ Episodio {episode_id}/{config.get('environment', {}).get('max_episodes', 1)-1} ] ---")
        try:
            initial_state_vector_cfg = config.get('initial_conditions', {}).get('x0')
            if initial_state_vector_cfg is None:
                raise KeyError("'initial_conditions: x0' no encontrado en la configuración.")

            # 1.7: Resetear componentes usando interfaces
            current_state_vector = environment.reset(initial_state_vector_cfg) # Env resetea agente/controller
            # MetricsCollector se resetea aquí porque es transient por episodio
            metrics_collector.reset(episode_id=episode_id)

            #self.logger.debug(f"SimulationManager -> _initialize_episode -> State after reset = {np.round(current_state_vector, 4)}") # [DEBUG ADDED]

            # Log estado inicial (t=0)
            self._log_initial_metrics(metrics_collector, current_state_vector, controller, agent, config)

            # Preparar estado inicial para el agente
            state_config_for_agent = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
            agent_state_dict = agent.build_agent_state(current_state_vector, controller, state_config_for_agent)

            # Seleccionar acción inicial A'
            initial_actions_prime = agent.select_action(agent_state_dict)
            
            #self.logger.debug(f"SimulationManager -> _initialize_episode -> Initial Action A' = {initial_actions_prime}") # [DEBUG ADDED]

            # Aplicar acción A' inicial al controlador
            self._apply_actions_to_controller(controller, initial_actions_prime, config)

            # Preparar datos para el primer intervalo de decisión
            initial_interval_data = {
                'start_state_dict': agent_state_dict, # S
                'start_raw_state_vector': np.copy(current_state_vector), # Vector crudo de S
                'actions_dict': initial_actions_prime.copy(), # A' (a aplicar durante el intervalo)
                'reward_sum': 0.0, # Acumulador R_real (para log/debug, no para R_learn)
                'w_stab_sum': 0.0, # Acumulador w_stab (para log/debug y Shadow Baseline)
                'steps_in_interval': 0, # Contador pasos dt
                'end_state_dict': None, # S' (se llena al final)
                'done': False, # Flag de terminación
                'reward_dict_echo': None, # R_diff (se llena si es Echo)
                'decision_count': 0 # Contador de decisiones en el episodio
            }
            #self.logger.debug(f"SimulationManager -> _initialize_episode -> End OK") # [DEBUG ADDED]
            return current_state_vector, initial_interval_data

        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
            self.logger.error(f"Error inicializando episodio {episode_id}: {e}", exc_info=True)
            raise # Relanzar para indicar fallo en inicialización (Fail-Fast)

    def _log_initial_metrics(self, metrics_collector: MetricsCollector, state: np.ndarray, controller: Controller, agent: RLAgent, config: Dict):
        """Registra las métricas iniciales (t=0) del episodio."""
        # 1.8: Simplificar logging, asumir que las interfaces proveen la info necesaria
        #      Los getters específicos (`epsilon`, `learning_rate`, `get_params`) vienen de las interfaces.
        #      Usar getattr para atributos opcionales/específicos de implementación.
        
        # (Loggeo inicial menos crítico para DEBUG selectivo, mantenerlo opcional)
        #self.logger.debug("SimulationManager -> _log_initial_metrics -> Start")
        
        metrics_collector.log('time', 0.0)
        try:
            metrics_collector.log('cart_position', state[0]); metrics_collector.log('cart_velocity', state[1])
            metrics_collector.log('pendulum_angle', state[2]); metrics_collector.log('pendulum_velocity', state[3])

            # Métricas del controlador
            ctrl_params = controller.get_params()
            metrics_collector.log('kp', ctrl_params.get('kp', np.nan))
            metrics_collector.log('ki', ctrl_params.get('ki', np.nan))
            metrics_collector.log('kd', ctrl_params.get('kd', np.nan))
            setpoint = getattr(controller, 'setpoint', 0.0) # Atributo específico PID
            metrics_collector.log('error', state[2] - setpoint)
            # Intentar loguear estado interno PID (opcional)
            metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan))
            metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))

            # Métricas del agente
            metrics_collector.log('epsilon', agent.epsilon)
            metrics_collector.log('learning_rate', agent.learning_rate)

            # Configuración de pasos de ganancia
            pid_adapt_cfg = config.get('pid_adaptation', {})
            gain_step_config = pid_adapt_cfg.get('gain_step', 5.0)
            variable_step = pid_adapt_cfg.get('variable_step', False)
            if variable_step and isinstance(gain_step_config, dict):
                 metrics_collector.log('gain_step_kp', float(gain_step_config.get('kp', np.nan)))
                 metrics_collector.log('gain_step_ki', float(gain_step_config.get('ki', np.nan)))
                 metrics_collector.log('gain_step_kd', float(gain_step_config.get('kd', np.nan)))
            elif isinstance(gain_step_config, (int,float)):
                 metrics_collector.log('gain_step', float(gain_step_config))
            else: metrics_collector.log('gain_step', np.nan) # Log NaN si no es numérico

            # Métricas que no existen al inicio
            metrics_collector.log('reward', 0.0); metrics_collector.log('cumulative_reward', 0.0)
            metrics_collector.log('force', 0.0); metrics_collector.log('stability_score', 1.0) # Asumir estable al inicio
            nan_metrics = ['action_kp', 'action_ki', 'action_kd', 'learn_select_duration_ms',
                           'td_error_kp', 'td_error_ki', 'td_error_kd',
                           'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
                           'id_agent_decision', 'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
                           'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
                           'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd',
                           'adaptive_mu_angle', 'adaptive_sigma_angle', # etc. stats adaptativas
                           ]
            adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
            for var in adaptive_vars: nan_metrics.extend([f'adaptive_mu_{var}', f'adaptive_sigma_{var}'])

            for m in nan_metrics: metrics_collector.log(m, np.nan)

            # Loguear Q/Visits/Baseline/Adaptive iniciales (como NaNs o defaults)
            # La implementación de MetricsCollector se encarga de llamar a los métodos del agente
            if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, {}) # type: ignore[operator]
            if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, {}) # type: ignore[operator]
            if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, {}) # type: ignore[operator]
            if hasattr(metrics_collector, 'log_adaptive_stats'): metrics_collector.log_adaptive_stats({}) # type: ignore[operator]

        except IndexError: self.logger.warning("Estado inicial con longitud inesperada al loguear métricas.")
        except Exception as e: self.logger.error(f"Error logueando métricas iniciales: {e}", exc_info=True)
        #self.logger.debug("SimulationManager -> _log_initial_metrics -> End")

    def _apply_actions_to_controller(self, controller: Controller, actions_dict: Dict[str, int], config: Dict):
        """Aplica las acciones seleccionadas para actualizar las ganancias del controlador."""
        
        #self.logger.debug(f"SimulationManager -> _apply_actions_to_controller -> Start: Actions={actions_dict}") # [DEBUG ADDED]
        
        # 1.9: Usar interfaces del controlador y extraer config de forma segura
        try:
            current_gains = controller.get_params()
            kp, ki, kd = current_gains.get('kp', 0.0), current_gains.get('ki', 0.0), current_gains.get('kd', 0.0)

            #self.logger.debug(f"SimulationManager -> _apply_actions_to_controller -> Current Gains = {current_gains}") # [DEBUG ADDED]

            # Obtener pasos de ganancia
            pid_adapt_cfg = config.get('pid_adaptation', {})
            gain_step_config = pid_adapt_cfg.get('gain_step', 1.0) # Default a 1.0 si no existe
            variable_step = pid_adapt_cfg.get('variable_step', False)
            step_kp, step_ki, step_kd = 0.0, 0.0, 0.0

            if variable_step and isinstance(gain_step_config, dict):
                step_kp = float(gain_step_config.get('kp', 0.0)) # Default a 0 si falta
                step_ki = float(gain_step_config.get('ki', 0.0))
                step_kd = float(gain_step_config.get('kd', 0.0))
            elif isinstance(gain_step_config, (int, float)):
                step_kp = step_ki = step_kd = float(gain_step_config)
            else:
                 self.logger.warning(f"Config 'gain_step' inválida ({gain_step_config}). Usando pasos 0.0.")
            
            #self.logger.debug(f"SimulationManager -> _apply_actions_to_controller -> Steps = (Kp:{step_kp:.3f}, Ki:{step_ki:.3f}, Kd:{step_kd:.3f})") # [DEBUG ADDED]

            # Calcular nuevas ganancias basadas en acción (0: Dec, 1: Keep, 2: Inc)
            new_kp = kp + (actions_dict.get('kp', 1) - 1) * step_kp
            new_ki = ki + (actions_dict.get('ki', 1) - 1) * step_ki
            new_kd = kd + (actions_dict.get('kd', 1) - 1) * step_kd

            # Aplicar clipping si está configurado
            gain_limits_cfg = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
            kp_cfg = gain_limits_cfg.get('kp', {}); ki_cfg = gain_limits_cfg.get('ki', {}); kd_cfg = gain_limits_cfg.get('kd', {})
            # Usar np.inf como default si min/max no están definidos
            kp_min, kp_max = kp_cfg.get('min', -np.inf), kp_cfg.get('max', np.inf)
            ki_min, ki_max = ki_cfg.get('min', -np.inf), ki_cfg.get('max', np.inf)
            kd_min, kd_max = kd_cfg.get('min', -np.inf), kd_cfg.get('max', np.inf)

            new_kp = np.clip(new_kp, kp_min, kp_max)
            new_ki = np.clip(new_ki, ki_min, ki_max)
            new_kd = np.clip(new_kd, kd_min, kd_max)

            #self.logger.debug(f"SimulationManager -> _apply_actions_to_controller -> Clipped Gains = (Kp:{new_kp:.3f}, Ki:{new_ki:.3f}, Kd:{new_kd:.3f})") # [DEBUG ADDED]
            
            # Actualizar controlador usando la interfaz
            controller.update_params(new_kp, new_ki, new_kd)

            #self.logger.debug(f"SimulationManager -> _apply_actions_to_controller -> Controller updated OK") # [DEBUG ADDED]

        except KeyError as e:
            self.logger.error(f"Error aplicando acciones al controlador: Clave faltante {e} en config o actions_dict {actions_dict}")
            # No relanzar, puede continuar con las ganancias anteriores.
        except Exception as e:
            self.logger.error(f"Error inesperado aplicando acciones al controlador: {e}", exc_info=True)
            # No relanzar.
        #self.logger.debug(f"SimulationManager -> _apply_actions_to_controller -> End") # [DEBUG ADDED]

    def _run_standard_interval_steps(self, start_time: float, duration: float, current_state: np.ndarray,
                                      environment: Environment, controller: Controller, agent: RLAgent,
                                      metrics_collector: MetricsCollector, config: Dict,
                                      actions_applied_in_interval: Dict # A' que se aplicó
                                     ) -> Tuple[float, float, np.ndarray, bool, str]:
        
        self.logger.debug(f"SimulationManager -> _run_standard_interval_steps -> Start: t={start_time:.4f}, dur={duration:.4f}, state={np.round(current_state[:4], 4)}..., actions={actions_applied_in_interval}") # [DEBUG ADDED]

        """ Ejecuta los pasos dt para un intervalo estándar. """
        interval_reward_sum = 0.0
        interval_stability_scores: List[float] = []
        done = False
        termination_reason = "unknown"
        final_state_in_interval = current_state # Estado al final del intervalo

        # 1.10: Obtener dt de la instancia de environment (inyectada)
        try:
            dt = environment.dt # type: ignore[attr-defined] # Asumir que Environment tiene dt
            if not isinstance(dt, (float, int)) or dt <= 0: raise ValueError("dt inválido desde environment")
        except (AttributeError, ValueError) as e:
             self.logger.error(f"No se pudo obtener dt válido del environment: {e}. Usando 0.001 (riesgoso).")
             dt = 0.001 # Fallback peligroso

        num_steps = max(1, int(round(duration / dt)))

        for step in range(num_steps):
            current_step_time = round(start_time + (step + 1) * dt, 6) # Redondear para evitar flotantes

            # [DEBUG ADDED] - Log llamada y resultado de env.step
            #self.logger.debug(f"SimMan -> _run_standard_interval -> [Step {step+1}] Calling env.step()")

            # A. Ejecutar paso del entorno (usando interfaz)
            try:
                # environment.step() devuelve (next_state, (reward, w_stab), force)
                next_state_vector, (reward_step, stability_score_step), force = environment.step()
                final_state_in_interval = next_state_vector # Actualizar estado final
            
                #self.logger.debug(f"SimMan -> _run_standard_interval -> [Step {step+1}] Result: next_st={np.round(next_state_vector[:4], 8)}, R={reward_step:.8f}, Ws={stability_score_step:.8f}, F={force:.8f}") # [DEBUG ADDED]

            except RuntimeError as e: # Capturar RuntimeError de environment.step
                self.logger.error(f"Error CRÍTICO en environment.step a t={current_step_time:.4f}: {e}. Terminando episodio.", exc_info=True)
                done = True; termination_reason = "env_step_error"
                metrics_collector.log('time', current_step_time) # Loguear tiempo del error
                # Loguear NaNs para el resto de métricas del paso fallido
                nan_metrics = ['reward', 'stability_score', 'force', 'cart_position', 'cart_velocity', 'pendulum_angle', 'pendulum_velocity', 'error', 'kp', 'ki', 'kd', 'integral_error', 'derivative_error', 'action_kp', 'action_ki', 'action_kd', 'epsilon', 'learning_rate', 'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd', 'cumulative_reward']
                for m in nan_metrics: metrics_collector.log(m, np.nan)
                # Nota: cumulative_reward también será NaN si reward lo es.
                break # Salir del bucle de pasos del intervalo

            # B. Acumular métricas del intervalo (asegurando tipos finitos)
            reward_f = float(reward_step) if np.isfinite(reward_step) else 0.0
            stability_score_f = float(stability_score_step) if np.isfinite(stability_score_step) else 0.0 # Usar 0 si w_stab es NaN/inf
            force_f = float(force) if np.isfinite(force) else np.nan

            interval_reward_sum += reward_f
            interval_stability_scores.append(stability_score_f)

            # C. Loguear métricas del paso dt usando MetricsCollector
            metrics_collector.log('time', current_step_time)
            # Estado
            metrics_collector.log('cart_position', next_state_vector[0]); metrics_collector.log('cart_velocity', next_state_vector[1])
            metrics_collector.log('pendulum_angle', next_state_vector[2]); metrics_collector.log('pendulum_velocity', next_state_vector[3])
            # Controlador y Agente
            try:
                ctrl_setpoint = getattr(controller, 'setpoint', 0.0); metrics_collector.log('error', next_state_vector[2] - ctrl_setpoint)
                gains = controller.get_params(); metrics_collector.log('kp', gains.get('kp', np.nan)); metrics_collector.log('ki', gains.get('ki', np.nan)); metrics_collector.log('kd', gains.get('kd', np.nan))
                metrics_collector.log('integral_error', getattr(controller, 'integral_error', np.nan)); metrics_collector.log('derivative_error', getattr(controller, 'derivative_error', np.nan))
                # Log actions_applied_in_interval (A' que se aplicó durante este intervalo)
                metrics_collector.log('action_kp', actions_applied_in_interval.get('kp', np.nan))
                metrics_collector.log('action_ki', actions_applied_in_interval.get('ki', np.nan))
                metrics_collector.log('action_kd', actions_applied_in_interval.get('kd', np.nan))
                metrics_collector.log('epsilon', agent.epsilon); metrics_collector.log('learning_rate', agent.learning_rate)
                # Log gain step
                pid_adapt_cfg = config.get('pid_adaptation', {}); gain_step_config = pid_adapt_cfg.get('gain_step', 5.0); variable_step = pid_adapt_cfg.get('variable_step', False)
                if variable_step and isinstance(gain_step_config, dict):
                     metrics_collector.log('gain_step_kp', float(gain_step_config.get('kp', np.nan)))
                     metrics_collector.log('gain_step_ki', float(gain_step_config.get('ki', np.nan)))
                     metrics_collector.log('gain_step_kd', float(gain_step_config.get('kd', np.nan)))
                elif isinstance(gain_step_config, (int,float)): metrics_collector.log('gain_step', float(gain_step_config))
                else: metrics_collector.log('gain_step', np.nan)
            except AttributeError as ae: self.logger.warning(f"No se pudo loguear atributo en step: {ae}")
            except IndexError: self.logger.warning("Estado con longitud inesperada en step log.")
            # Recompensa, Fuerza, Estabilidad
            metrics_collector.log('reward', reward_f)
            # Calcular cumulativo on-the-fly (más robusto)
            # 1.11: Usar np.nansum para calcular cumulativo
            cumulative_rewards_list = metrics_collector.get_metrics().get('reward', [])
            metrics_collector.log('cumulative_reward', np.nansum(np.array(cumulative_rewards_list, dtype=float)))
            metrics_collector.log('force', force_f); metrics_collector.log('stability_score', stability_score_f)
            # Log NaNs para métricas que solo ocurren en boundary
            nan_boundary_metrics = ['learn_select_duration_ms', 'id_agent_decision',
                                    'td_error_kp', 'td_error_ki', 'td_error_kd',
                                    'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
                                    'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
                                    'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
                                    'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd', 
                                    # Añadir log de NaNs para w_stab virtuales en modo no-Echo
                                    'virtual_w_stab_kp_cf', 'virtual_w_stab_ki_cf', 'virtual_w_stab_kd_cf']
            for m in nan_boundary_metrics: metrics_collector.log(m, np.nan)
            # Log stats adaptativas (si existen)
            if hasattr(metrics_collector, 'log_adaptive_stats'):
                # Intentar obtener stats del stability calculator a través de reward_function
                adaptive_stats = {}
                try:
                    # Asumir que reward_function tiene stability_calculator y este tiene get_current...
                    calculator = getattr(environment.reward_function, 'stability_calculator', None) # type: ignore[attr-defined]
                    if calculator and hasattr(calculator, 'get_current_adaptive_stats'):
                         adaptive_stats = calculator.get_current_adaptive_stats()
                except Exception as e_stats: self.logger.debug(f"No se pudieron obtener stats adaptativas: {e_stats}")
                metrics_collector.log_adaptive_stats(adaptive_stats) # type: ignore[operator]


            # D. Verificar Terminación (usando interfaz)
            if not done: # Solo chequear si no ha terminado ya por error
                try:
                    # Usar config local (pasada como argumento)
                    angle_exc, cart_exc, stab_term = environment.check_termination(config)
                    total_sim_time_per_episode = config.get('environment', {}).get('total_time', 5.0)
                    # 1.12: Usar dt/2 para comparar tiempos flotantes
                    time_limit_reached = (current_step_time >= total_sim_time_per_episode - (dt / 2.0))

                    if angle_exc or cart_exc or stab_term or time_limit_reached:
                        done = True
                        # Asignar razón solo la primera vez que 'done' se vuelve True
                        if termination_reason == "unknown":
                            if angle_exc: termination_reason = "angle_limit"
                            elif cart_exc: termination_reason = "cart_limit"
                            elif stab_term: termination_reason = "stabilized"
                            elif time_limit_reached: termination_reason = "time_limit"
                            else: termination_reason = "unknown_done" # Caso raro
                        self.logger.info(f"Episodio terminando: {termination_reason} at t={current_step_time:.3f}")
                        # No romper aquí, terminar el bucle normalmente
                except Exception as e:
                    self.logger.error(f"Error en environment.check_termination t={current_step_time:.3f}: {e}", exc_info=True)
                    done = True; termination_reason = "termination_check_error"

            # Salir del bucle de pasos si termina el episodio
            if done:
                break

        # Calcular media de estabilidad (asegurando que sea finito)
        avg_interval_stability = np.nanmean(interval_stability_scores) if interval_stability_scores else 1.0
        if not np.isfinite(avg_interval_stability): avg_interval_stability = 1.0 # Default a 1.0 si es NaN/inf

        # [DEBUG ADDED]
        #self.logger.debug(f"SimulationManager -> _run_standard_interval_steps -> End: Rsum={interval_reward_sum:.8f}, AvgWs={avg_interval_stability:.8f}, FinalSt={np.round(final_state_in_interval[:4],8)}, Done={done}, Reason='{termination_reason}'")

        return interval_reward_sum, avg_interval_stability, final_state_in_interval, done, termination_reason

    def _run_echo_baseline_interval_steps(self, start_time: float, duration: float, current_state: np.ndarray,
                                           environment: Environment, controller: Controller, agent: RLAgent,
                                           metrics_collector: MetricsCollector, virtual_simulator: VirtualSimulator,
                                           config: Dict, actions_applied_in_interval: Dict
                                          ) -> Tuple[float, float, np.ndarray, bool, str, Dict[str, float]]:
        
        #self.logger.debug(f"SimulationManager -> _run_echo_baseline_interval_steps -> Start: t={start_time:.4f}, dur={duration:.4f}") # [DEBUG ADDED]

        """ Ejecuta intervalo real y simulaciones virtuales para Echo Baseline. """

        # 1. Ejecutar intervalo real (igual que _run_standard_interval_steps)
        interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real = \
            self._run_standard_interval_steps(
                start_time, duration, current_state, environment, controller, agent,
                metrics_collector, config, actions_applied_in_interval
            )
        
        # [DEBUG ADDED]
        #self.logger.debug(f"SimulationManager -> _run_echo_baseline -> Real Interval Result: R={interval_reward_real:.8f}, Ws={avg_w_stab_real:.8f}, Done={done_real}, Reason='{termination_reason_real}'")

        # Si falló el entorno real críticamente, no simular
        if done_real and termination_reason_real == "env_step_error":
             self.logger.error("EchoBaseline: Fallo crítico en intervalo real, saltando simulaciones virtuales.")
             return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, {}

        # 2. Preparar para simulaciones virtuales
        reward_dict_echo: Dict[str, float] = {}
        interval_start_state = current_state # Estado al INICIO del intervalo real
        # 1.13: Obtener ganancias previas del controlador de forma segura
        gains_applied = controller.get_params()
        prev_kp = getattr(controller, 'prev_kp', gains_applied.get('kp', np.nan))
        prev_ki = getattr(controller, 'prev_ki', gains_applied.get('ki', np.nan))
        prev_kd = getattr(controller, 'prev_kd', gains_applied.get('kd', np.nan))

        # [DEBUG ADDED]
        #self.logger.debug(f"SimulationManager -> _run_echo_baseline -> Virtual Sim Prep: StartSt={np.round(interval_start_state[:4],4)}, PrevGains=(Kp:{prev_kp:.3f}, Ki:{prev_ki:.3f}, Kd:{prev_kd:.3f})")

        # Validar que las ganancias previas sean numéricas
        if any(pd.isna(k) or not np.isfinite(k) for k in [prev_kp, prev_ki, prev_kd]):
             self.logger.warning(f"EchoBaseline: Ganancias previas inválidas (Kp:{prev_kp}, Ki:{prev_ki}, Kd:{prev_kd}). No se pueden ejecutar simulaciones contrafactuales.")
             return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, {}


        # 3. Ejecutar simulaciones virtuales (solo si el intervalo real no falló)
        virtual_sim_start_time = time.time()
        try:
            # Simulación contrafactual para Kp (manteniendo Kp anterior)
            gains_p_cf = {'kp': prev_kp, 'ki': gains_applied['ki'], 'kd': gains_applied['kd']}
            #self.logger.debug(f"SimMan -> _run_echo_baseline -> Running Virtual Sim (Kp CF) with gains: {gains_p_cf}") # [DEBUG ADDED]
            R_p_cf, Ws_p_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_p_cf)

            # Simulación contrafactual para Ki
            gains_i_cf = {'kp': gains_applied['kp'], 'ki': prev_ki, 'kd': gains_applied['kd']}
            #self.logger.debug(f"SimMan -> _run_echo_baseline -> Running Virtual Sim (Ki CF) with gains: {gains_i_cf}") # [DEBUG ADDED]
            R_i_cf, Ws_i_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_i_cf)

            # Simulación contrafactual para Kd
            gains_d_cf = {'kp': gains_applied['kp'], 'ki': gains_applied['ki'], 'kd': prev_kd}
            #self.logger.debug(f"SimMan -> _run_echo_baseline -> Running Virtual Sim (Kd CF) with gains: {gains_d_cf}") # [DEBUG ADDED]
            R_d_cf, Ws_d_cf = virtual_simulator.run_interval_simulation(interval_start_state, start_time, duration, gains_d_cf)

            # Calcular recompensas diferenciales (R_diff = R_real - R_cf)
            reward_dict_echo = {
                'kp': interval_reward_real - R_p_cf,
                'ki': interval_reward_real - R_i_cf,
                'kd': interval_reward_real - R_d_cf
            }

            virtual_sim_duration = (time.time() - virtual_sim_start_time) * 1000
            #self.logger.debug(f"EchoBaseline: Simulaciones virtuales completadas ({virtual_sim_duration:.3f} ms)")
            # [DEBUG ADDED]
            #self.logger.debug(f"SimulationManager -> _run_echo_baseline -> Virtual Sims OK: R_cf=(Kp:{R_p_cf:.8f}, Ki:{R_i_cf:.8f}, Kd:{R_d_cf:.8f}), R_diff={reward_dict_echo}")

            # Log R_diffs virtuales usando MetricsCollector
            # 1.14: Usar método específico del collector si existe
            if hasattr(metrics_collector, 'log_virtual_rewards'):
                metrics_collector.log_virtual_rewards(reward_dict_echo) # type: ignore[operator]
            else: # Log manual si no existe el método específico
                metrics_collector.log('virtual_reward_kp', reward_dict_echo.get('kp', np.nan))
                metrics_collector.log('virtual_reward_ki', reward_dict_echo.get('ki', np.nan))
                metrics_collector.log('virtual_reward_kd', reward_dict_echo.get('kd', np.nan))
            # Loguear Ws virtuales
            metrics_collector.log('virtual_w_stab_kp_cf', Ws_p_cf if np.isfinite(Ws_p_cf) else np.nan)
            metrics_collector.log('virtual_w_stab_ki_cf', Ws_i_cf if np.isfinite(Ws_i_cf) else np.nan)
            metrics_collector.log('virtual_w_stab_kd_cf', Ws_d_cf if np.isfinite(Ws_d_cf) else np.nan)

        except Exception as e:
            self.logger.error(f"EchoBaseline: Error durante simulaciones virtuales: {e}", exc_info=True)
            # Devolver dict vacío si fallan las simulaciones
            reward_dict_echo = {}
            # Loguear NaNs
            if hasattr(metrics_collector, 'log_virtual_rewards'):
                metrics_collector.log_virtual_rewards({}) # type: ignore[operator]
            else:
                metrics_collector.log('virtual_reward_kp', np.nan)
                metrics_collector.log('virtual_reward_ki', np.nan)
                metrics_collector.log('virtual_reward_kd', np.nan)
            # Loguear NaNs para Ws virtuales también en caso de error
            metrics_collector.log('virtual_w_stab_kp_cf', np.nan)
            metrics_collector.log('virtual_w_stab_ki_cf', np.nan)
            metrics_collector.log('virtual_w_stab_kd_cf', np.nan)
        
        #self.logger.debug(f"SimulationManager -> _run_echo_baseline_interval_steps -> End") # [DEBUG ADDED]
        return interval_reward_real, avg_w_stab_real, final_state_real, done_real, termination_reason_real, reward_dict_echo


    def _handle_decision_boundary(self,
                                 current_time: float, # Tiempo al final del intervalo
                                 current_state: np.ndarray, # Estado crudo al final (S')
                                 last_interval_data: Dict, # Datos del intervalo que ACABA de terminar
                                 interval_run_results: Dict, # Resultados del intervalo (R, w_stab, R_diff)
                                 agent: RLAgent, controller: Controller,
                                 metrics_collector: MetricsCollector, config: Dict
                                 ) -> Tuple[Dict[str, int], Optional[Dict]]:
        """ Maneja el aprendizaje, selección de nueva acción (A'') y logueo en el límite de decisión. """
        decision_start_time = time.time()
        # Incrementar contador de decisiones
        agent_decision_count = last_interval_data.get('decision_count', 0) + 1
        metrics_collector.log('id_agent_decision', agent_decision_count)
        #self.logger.debug(f"SimulationManager -> === Decision Boundary Start === (Decision #{agent_decision_count} @ t={current_time:.4f})") # [DEBUG ADDED]
        #self.logger.debug(f"SimMan -> DecisionBoundary -> Input state S' (raw) = {np.round(current_state[:4],8)}") # [DEBUG ADDED]

        # 1. Estado S' (del final del intervalo)
        next_raw_state_vector = current_state # Es el estado final del intervalo anterior
        state_config_for_agent = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        # Construir S' para el agente
        next_agent_state_dict = agent.build_agent_state(next_raw_state_vector, controller, state_config_for_agent)
        episode_done = interval_run_results['done']

        #self.logger.debug(f"SimMan -> DecisionBoundary -> Built next state S' (agent dict) = {next_agent_state_dict}") # [DEBUG ADDED]

        # 2. Preparar datos para Agent Learn (sobre intervalo ANTERIOR S -> A' -> S')
        #    `last_interval_data` contiene S y A'
        #    `interval_run_results` contiene R_real, w_stab (o R_diff)
        #    `next_agent_state_dict` contiene S'
        learn_data = {
            'current_agent_state_dict': last_interval_data['start_state_dict'], # S
            'actions_dict': last_interval_data['actions_dict'],             # A'
            'next_agent_state_dict': next_agent_state_dict,                 # S'
            'controller': controller,                                       # Pasar controller a learn
            'done': episode_done
        }

        # Determinar reward_info basado en la estrategia usada (la info viene de interval_run_results)
        reward_info_for_agent: Union[float, Tuple[float, float], Dict[str, float]]
        if 'reward_dict_echo' in interval_run_results and interval_run_results['reward_dict_echo']: # Si es Echo y R_diff no es vacío
            reward_info_for_agent = interval_run_results['reward_dict_echo']
        else: # Modo Global o Shadow (o Echo con simulación virtual fallida)
            interval_reward_val = interval_run_results.get('interval_reward', 0.0)
            avg_w_stab_val = interval_run_results.get('avg_w_stab', 1.0)
            reward_info_for_agent = (float(interval_reward_val), float(avg_w_stab_val))

        learn_data['reward_info'] = reward_info_for_agent # Añadir info cruda de recompensa

        #self.logger.debug(f"SimMan -> DecisionBoundary -> Reward Info for Learn = {reward_info_for_agent}") # [DEBUG ADDED]

        # 3. Llamar a Agent Learn (usa RewardStrategy internamente)
        last_td_errors_logged = False
        try:
            #self.logger.debug("SimMan -> DecisionBoundary -> Calling agent.learn()") # [DEBUG ADDED]
            agent.learn(**learn_data)
            td_errors = agent.get_last_td_errors()
            #self.logger.debug(f"SimMan -> DecisionBoundary -> TD Errors from agent = {td_errors}") # [DEBUG ADDED]
            # Loguear TD errors DESPUÉS de learn
            # 1.15: Usar método específico del collector si existe
            if hasattr(metrics_collector, 'log_td_errors'):
                # Usar método get_last_td_errors de la interfaz del agente
                metrics_collector.log_td_errors(td_errors) # type: ignore[operator]
                last_td_errors_logged = True
            else: # Log manual
                 metrics_collector.log('td_error_kp', td_errors.get('kp', np.nan))
                 metrics_collector.log('td_error_ki', td_errors.get('ki', np.nan))
                 metrics_collector.log('td_error_kd', td_errors.get('kd', np.nan))
                 last_td_errors_logged = True
                 #self.logger.debug(f"SimMan -> DecisionBoundary -> TD Errors from agent (manual) = {td_errors}") # [DEBUG ADDED]
        except Exception as learn_e:
            self.logger.error(f"Error en agent.learn(): {learn_e}", exc_info=True)
            # Loguear NaN para TD errors si learn falla
            if not last_td_errors_logged:
                 if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors({}) # type: ignore[operator]
                 else: metrics_collector.log('td_error_kp', np.nan); metrics_collector.log('td_error_ki', np.nan); metrics_collector.log('td_error_kd', np.nan)
                 last_td_errors_logged = True

        # 4. Seleccionar Siguiente Acción (A'') basada en S'
        if not episode_done:
            #self.logger.debug(f"SimMan -> DecisionBoundary -> Calling agent.select_action(S')") # [DEBUG ADDED]
            next_actions_prime = agent.select_action(next_agent_state_dict) # Nueva A''
            #self.logger.debug(f"SimMan -> DecisionBoundary -> Selected next action A'' = {next_actions_prime}") # [DEBUG ADDED]
            #self.logger.debug(f"SimMan -> DecisionBoundary -> Applying next action A'' to controller") # [DEBUG ADDED]
        else:
            # Acción neutral si termina, no se usará pero evita errores
            next_actions_prime = {g: 1 for g in ['kp', 'ki', 'kd']}
            #self.logger.debug(f"SimMan -> DecisionBoundary -> Episode done, neutral next action A'' = {next_actions_prime}") # [DEBUG ADDED]

        # 5. Aplicar Nueva Acción A'' al controlador (para el *próximo* intervalo)
        #    Solo aplicar si el episodio NO ha terminado
        if not episode_done:
            self._apply_actions_to_controller(controller, next_actions_prime, config)

        # 6. Loguear Métricas del Bloque de Decisión (asociadas a estado S')
        # 1.16: Usar métodos específicos del collector si existen
        if hasattr(metrics_collector, 'log_q_values'): metrics_collector.log_q_values(agent, next_agent_state_dict) # type: ignore[operator]
        if hasattr(metrics_collector, 'log_q_visit_counts'): metrics_collector.log_q_visit_counts(agent, next_agent_state_dict) # type: ignore[operator]
        if hasattr(metrics_collector, 'log_baselines'): metrics_collector.log_baselines(agent, next_agent_state_dict) # type: ignore[operator]

        # Loguear TD errors si no se hizo antes (en caso de error learn)
        if not last_td_errors_logged:
             if hasattr(metrics_collector, 'log_td_errors'): metrics_collector.log_td_errors(agent.get_last_td_errors()) # type: ignore[operator]
             else: # Log manual
                  td_errors = agent.get_last_td_errors()
                  metrics_collector.log('td_error_kp', td_errors.get('kp', np.nan)); metrics_collector.log('td_error_ki', td_errors.get('ki', np.nan)); metrics_collector.log('td_error_kd', td_errors.get('kd', np.nan))

        decision_duration_ms = (time.time() - decision_start_time) * 1000
        metrics_collector.log('learn_select_duration_ms', decision_duration_ms)

        # 7. Preparar datos para el *siguiente* intervalo (si no 'done')
        next_interval_data = None
        if not episode_done:
            next_interval_data = {
                'start_state_dict': next_agent_state_dict, # S del próximo será S' actual
                'start_raw_state_vector': np.copy(next_raw_state_vector),
                'actions_dict': next_actions_prime.copy(),      # A' del próximo será A'' actual
                'reward_sum': 0.0, 'w_stab_sum': 0.0, 'steps_in_interval': 0, # Resetear acumuladores
                'end_state_dict': None, 'done': False, 'reward_dict_echo': None,
                'decision_count': agent_decision_count # Pasar contador actualizado
             }
            #self.logger.debug(f"SimMan -> DecisionBoundary -> Prepared next interval data") # [DEBUG ADDED]

        #self.logger.debug(f"SimulationManager -> === Decision Boundary End === (Duration: {decision_duration_ms:.3f} ms)") # [DEBUG ADDED]
        # Devolver A'' (aplicada si !done) y datos para próximo intervalo (o None si done)
        return next_actions_prime, next_interval_data

    def _finalize_episode(self, episode: int, episode_metrics_dict: Dict, termination_reason: str,
                          episode_start_time: float, controller: Controller, environment: Environment,
                          agent: RLAgent, results_folder: str, config: Dict,
                          summary_data_list: List, current_episode_batch: List,
                          agent_state_save_freq: int):
        
        #self.logger.debug(f"SimulationManager -> _finalize_episode -> Start (Ep: {episode}, Reason: {termination_reason})") # [DEBUG ADDED]

        """ Realiza el resumen, loggeo y guardado parcial al final de un episodio. """
        episode_duration_s = time.time() - episode_start_time
        # Usar np.nanmax para obtener el último ID de decisión o NaN si no hubo
        last_decision_id = episode_metrics_dict.get('id_agent_decision', [])
        final_decision_count = np.nanmax(last_decision_id) if last_decision_id else 0 # Default a 0 si vacío

        # --- Calcular métricas agregadas del episodio ---
        rewards = np.array(episode_metrics_dict.get('reward', []), dtype=float)
        stability_scores = np.array(episode_metrics_dict.get('stability_score', []), dtype=float)
        total_reward = np.nansum(rewards)
        avg_stability = np.nanmean(stability_scores) if stability_scores.size > 0 else np.nan
        final_time = np.nanmax(episode_metrics_dict.get('time', [])) if episode_metrics_dict.get('time') else np.nan

        # --- Añadir métricas finales al dict detallado ANTES de resumir ---
        # (Necesario para que summarize_episode las incluya)
        len_first_metric = len(next(iter(episode_metrics_dict.values()), []))
        if len_first_metric > 0: # Solo añadir si hay datos
             episode_metrics_dict['termination_reason'] = [termination_reason] * len_first_metric
             episode_metrics_dict['episode_duration_s'] = [episode_duration_s] * len_first_metric
             final_gains = controller.get_params()
             episode_metrics_dict['final_kp'] = [final_gains.get('kp', np.nan)] * len_first_metric
             episode_metrics_dict['final_ki'] = [final_gains.get('ki', np.nan)] * len_first_metric
             episode_metrics_dict['final_kd'] = [final_gains.get('kd', np.nan)] * len_first_metric
             episode_metrics_dict['total_agent_decisions'] = [final_decision_count] * len_first_metric
             episode_metrics_dict['avg_stability_score'] = [avg_stability] * len_first_metric
             episode_metrics_dict['total_reward'] = [total_reward] * len_first_metric
             episode_metrics_dict['episode_time'] = [final_time] * len_first_metric
             # Calcular performance y añadirla también
             perf = total_reward / final_time if pd.notna(total_reward) and pd.notna(final_time) and final_time > 1e-9 else np.nan
             episode_metrics_dict['performance'] = [perf] * len_first_metric
        
        # [DEBUG ADDED] - Log resumen clave
        #self.logger.debug(f"SimMan -> FinalizeEp -> Aggregated: R={total_reward:.8f}, StabAvg={avg_stability:.8f}, Perf={perf:.8f}, T={final_time:.3f}s, Decisions={final_decision_count}, FinalGains={final_gains}")
        #self.logger.debug(f"SimMan -> FinalizeEp -> Calling summarize_episode") # [DEBUG ADDED]

        # --- Llamar a summarize_episode con el dict completo ---
        summary = summarize_episode(episode_metrics_dict)
        # Asegurar que 'episode' está en el resumen (lo añade summarize_episode si falta)
        summary['episode'] = episode
        summary_data_list.append(summary)

        # --- Añadir datos detallados al batch actual ---
        current_episode_batch.append(episode_metrics_dict)

        # --- Log Resumen del Episodio ---
        self.logger.info(f"Ep {episode} Resumen: "
                         f"Term='{summary.get('termination_reason', '?')}', "
                         f"Reward={summary.get('total_reward', np.nan):.2f}, "
                         f"Perf={summary.get('performance', np.nan):.2f}, "
                         f"Stab={summary.get('avg_stability_score', np.nan):.3f}, "
                         f"Time={summary.get('episode_time', np.nan):.2f}s")
        self.logger.info(f"Ep {episode} Resumen: "
                         f"Decisions={summary.get('total_agent_decisions', 0)}, "
                         f"Eps={summary.get('final_epsilon', np.nan):.3f}, "
                         f"LR={summary.get('final_learning_rate', np.nan):.4f}, "
                         f"Gains(Kp={summary.get('final_kp', np.nan):.2f}, "
                         f"Ki={summary.get('final_ki', np.nan):.2f}, "
                         f"Kd={summary.get('final_kd', np.nan):.3f}), "
                         f"Dur={summary.get('episode_duration_s', np.nan):.2f}s")


        # --- Actualizar stats adaptativas (si aplica) ---
        try:
            #self.logger.debug(f"SimMan -> FinalizeEp -> Calling env.update_reward_calculator_stats()") # [DEBUG ADDED]
            # Usar interfaz del entorno
            environment.update_reward_calculator_stats(episode_metrics_dict, episode)
        except Exception as e:
            self.logger.error(f"Error update reward stats ep {episode}: {e}", exc_info=True)

        # --- Guardado Periódico de Estado del Agente ---
        if agent_state_save_freq > 0 and (episode + 1) % agent_state_save_freq == 0:
            #self.logger.debug(f"SimMan -> FinalizeEp -> Calling result_handler.save_agent_state()") # [DEBUG ADDED]
            # Usar result_handler (recibe results_folder explícitamente)
            self.result_handler.save_agent_state(agent, episode, results_folder)
        
        #self.logger.debug(f"SimulationManager -> _finalize_episode -> End (Ep: {episode})") # [DEBUG ADDED]

    # --- Método Público Principal ---

    def run(self) -> Tuple[List[Dict], List[Dict]]:
        """ Ejecuta el bucle principal de simulación. """
        self.logger.info("--- Iniciando Bucle de Simulación Principal ---")
        # 1.17: Inicializar variables locales
        all_episodes_detailed_data: List[Dict] = [] # Rara vez usado si guardamos por batch
        summary_data: List[Dict] = []
        results_folder: Optional[str] = None
        current_episode_batch: List[Dict] = []
        file_handlers: List[logging.Handler] = []
        environment: Optional[Environment] = None
        agent: Optional[RLAgent] = None
        controller: Optional[Controller] = None
        # MetricsCollector es transient, se resuelve por episodio
        reward_strategy: Optional[RewardStrategy] = None
        virtual_simulator: Optional[VirtualSimulator] = None
        config: Optional[Dict] = None
        episode = -1

        # --- 1. Resolver Dependencias y Extraer Config ---
        try:
            environment, agent, controller, _, reward_strategy, \
                virtual_simulator, config, results_folder = self._resolve_dependencies()
            # MetricsCollector se resuelve dentro del bucle de episodios

            # --- Extraer Parámetros Clave de Config ---
            sim_cfg = config.get('simulation', {}); env_cfg = config.get('environment', {})
            logging_cfg = config.get('logging', {})

            max_episodes = env_cfg.get('max_episodes', 1)
            decision_interval = env_cfg.get('decision_interval', 0.01)
            dt_env = env_cfg.get('dt', 0.001) # Obtener dt de config (usado para fallback)
            # Usar dt del environment si está disponible
            try: dt = environment.dt # type: ignore[attr-defined]
            except AttributeError: dt = dt_env

            if not isinstance(decision_interval, (float, int)) or decision_interval < dt:
                 self.logger.warning(f"Decision interval ({decision_interval}) < dt ({dt}). Usando dt como intervalo.")
                 decision_interval = dt
            episodes_per_file = sim_cfg.get('episodes_per_file', 100) # Batch size
            agent_state_save_freq = sim_cfg.get('agent_state_save_frequency', 0) # Default a 0 (deshabilitado)
            if sim_cfg.get('save_agent_state', False) and agent_state_save_freq <= 0:
                 agent_state_save_freq = max_episodes # Guardar al final si está habilitado y freq=0
            elif not sim_cfg.get('save_agent_state', False):
                 agent_state_save_freq = 0 # Deshabilitar si save_agent_state es false

            log_flush_frequency = logging_cfg.get('log_save_frequency', 0) # Freq para flush

            # Determinar si es Echo Baseline
            is_echo_baseline = isinstance(reward_strategy, EchoBaselineRewardStrategy)
            if is_echo_baseline and virtual_simulator is None:
                 # Fail-Fast si Echo requiere simulador pero no está
                 msg = "Echo Baseline activado pero VirtualSimulator no resuelto/configurado."
                 self.logger.critical(msg); raise ValueError(msg)
            
            # [DEBUG ADDED] - Log config clave una vez
            #self.logger.debug(f"SimulationManager -> run -> Config: max_ep={max_episodes}, dec_int={decision_interval}, dt={dt}, batch={episodes_per_file}, save_freq={agent_state_save_freq}, echo={is_echo_baseline}")

            # Preparar handlers para flush periódico
            if log_flush_frequency > 0:
                 # Obtener handlers del logger raíz
                 file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
                 if not file_handlers:
                      self.logger.warning("log_flush_frequency > 0 pero no se encontraron FileHandlers.")
                      log_flush_frequency = 0 # Deshabilitar si no hay handlers

            # --- 2. Bucle Principal de Episodios ---
            self.logger.info(f"Iniciando simulación para {max_episodes} episodios...")
            for episode in range(max_episodes):
                #self.logger.debug(f"SimulationManager -> === Episodio {episode} Start ===") # [DEBUG ADDED]
                episode_start_time = time.time()
                # 1.18: Resolver MetricsCollector (transient) para este episodio
                metrics_collector = self.container.resolve(MetricsCollector)
                if metrics_collector is None: # Debería fallar DI si no se puede resolver
                     self.logger.critical(f"Fallo crítico al resolver MetricsCollector para episodio {episode}. Abortando.")
                     break # Salir del bucle

                # --- 2.a Inicializar Episodio ---
                try:
                    # Pasar componentes resueltos y config
                    current_state, last_interval_data = self._initialize_episode(
                        episode, environment, agent, controller, metrics_collector, config
                    )
                except Exception as init_e:
                    # Loguear error y saltar al siguiente episodio
                    self.logger.error(f"Fallo inicializando episodio {episode}: {init_e}. Saltando episodio.", exc_info=True)
                    # Limpiar datos del episodio fallido si es necesario (MetricsCollector ya está limpio)
                    continue # Saltar al siguiente episodio

                # --- 2.b Bucle de Intervalos de Decisión ---
                current_time = 0.0
                episode_done = False
                termination_reason = "unknown"
                total_sim_time_per_episode = env_cfg.get('total_time', 5.0)

                #self.logger.debug(f"SimMan -> run -> [Ep {episode}] Entering Interval Loop (Max T={total_sim_time_per_episode}s)") # [DEBUG ADDED]

                while not episode_done and current_time < total_sim_time_per_episode:
                    # Calcular duración del intervalo (clamp al tiempo restante)
                    interval_duration = min(decision_interval, total_sim_time_per_episode - current_time)
                    if interval_duration <= dt / 2.0: # Evitar intervalos demasiado cortos
                         break # Salir si el tiempo restante es menor que medio dt
                    
                    # [DEBUG ADDED]
                    #self.logger.debug(f"SimMan -> run -> [Ep {episode}] --- Interval Start --- (t={current_time:.4f}, dur={interval_duration:.4f})")

                    # Acciones A' aplicadas durante este intervalo (vienen de last_interval_data)
                    actions_applied = last_interval_data['actions_dict'].copy()

                    # --- Ejecutar Pasos del Intervalo (Real o con Echo) ---
                    interval_run_results: Dict[str, Any] = {}
                    try:
                        if is_echo_baseline:
                            # Ejecutar con simulaciones virtuales
                            interval_reward, avg_w_stab, final_state, interval_done, term_reason, reward_dict_echo = \
                                self._run_echo_baseline_interval_steps(
                                    current_time, interval_duration, current_state, environment, controller, agent,
                                    metrics_collector, virtual_simulator, config, actions_applied # type: ignore[arg-type]
                                )
                            interval_run_results = {
                                'interval_reward': interval_reward, 'avg_w_stab': avg_w_stab,
                                'final_state': final_state, 'done': interval_done,
                                'termination_reason': term_reason, 'reward_dict_echo': reward_dict_echo
                            }
                        else: # Modo Global o Shadow
                            interval_reward, avg_w_stab, final_state, interval_done, term_reason = \
                                self._run_standard_interval_steps(
                                    current_time, interval_duration, current_state, environment, controller, agent,
                                    metrics_collector, config, actions_applied
                                )
                            interval_run_results = {
                                'interval_reward': interval_reward, 'avg_w_stab': avg_w_stab,
                                'final_state': final_state, 'done': interval_done,
                                'termination_reason': term_reason
                            }
                    except RuntimeError as step_e: # Capturar error de _run_standard_interval_steps
                         self.logger.error(f"Error Runtime durante ejecución intervalo ep {episode} t={current_time:.3f}: {step_e}. Terminando episodio.")
                         episode_done = True
                         termination_reason = "interval_runtime_error"
                         # El estado actual podría ser inválido, intentar finalizar con lo que hay
                         current_state = last_interval_data.get('start_raw_state_vector', current_state) # Volver al estado inicial del intervalo
                         break # Salir del bucle while not episode_done

                    # Actualizar estado y tiempo
                    current_state = interval_run_results['final_state']
                    # Usar tiempo acumulado del collector para mayor precisión
                    last_logged_time = metrics_collector.get_metrics().get('time', [current_time])[-1]
                    current_time = last_logged_time # Actualizar tiempo basado en último log
                    # Actualizar 'done' y razón de terminación
                    episode_done = interval_run_results['done']
                    if episode_done and termination_reason == "unknown":
                         # [DEBUG ADDED]
                         #self.logger.debug(f"SimMan -> run -> [Ep {episode}] --- Interval End --- (t={current_time:.4f}, done={episode_done}, reason='{termination_reason}')")
                         termination_reason = interval_run_results.get('termination_reason', 'interval_ended_done')

                    # --- Manejar Límite de Decisión (Aprendizaje y Selección A'') ---
                    try:
                        _, next_interval_data = self._handle_decision_boundary(
                            current_time, current_state, last_interval_data, interval_run_results,
                            agent, controller, metrics_collector, config
                        )
                    except Exception as decision_e:
                         self.logger.error(f"Error durante _handle_decision_boundary ep {episode} t={current_time:.3f}: {decision_e}. Terminando episodio.", exc_info=True)
                         episode_done = True
                         termination_reason = "decision_boundary_error"
                         next_interval_data = None # No continuar

                    # Preparar para el siguiente intervalo
                    if episode_done:
                        last_interval_data = None # No más intervalos
                        if termination_reason == "unknown": # Si termina por tiempo exacto
                             termination_reason = "time_limit"
                    elif next_interval_data is not None:
                        last_interval_data = next_interval_data
                    else: # Caso raro: no 'done' pero no hay datos para el siguiente
                        self.logger.error("Error lógico: Episodio no terminado pero no hay datos para el siguiente intervalo. Terminando episodio.")
                        episode_done = True
                        termination_reason = "interval_logic_error"
                        last_interval_data = None

                    # Salir si el episodio terminó por alguna razón
                    if episode_done:
                         break

                # --- Fin Bucle de Intervalos ---
                #self.logger.debug(f"SimMan -> run -> [Ep {episode}] Exited Interval Loop") # [DEBUG ADDED]

                # --- 2.c Finalizar Episodio ---
                try:
                    # Recopilar métricas finales del collector de este episodio
                    episode_metrics = metrics_collector.get_metrics()
                    self._finalize_episode(
                        episode, episode_metrics, termination_reason, episode_start_time, controller,
                        environment, agent, results_folder, config, summary_data, current_episode_batch,
                        agent_state_save_freq
                    )
                except Exception as finalize_e:
                     self.logger.error(f"Error durante finalización del episodio {episode}: {finalize_e}", exc_info=True)
                     # No continuar con el guardado de batch si la finalización falló

                # --- Guardado de Batch de Episodios ---
                if episodes_per_file > 0 and ((episode + 1) % episodes_per_file == 0 or episode == max_episodes - 1):
                    if current_episode_batch:
                        #self.logger.debug(f"SimMan -> run -> Saving episode batch (up to ep {episode})") # [DEBUG ADDED]
                        # Pasar results_folder explícitamente
                        self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode)
                        current_episode_batch = [] # Limpiar batch

                # --- Flush Periódico de Logs ---
                if log_flush_frequency > 0 and (episode + 1) % log_flush_frequency == 0:
                    #self.logger.debug(f"Flushing logs file after ep {episode}...")
                    for h in file_handlers:
                        try: h.flush()
                        except Exception as e_flush: self.logger.warning(f"Error flushing handler {h}: {e_flush}")
                #self.logger.debug(f"SimulationManager -> === Episodio {episode} End ===") # [DEBUG ADDED]

            # --- Fin Bucle de Episodios ---

        except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as e:
            # Capturar errores críticos durante la configuración o el bucle principal
            self.logger.critical(f"Error CRÍTICO en bucle de simulación (Ep ~{episode}): {e}", exc_info=True)
            # Intentar guardar batch parcial si existe
            if current_episode_batch and results_folder and os.path.isdir(results_folder):
                 self.logger.warning("Intentando guardar batch parcial de episodios tras error crítico...")
                 try:
                     self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode if 'episode' in locals() else -1)
                 except Exception as save_e:
                      self.logger.error(f"Fallo al guardar batch parcial: {save_e}")
        except Exception as e: # Capturar cualquier otra excepción inesperada
            self.logger.critical(f"Error INESPERADO en bucle de simulación (Ep ~{episode}): {e}", exc_info=True)
            # Intentar guardar batch parcial
            if current_episode_batch and results_folder and os.path.isdir(results_folder):
                 self.logger.warning("Intentando guardar batch parcial de episodios tras error inesperado...")
                 try:
                      self.result_handler.save_episode_batch(current_episode_batch, results_folder, episode if 'episode' in locals() else -1)
                 except Exception as save_e:
                      self.logger.error(f"Fallo al guardar batch parcial: {save_e}")

        finally:
            # Flush final de logs
            if file_handlers:
                 self.logger.info("Realizando flush final de logs...")
                 for h in file_handlers:
                      try: h.flush()
                      except Exception as e_flush_final: self.logger.warning(f"Error flush final handler {h}: {e_flush_final}")
            self.logger.info("--- Simulación Principal Finalizada ---")

        # Devolver datos de resumen (los datos detallados se guardan por batch)
        return all_episodes_detailed_data, summary_data # all_episodes_data suele estar vacío