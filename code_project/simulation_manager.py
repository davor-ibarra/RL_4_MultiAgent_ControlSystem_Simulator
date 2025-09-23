import logging
import time # Para medir duración de intervalos/episodios
from typing import Dict, Any, List, Tuple, Optional, Union # Tipos necesarios
import numpy as np # Para cálculos y NaN
import copy

# Interfaces de componentes que serán resueltos o usados
from interfaces.environment import Environment
from interfaces.rl_agent import RLAgent
from interfaces.controller import Controller
from interfaces.virtual_simulator import VirtualSimulator
from interfaces.metrics_collector import MetricsCollector
from interfaces.reward_strategy import RewardStrategy # Para identificar tipo

# Servicios auxiliares
from result_handler import ResultHandler
from utils.data_processing import summarize_episode # Función utilitaria


class SimulationManager:
    """
    Servicio inyectable que orquesta la ejecución de la simulación principal,
    incluyendo el bucle de episodios y pasos, interacción con componentes
    (Environment, Agent, Controller), cálculo de recompensas contrafactuales
    (si aplica), recolección de métricas y guardado periódico de resultados.
    """

    def __init__(self,
                 logger: logging.Logger,
                 result_handler: ResultHandler,
                 container: None # Inyectar el contenedor para resolver dependencias
                ):
        """
        Inicializa el SimulationManager con sus dependencias principales.
        Componentes específicos como Environment, Agent, etc., se resuelven
        desde el contenedor dentro del método run() o aquí si se prefieren como
        atributos fijos.

        Args:
            logger: Instancia del logger configurado.
            result_handler: Instancia del manejador de resultados.
            container: Instancia del contenedor DI para resolver otras dependencias.
        """
        self.logger = logger
        self.result_handler = result_handler
        self.container = container
        self.logger.info("SimulationManager instance created.")

        # Podríamos resolver componentes aquí si fueran constantes durante toda la vida del manager
        # self.environment = container.resolve(Environment)
        # self.agent = container.resolve(RLAgent)
        # self.controller = container.resolve(Controller)
        # ... etc
        # O resolverlos dentro de run() para mayor flexibilidad si el contenedor cambia

    def _get_dependencies(self) -> Tuple[Environment, RLAgent, Controller, MetricsCollector, RewardStrategy, Optional[VirtualSimulator], Dict[str, Any]]:
        """Resuelve las dependencias necesarias del contenedor."""
        self.logger.debug("Resolviendo dependencias para SimulationManager.run...")
        try:
            environment = self.container.resolve(Environment)
            agent = self.container.resolve(RLAgent)
            controller = self.container.resolve(Controller)
            metrics_collector = self.container.resolve(MetricsCollector) # Obtener nueva instancia
            reward_strategy = self.container.resolve(RewardStrategy)
            # Resolver Optional[VirtualSimulator] devuelve None si no está o no aplica
            virtual_simulator = self.container.resolve(Optional[VirtualSimulator])
            config = self.container.resolve(dict)
            self.logger.debug("Dependencias resueltas exitosamente.")
            return environment, agent, controller, metrics_collector, reward_strategy, virtual_simulator, config
        except ValueError as e:
             self.logger.critical(f"Error fatal resolviendo dependencias: {e}", exc_info=True)
             raise # Relanzar para detener la ejecución

    def run(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Ejecuta el bucle principal de la simulación (episodios y pasos).

        Returns:
            Tuple[List[Dict], List[Dict]]: (all_episodes_detailed_data, summary_data)
        """
        self.logger.info("--- Iniciando Bucle de Simulación ---")
        all_episodes_detailed_data: List[Dict] = []
        summary_data: List[Dict] = []

        try:
            # Resolver dependencias principales al inicio de la ejecución
            environment, agent, controller, metrics_collector, reward_strategy, \
                virtual_simulator, config = self._get_dependencies()

            # Extraer configuraciones relevantes
            sim_cfg = config.get('simulation', {})
            env_cfg = config.get('environment', {})
            pid_adapt_cfg = config.get('pid_adaptation', {})
            init_cond_cfg = config.get('initial_conditions', {})
            reward_setup_cfg = env_cfg.get('reward_setup', {})

            max_episodes = env_cfg.get('max_episodes', 1)
            total_sim_time_per_episode = env_cfg.get('total_time', 5.0)
            decision_interval = env_cfg.get('decision_interval', 0.01)
            dt = env_cfg.get('dt', 0.001)
            episodes_per_file = sim_cfg.get('episodes_per_file', 100)
            agent_state_save_freq = sim_cfg.get('agent_state_save_frequency', 1000)
            log_flush_frequency = config.get('logging', {}).get('log_save_frequency', 0)
            results_folder = self.container.resolve(str) # Obtener carpeta de resultados

            initial_conditions = init_cond_cfg.get('x0', [0.0, 0.0, 0.0, 0.0])

            # Determinar si es modo Echo Baseline
            is_echo_baseline = reward_setup_cfg.get('learning_strategy') == 'echo_baseline'
            if is_echo_baseline and virtual_simulator is None:
                 self.logger.error("Modo Echo Baseline seleccionado pero VirtualSimulator no está disponible/configurado. Abortando.")
                 return [], []

            # Obtener file handlers para flush periódico
            file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and hasattr(h, 'flush')]

            # --- Bucle de Episodios ---
            for episode in range(max_episodes):
                episode_start_time = time.time()
                self.logger.info(f"--- Iniciando Episodio {episode}/{max_episodes-1} ---")
                metrics_collector.reset(episode_id=episode) # Resetear colector para nuevo episodio

                try:
                    current_state_vector = environment.reset(initial_conditions) # Resetea env, controller (si aplica), agent (epsilon/lr)
                    self.logger.debug(f"Episodio {episode}: Estado inicial={np.round(current_state_vector, 4)}")
                except Exception as e:
                    self.logger.error(f"Error crítico reseteando el entorno para episodio {episode}: {e}", exc_info=True)
                    continue # Saltar al siguiente episodio si el reset falla

                t = 0.0
                interval_start_time = t
                interval_reward_accumulator = 0.0
                interval_w_stab_accumulator = 0.0
                interval_step_count = 0

                # Datos del intervalo anterior para aprendizaje diferido
                last_interval_data = None

                # --- Bucle de Pasos dentro del Episodio ---
                while t < total_sim_time_per_episode:
                    current_step_start_time = time.time() # Para medir duración del paso
                    # --- Inicio del Intervalo de Decisión ---
                    if t >= interval_start_time + decision_interval or t == 0.0: # t==0.0 para la primera decisión

                        # 1. APRENDER del intervalo ANTERIOR (si existe)
                        if last_interval_data is not None:
                            # Calcular avg_w_stab del intervalo anterior
                            avg_w_stab = (last_interval_data['w_stab_sum'] / last_interval_data['steps']) \
                                          if last_interval_data['steps'] > 0 else 1.0
                            # Preparar reward_info según estrategia
                            reward_info: Union[float, Tuple[float, float], Dict[str, float]]
                            if is_echo_baseline:
                                reward_info = last_interval_data['reward_dict'] # Usar R_diff calculado antes
                            elif isinstance(reward_strategy, ShadowBaselineRewardStrategy): # type: ignore # Comparar con tipo
                                reward_info = (last_interval_data['reward_sum'], avg_w_stab)
                            else: # Global
                                reward_info = last_interval_data['reward_sum']

                            # Agent learns using S, A from start of previous interval, and R, S' from end
                            agent.learn(
                                current_agent_state_dict=last_interval_data['start_state_dict'],
                                actions_dict=last_interval_data['actions_dict'],
                                reward_info=reward_info,
                                next_agent_state_dict=last_interval_data['end_state_dict'],
                                done=last_interval_data['done'] # Done se determina al final del intervalo
                            )
                            # Log métricas de aprendizaje (TD errors, etc.)
                            metrics_collector.log_td_errors(agent.get_last_td_errors())

                        # 2. OBTENER ESTADO Y SELECCIONAR ACCIÓN para el NUEVO intervalo
                        # Construir estado para el agente (puede incluir ganancias actuales)
                        start_state_dict = agent.build_agent_state(current_state_vector, controller, env_cfg['agent']['params']['state_config'])
                        actions_dict = agent.select_action(start_state_dict) # Diccionario {'kp': 0, 'ki': 1, ...}

                        # Log Q-values, visit counts, baselines ANTES de aplicar acción
                        metrics_collector.log_q_values(agent, start_state_dict)
                        metrics_collector.log_q_visit_counts(agent, start_state_dict)
                        # Baselines solo son relevantes en Shadow mode, pero loguear igual (serán 0 o NaN si no)
                        metrics_collector.log_baselines(agent, start_state_dict)

                        # 3. APLICAR ACCIONES AL CONTROLADOR (actualizar ganancias Kp, Ki, Kd)
                        # Obtener ganancias actuales y calcular las nuevas
                        current_gains = controller.get_params()
                        new_gains = {}
                        gain_step_config = pid_adapt_cfg.get('gain_step', 5.0) # Puede ser float o dict
                        variable_step = pid_adapt_cfg.get('variable_step', False)

                        for gain, action_idx in actions_dict.items():
                            step_size = gain_step_config if not variable_step else gain_step_config.get(gain, 5.0)
                            delta = 0
                            if action_idx == 0: delta = -step_size # Decrease
                            elif action_idx == 2: delta = step_size # Increase
                            # Calcular nueva ganancia y asegurar que no sea negativa
                            new_gains[gain] = max(0.0, current_gains[gain] + delta)

                        # Actualizar controlador con las nuevas ganancias calculadas
                        controller.update_params(new_gains['kp'], new_gains['ki'], new_gains['kd'])

                        # --- Lógica Específica ECHO BASELINE ---
                        reward_dict_for_learn: Optional[Dict[str, float]] = None
                        if is_echo_baseline and virtual_simulator is not None:
                            # Ejecutar simulación REAL para el intervalo para obtener R_real
                            # (Necesitamos simular el intervalo que viene antes de poder calcular R_diff)
                            real_reward_in_interval = 0.0
                            temp_state = np.array(current_state_vector) # Copia para simulación real
                            temp_time = t
                            temp_steps = 0
                            temp_controller = copy.deepcopy(controller) # Usar copia para no afectar estado real
                            temp_controller.reset_internal_state() # Resetear errores/integral de la copia

                            for _ in range(int(round(decision_interval / dt))):
                                if temp_time >= total_sim_time_per_episode: break
                                step_real_reward, step_real_w_stab, _ = environment.reward_function.calculate(
                                    temp_state, temp_controller.compute_action(temp_state),
                                    environment.system.apply_action(temp_state, temp_controller.compute_action(temp_state), temp_time, dt),
                                    temp_time
                                )
                                temp_state = environment.system.apply_action(temp_state, temp_controller.compute_action(temp_state), temp_time, dt)
                                real_reward_in_interval += step_real_reward
                                temp_time += dt
                                temp_steps +=1
                            del temp_controller # Liberar copia

                            if temp_steps == 0: real_reward_in_interval = 0.0 # Evitar división por cero si el intervalo es 0

                            # Ejecutar simulaciones VIRTUALES contrafactuales
                            reward_dict_for_learn = {}
                            gain_to_vary: str
                            for gain_to_vary in ['kp', 'ki', 'kd']:
                                counterfactual_gains = new_gains.copy()
                                # Asumimos que action=1 (mantener) es la acción base
                                # Necesitamos saber qué acción se habría tomado si la ganancia se mantenía
                                # Esto complica la lógica, ¿quizás el R_diff es R_real - R_virtual(gains_sin_cambio)?
                                # O R_diff = R_virtual(gains_con_cambio) - R_virtual(gains_sin_cambio)?
                                # --> Adoptemos R_diff = R_real(Acción Tomada) - R_virtual(Acción Mantener)
                                # Necesitamos R_virtual si la acción para 'gain_to_vary' hubiera sido 1 (mantener)
                                cf_gains_maintain = current_gains.copy() # Empezar desde las ganancias *antes* del cambio
                                cf_gains_maintain.update({g: new_gains[g] for g in new_gains if g != gain_to_vary}) # Actualizar las *otras* ganancias

                                virtual_reward = virtual_simulator.run_interval_simulation(
                                    initial_state_vector=current_state_vector,
                                    start_time=t,
                                    duration=decision_interval,
                                    controller_gains_dict=cf_gains_maintain # Usar ganancias donde 'gain_to_vary' se mantuvo
                                )
                                reward_dict_for_learn[gain_to_vary] = real_reward_in_interval - virtual_reward
                                # Loguear recompensa virtual calculada
                                metrics_collector.log(f'virtual_reward_{gain_to_vary}', virtual_reward)

                        # 4. PREPARAR DATOS para el aprendizaje en el *próximo* intervalo
                        last_interval_data = {
                            'start_state_dict': start_state_dict, # Estado S al inicio del intervalo actual
                            'actions_dict': actions_dict.copy(),  # Acción A tomada al inicio del intervalo actual
                            'reward_sum': 0.0,                  # Se acumulará durante el intervalo actual
                            'w_stab_sum': 0.0,                  # Se acumulará durante el intervalo actual
                            'steps': 0,                         # Se contará durante el intervalo actual
                            'end_state_dict': None,             # Se rellenará al final del intervalo actual
                            'done': False,                      # Se rellenará al final del intervalo actual
                            'reward_dict': reward_dict_for_learn # R_diff calculado (solo para Echo)
                        }

                        # Resetear acumuladores para el nuevo intervalo
                        interval_start_time = t
                        interval_reward_accumulator = 0.0
                        interval_w_stab_accumulator = 0.0
                        interval_step_count = 0
                        # Log acción tomada y ganancias actualizadas
                        metrics_collector.log('action_kp', actions_dict.get('kp', np.nan))
                        metrics_collector.log('action_ki', actions_dict.get('ki', np.nan))
                        metrics_collector.log('action_kd', actions_dict.get('kd', np.nan))
                        metrics_collector.log('kp', controller.get_params()['kp'])
                        metrics_collector.log('ki', controller.get_params()['ki'])
                        metrics_collector.log('kd', controller.get_params()['kd'])
                        metrics_collector.log('epsilon', agent.epsilon) # Log current epsilon
                        metrics_collector.log('learning_rate', agent.learning_rate) # Log current LR


                    # --- Ejecutar un paso de la simulación física ---
                    try:
                        next_state_vector, (reward, w_stab), force = environment.step()
                    except Exception as e:
                         self.logger.error(f"Error durante environment.step() en t={t:.4f}: {e}", exc_info=True)
                         # Decidir cómo manejar el error: terminar episodio? usar estado anterior?
                         # Por ahora, terminamos el episodio
                         if last_interval_data: last_interval_data['done'] = True
                         break # Salir del bucle de pasos


                    # --- Acumular recompensa y estabilidad del intervalo ---
                    interval_reward_accumulator += reward
                    interval_w_stab_accumulator += w_stab
                    interval_step_count += 1
                    if last_interval_data:
                        last_interval_data['reward_sum'] += reward
                        last_interval_data['w_stab_sum'] += w_stab
                        last_interval_data['steps'] += 1


                    # --- Loggear métricas del paso actual ---
                    metrics_collector.log('time', t + dt) # Log time at end of step
                    metrics_collector.log('reward', reward)
                    metrics_collector.log('stability_score', w_stab)
                    metrics_collector.log('force', force)
                    if next_state_vector is not None and len(next_state_vector) >= 4:
                        metrics_collector.log('cart_position', next_state_vector[0])
                        metrics_collector.log('cart_velocity', next_state_vector[1])
                        metrics_collector.log('pendulum_angle', next_state_vector[2])
                        metrics_collector.log('pendulum_velocity', next_state_vector[3])
                    # Log controller internal state if available
                    if hasattr(controller, 'integral_error'):
                        metrics_collector.log('integral_error', controller.integral_error) # type: ignore
                    if hasattr(controller, 'derivative_error'):
                        metrics_collector.log('derivative_error', controller.derivative_error) # type: ignore
                    if hasattr(controller, 'prev_error'):
                        metrics_collector.log('error', controller.prev_error) # type: ignore
                    # Log adaptive stats if calculator exists and has the method
                    if hasattr(environment.reward_function, 'stability_calculator') and \
                       environment.reward_function.stability_calculator and \
                       hasattr(environment.reward_function.stability_calculator, 'get_current_adaptive_stats'):
                        adaptive_stats = environment.reward_function.stability_calculator.get_current_adaptive_stats()
                        if adaptive_stats: # Solo loguear si no está vacío
                            metrics_collector.log_adaptive_stats(adaptive_stats)


                    # --- Actualizar estado y tiempo ---
                    current_state_vector = next_state_vector
                    t += dt

                    # --- Verificar Terminación del Episodio ---
                    angle_exceeded, cart_exceeded, stabilized = environment.check_termination(config)
                    done = angle_exceeded or cart_exceeded or stabilized or (t >= total_sim_time_per_episode)
                    termination_reason = "max_time"
                    if angle_exceeded: termination_reason = "angle_limit"
                    elif cart_exceeded: termination_reason = "cart_limit"
                    elif stabilized: termination_reason = "stabilized"

                    # Medir duración del paso
                    step_duration_ms = (time.time() - current_step_start_time) * 1000
                    metrics_collector.log('step_duration_ms', step_duration_ms)

                    if done:
                        if last_interval_data:
                            last_interval_data['done'] = True
                            # Estado final para el aprendizaje diferido
                            last_interval_data['end_state_dict'] = agent.build_agent_state(current_state_vector, controller, env_cfg['agent']['params']['state_config'])
                        self.logger.info(f"Episodio {episode} terminado en t={t:.4f}. Razón: {termination_reason}")
                        break # Salir del bucle de pasos

                # --- Fin del Bucle de Pasos ---

                # Procesamiento Post-Episodio
                episode_duration_s = time.time() - episode_start_time
                episode_metrics = metrics_collector.get_metrics() # Obtener datos recolectados
                episode_metrics['termination_reason'] = [termination_reason] * len(episode_metrics.get('time', [1])) # Añadir razón (replicada)
                episode_metrics['episode_duration_s'] = [episode_duration_s] * len(episode_metrics.get('time', [1]))
                episode_metrics['final_kp'] = [controller.get_params()['kp']] * len(episode_metrics.get('time', [1]))
                episode_metrics['final_ki'] = [controller.get_params()['ki']] * len(episode_metrics.get('time', [1]))
                episode_metrics['final_kd'] = [controller.get_params()['kd']] * len(episode_metrics.get('time', [1]))
                # Calcular decisiones totales
                total_decisions = sum(1 for x in episode_metrics.get('action_kp', []) if not np.isnan(x))
                episode_metrics['total_agent_decisions'] = [total_decisions] * len(episode_metrics.get('time', [1]))

                all_episodes_detailed_data.append(episode_metrics)
                summary = summarize_episode(episode_metrics) # Generar resumen
                summary_data.append(summary)
                self.logger.info(f"Episodio {episode} Resumen: Reward={summary.get('total_reward', np.nan):.2f}, "
                                 f"Perf={summary.get('performance', np.nan):.2f}, "
                                 f"Eps={agent.epsilon:.3f}, LR={agent.learning_rate:.4f}, "
                                 f"Dur={episode_duration_s:.2f}s")

                # --- Actualizar estadísticas adaptativas (si aplica) ---
                try:
                     environment.update_reward_calculator_stats(episode_metrics, episode)
                except Exception as e:
                     self.logger.error(f"Error actualizando estadísticas del reward calculator: {e}", exc_info=True)


                # --- Guardado Periódico ---
                if (episode + 1) % episodes_per_file == 0 or episode == max_episodes - 1:
                    self.result_handler.save_episode_batch(
                        all_episodes_detailed_data, results_folder, episode
                    )
                    all_episodes_detailed_data = [] # Limpiar batch después de guardar

                if agent_state_save_freq > 0 and (episode + 1) % agent_state_save_freq == 0:
                    self.result_handler.save_agent_state(agent, episode, results_folder)

                # --- Flush de Logs Periódico ---
                if log_flush_frequency > 0 and (episode + 1) % log_flush_frequency == 0:
                     self.logger.debug(f"Flushing logs to file after episode {episode}...")
                     for h in file_handlers:
                          try: h.flush()
                          except Exception as e_flush: self.logger.warning(f"Error flushing handler {h}: {e_flush}")


            # --- Fin del Bucle de Episodios ---

        except Exception as e:
            self.logger.error(f"Error inesperado en el bucle de simulación principal: {e}", exc_info=True)
            # Considerar guardar datos parciales si es posible
            if all_episodes_detailed_data:
                 self.logger.info("Intentando guardar batch de episodios parcial...")
                 self.result_handler.save_episode_batch(all_episodes_detailed_data, results_folder, episode if 'episode' in locals() else -1)


        finally:
            # --- Flush final de logs ---
            self.logger.info("Realizando flush final de logs...")
            for h in file_handlers:
                 try: h.flush()
                 except Exception as e_flush: self.logger.warning(f"Error en flush final del handler {h}: {e_flush}")

            self.logger.info("--- Simulación Principal Completada ---")

        return all_episodes_detailed_data, summary_data # Devuelve datos (puede estar vacío si hubo error temprano)