import numpy as np
import pandas as pd
import logging
from interfaces.rl_agent import RLAgent # Importar Interfaz
from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from interfaces.controller import Controller # Para type hint
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List

# Importar Shadow strategy sólo para isinstance check (opcional pero claro)
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy

# 1.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class PIDQLearningAgent(RLAgent): # Implementar Interfaz RLAgent
    """
    Agente Q-Learning para ajustar ganancias PID. Implementa RLAgent.
    Utiliza RewardStrategy inyectada para el cálculo de R_learn.
    """
    def __init__(self,
                 # --- Dependencia Inyectada ---
                 reward_strategy: RewardStrategy,
                 # --- Configuración del Agente (desde agent_params en factory) ---
                 state_config: Dict[str, Dict[str, Any]],
                 num_actions: int,
                 gain_step: Union[float, Dict[str, float]],
                 variable_step: bool,
                 # --- Parámetros de Aprendizaje ---
                 discount_factor: float = 0.98,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.99954,
                 learning_rate: float = 1.0,
                 learning_rate_min: float = 0.01,
                 learning_rate_decay: float = 0.999425,
                 use_epsilon_decay: bool = True,
                 use_learning_rate_decay: bool = True,
                 # --- Inicialización de Tablas ---
                 q_init_value: float = 0.0,
                 visit_init_value: int = 0,
                 # --- Parámetros Opcionales (para estrategias específicas) ---
                 shadow_baseline_params: Optional[Dict] = None # Inyectado por factory si aplica
                 ):
        """
        Inicializa el agente PID Q-Learning.
        Valida dependencias y configuración esencial.
        """
        logger.info("Inicializando PIDQLearningAgent...")
        # 1.2: Validar dependencia RewardStrategy (Fail-Fast)
        if not isinstance(reward_strategy, RewardStrategy):
             msg = f"Dependencia inválida: 'reward_strategy' debe ser instancia de RewardStrategy, no {type(reward_strategy).__name__}."
             logger.critical(msg)
             raise TypeError(msg)
        self.reward_strategy = reward_strategy
        logger.info(f"Usando Reward Strategy: {type(self.reward_strategy).__name__}")

        # 1.3: Validar y preparar state_config (Fail-Fast)
        try:
             self.state_config = self._validate_and_prepare_state_config(state_config)
        except ValueError as e:
             logger.critical(f"Configuración de estado inválida: {e}", exc_info=True)
             raise # Relanzar error crítico de config

        # 1.4: Asignar parámetros básicos y validar tipos/valores esenciales
        if not isinstance(num_actions, int) or num_actions <= 0:
             raise ValueError(f"num_actions ({num_actions}) debe ser un entero positivo.")
        self.num_actions = num_actions
        self.q_init_value = float(q_init_value)
        self.visit_init_value = int(visit_init_value)

        self.variable_step = variable_step
        if variable_step:
            if not isinstance(gain_step, dict): raise ValueError("variable_step=True, pero gain_step no es dict.")
        elif not isinstance(gain_step, (float, int)): raise ValueError("variable_step=False, pero gain_step no es numérico.")
        self.gain_step = gain_step # Guardar config original

        self.discount_factor = float(discount_factor)
        if not (0 <= self.discount_factor <= 1): raise ValueError("discount_factor debe estar entre 0 y 1.")
        # --- Learning Rates & Epsilon ---
        self._initial_epsilon = float(epsilon)
        self._initial_learning_rate = float(learning_rate)
        self._epsilon = self._initial_epsilon
        self._learning_rate = self._initial_learning_rate
        self._epsilon_min = float(epsilon_min)
        self._learning_rate_min = float(learning_rate_min)
        self._epsilon_decay = float(epsilon_decay)
        self._learning_rate_decay = float(learning_rate_decay)
        self.use_epsilon_decay = bool(use_epsilon_decay)
        self.use_learning_rate_decay = bool(use_learning_rate_decay)

        # Configuración Baseline (Shadow Mode)
        self.is_shadow_mode = isinstance(self.reward_strategy, ShadowBaselineRewardStrategy)
        self.baseline_init_value = 0.0
        if self.is_shadow_mode:
             # Extraer baseline_init_value de los params opcionales
             baseline_params_dict = shadow_baseline_params if isinstance(shadow_baseline_params, dict) else {}
             self.baseline_init_value = float(baseline_params_dict.get('baseline_init_value', 0.0))
             logger.info(f"Shadow Baseline activado. B(s) init: {self.baseline_init_value}")

        # Inicialización de Estructuras Internas
        self.ordered_state_vars_for_gain: Dict[str, List[str]] = {}
        self.gain_variables = ['kp', 'ki', 'kd'] # Dimensiones controlables
        self.q_tables_np: Dict[str, np.ndarray] = {}
        self.visit_counts_np: Dict[str, np.ndarray] = {}
        self.baseline_tables_np: Dict[str, np.ndarray] = {}
        self._last_td_errors: Dict[str, float] = {gain: np.nan for gain in self.gain_variables}

        # Crear e Inicializar Tablas NumPy
        self._initialize_tables()

        logger.info("PIDQLearningAgent inicializado exitosamente.")
        if not self.q_tables_np:
            logger.warning("¡Ninguna tabla Q fue inicializada! Verificar 'state_config' en config.")


    def _validate_and_prepare_state_config(self, config: Dict) -> Dict:
        """Valida state_config. Lanza ValueError si es inválido."""
        #logger.debug("Validando state_config...")
        validated_config = {}
        if not isinstance(config, dict): raise ValueError("state_config debe ser dict.")

        required_keys_enabled = ['min', 'max', 'bins']
        for var, cfg in config.items():
            if not isinstance(cfg, dict): raise ValueError(f"Config para '{var}' debe ser dict.")
            if 'enabled' not in cfg: raise ValueError(f"Config para '{var}' falta 'enabled'.")
            if not isinstance(cfg['enabled'], bool): raise ValueError(f"Config '{var}': 'enabled' debe ser bool.")

            validated_config[var] = cfg.copy()
            if cfg['enabled']:
                missing_keys = [key for key in required_keys_enabled if key not in cfg]
                if missing_keys: raise ValueError(f"Config habilitada '{var}' faltan claves: {missing_keys}.")
                if not all(isinstance(cfg.get(k), (int, float)) for k in ['min', 'max']):
                     raise ValueError(f"Config '{var}': 'min'/'max' deben ser numéricos.")
                if cfg['min'] >= cfg['max']:
                     raise ValueError(f"Config '{var}': 'min' ({cfg['min']}) debe ser < 'max' ({cfg['max']}).")
                if not isinstance(cfg.get('bins'), int) or cfg['bins'] <= 0:
                     raise ValueError(f"Config '{var}': 'bins' ({cfg.get('bins')}) debe ser entero positivo.")
                #logger.debug(f" - Var '{var}': Habilitada, Min={cfg['min']}, Max={cfg['max']}, Bins={cfg['bins']}")
        #logger.debug("state_config validado.")
        return validated_config

    def _initialize_tables(self):
        """Inicializa las tablas Q, N y B basadas en state_config y state_vars."""
        logger.info("Inicializando tablas NumPy (Q, N, B)...")
        self.ordered_state_vars_for_gain = {} # Resetear mapa
        self.q_tables_np = {}
        self.visit_counts_np = {}
        self.baseline_tables_np = {}

        for gain in self.gain_variables:
            # Obtener la lista de variables de estado para esta tabla Q
            # _get_ordered_vars_for_gain ya verifica si la ganancia está habilitada
            ordered_vars_list = self._get_ordered_vars_for_gain(gain)

            # Si la lista está vacía, no crear tabla Q para esta ganancia
            if not ordered_vars_list:
                logger.info(f"No se creará tabla Q para ganancia '{gain}' (no habilitada o sin variables de estado).")
                continue

            logger.debug(f"Inicializando tablas para ganancia '{gain}' con estado: {ordered_vars_list}")
            self.ordered_state_vars_for_gain[gain] = ordered_vars_list # Guardar las variables usadas

            try:
                # Obtener las dimensiones (bins) de cada variable de estado
                state_dims = []
                for var_name in ordered_vars_list:
                    # Acceder a la config global de la variable (ya sabemos que existe y está habilitada)
                    var_config = self.state_config[var_name]
                    bins = var_config.get('bins')
                    if bins is None or not isinstance(bins, int) or bins <= 0:
                         # Este error no debería ocurrir si _validate_and_prepare_state_config funcionó
                         raise ValueError(f"Configuración de 'bins' inválida para variable habilitada '{var_name}'.")
                    state_dims.append(bins)

                if not state_dims:
                    # Esto no debería ocurrir si ordered_vars_list no estaba vacía
                    raise ValueError(f"No se pudieron determinar dimensiones de estado para '{gain}'.")

                q_visit_shape = tuple(state_dims + [self.num_actions])
                baseline_shape = tuple(state_dims) # Baseline no tiene dimensión de acción
                #logger.debug(f"  - Shape Q/Visit '{gain}': {q_visit_shape}")

                # Crear tablas NumPy
                self.q_tables_np[gain] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
                self.visit_counts_np[gain] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)

                # Crear tabla Baseline solo si estamos en Shadow mode
                if self.is_shadow_mode:
                    self.baseline_tables_np[gain] = np.full(baseline_shape, self.baseline_init_value, dtype=np.float32)
                    #logger.debug(f"  - Shape Baseline '{gain}': {baseline_shape}")

            # Manejo de errores durante la creación de tablas
            except KeyError as e:
                logger.critical(f"Error CRÍTICO inicializando tablas para '{gain}': Falta config para variable {e}. Abortando.", exc_info=True)
                raise RuntimeError(f"Fallo al inicializar tablas para {gain}: Falta config {e}") from e
            except ValueError as e:
                 logger.critical(f"Error CRÍTICO inicializando tablas para '{gain}': {e}. Abortando.", exc_info=True)
                 raise RuntimeError(f"Fallo al inicializar tablas para {gain}: {e}") from e
            except Exception as e:
                 logger.critical(f"Error CRÍTICO inesperado inicializando tablas para '{gain}': {e}", exc_info=True)
                 raise RuntimeError(f"Fallo inesperado al inicializar tablas para {gain}") from e

    def _get_ordered_vars_for_gain(self, gain_to_define: str) -> List[str]:
        """Obtiene la lista ordenada de variables de estado *habilitadas* relevantes para una ganancia."""
        # Verificar si la propia ganancia está habilitada
        gain_config = self.state_config.get(gain_to_define, {})
        if not gain_config.get('enabled', False):
            # Si la ganancia principal no está habilitada, no se crea tabla Q para ella.
            #logger.debug(f"Ganancia '{gain_to_define}' no habilitada. No se definen variables de estado.")
            return []

        # Empezar con la propia ganancia como primera variable de estado
        ordered_state_var_names = OrderedDict()
        ordered_state_var_names[gain_to_define] = True

        # Obtener la lista de variables adicionales desde la config de la ganancia
        additional_vars = gain_config.get('state_vars', []) # Default a lista vacía si no existe

        if not isinstance(additional_vars, list):
            logger.warning(f"Config 'state_vars' para ganancia '{gain_to_define}' no es una lista. Ignorando variables adicionales.")
            additional_vars = []

        # Añadir variables adicionales SOLO SI están habilitadas globalmente en state_config
        for var_name in additional_vars:
            if var_name == gain_to_define: continue # Evitar duplicados
            var_global_config = self.state_config.get(var_name)
            if var_global_config and isinstance(var_global_config, dict) and var_global_config.get('enabled', False):
                # Añadir al OrderedDict para mantener orden y evitar duplicados
                ordered_state_var_names[var_name] = True
            else:
                logger.warning(f"Variable '{var_name}' especificada en 'state_vars' de '{gain_to_define}' "
                               f"no existe o no está habilitada globalmente en state_config. Será ignorada.")

        final_list = list(ordered_state_var_names.keys())
        #logger.debug(f"Variables de estado ordenadas para Q-table de '{gain_to_define}': {final_list}")
        return final_list

    def _cleanup_failed_gain_init(self, gain: str):
        """Helper para limpiar tablas si la inicialización falla para una ganancia."""
        self.ordered_state_vars_for_gain.pop(gain, None)
        self.q_tables_np.pop(gain, None)
        self.visit_counts_np.pop(gain, None)
        self.baseline_tables_np.pop(gain, None)

    def _discretize_value(self, value: float, var_name: str) -> Optional[int]:
        """
        Discretiza un valor asignándolo al índice del bin cuyo *centro* está más cercano.
        Devuelve None si la variable no está configurada, no habilitada, o el valor es inválido.
        """
        if var_name not in self.state_config: return None
        config = self.state_config[var_name]
        if not config.get('enabled', False): return None

        min_val, max_val, bins = config['min'], config['max'], config['bins']

        if pd.isna(value) or not np.isfinite(value):
            logger.warning(f"Valor inválido (NaN/inf) para discretizar '{var_name}': {value}.")
            return None
        if not isinstance(bins, int) or bins <= 0:
            logger.error(f"Config 'bins' inválida para '{var_name}': {bins}")
            return None

        # Clip value to the defined range [min_val, max_val]
        clipped_value = np.clip(value, min_val, max_val)

        # Handle edge case: bins = 1
        if bins == 1:
            return 0
        # Handle edge case: min_val == max_val
        if np.isclose(min_val, max_val):
            return 0

        # --- Lógica "Punto Representativo Más Cercano" ---
        # 1. Calcular los 'bins' puntos representativos uniformemente distribuidos
        #    INCLUYENDO min_val y max_val.
        representative_points = np.linspace(min_val, max_val, bins)

        # 2. Encontrar el índice del punto representativo más cercano
        closest_point_index = int(np.argmin(np.abs(representative_points - clipped_value)))

        #logger.debug(f"Discretize '{var_name}': val={value:.4f}, clipped={clipped_value:.4f}, bins={bins}, points={np.round(representative_points, 3)}, closest_idx={closest_point_index}")

        return closest_point_index


    def get_discrete_state_indices_tuple(self, agent_state_dict: Optional[Dict[str, Any]], gain_variable: str) -> Optional[tuple]:
        """Convierte el dict de estado en tupla de índices para una ganancia. Devuelve None si falla."""
        if agent_state_dict is None:
            #logger.debug(f"get_indices: agent_state_dict es None para '{gain_variable}'.")
            return None
        if gain_variable not in self.ordered_state_vars_for_gain:
            #logger.debug(f"get_indices: No hay variables ordenadas para '{gain_variable}'.")
            return None # No hay vars habilitadas para esta ganancia

        ordered_vars = self.ordered_state_vars_for_gain[gain_variable]
        indices = []
        try:
            for var_name in ordered_vars:
                # 1.5: Validar presencia de la clave en el estado (Fail-Fast si falta)
                if var_name not in agent_state_dict:
                    #logger.warning(f"Variable habilitada '{var_name}' no encontrada en agent_state_dict para '{gain_variable}'. Keys: {list(agent_state_dict.keys())}")
                    # Devolver None para indicar que no se puede formar el índice completo
                    return None

                value = agent_state_dict[var_name]
                index = self._discretize_value(value, var_name)

                # Si la discretización falla (e.g., valor NaN, var_name no existe), devuelve None
                if index is None:
                    #logger.warning(f"Fallo al discretizar '{var_name}' (valor: {value}) para '{gain_variable}'.")
                    return None # No se puede formar el índice completo
                indices.append(index)

            return tuple(indices)

        except Exception as e: # Captura errores inesperados durante el proceso
            logger.error(f"Error inesperado obteniendo índices para '{gain_variable}': {e}", exc_info=True)
            return None


    # --- Implementación de Métodos Principales de RLAgent ---

    @property
    def epsilon(self) -> float: return self._epsilon

    @property
    def learning_rate(self) -> float: return self._learning_rate

    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """Selecciona acción epsilon-greedy para cada ganancia."""
        actions: Dict[str, int] = {}
        perform_exploration = np.random.rand() < self._epsilon

        for gain in self.gain_variables:
            action_index = 1 # Default: maintain (acción 1)
            if gain in self.q_tables_np: # Solo si la tabla existe para esta ganancia
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        if perform_exploration:
                            action_index = np.random.randint(self.num_actions)
                        else:
                            # Obtener Q-values para el estado actual
                            q_values_for_state = self.q_tables_np[gain][state_indices]
                            # Elegir la acción con el máximo Q-value (romper empates aleatoriamente)
                            max_q = np.nanmax(q_values_for_state)
                            if pd.isna(max_q): # Si todos son NaN, explorar
                                 action_index = np.random.randint(self.num_actions)
                            else:
                                 # Encontrar todos los índices con el valor máximo
                                 best_actions = np.where(np.isclose(q_values_for_state, max_q))[0]
                                 action_index = int(np.random.choice(best_actions))

                    except IndexError:
                        logger.error(f"IndexError en select_action '{gain}'. Índices: {state_indices}. Shape Q: {self.q_tables_np[gain].shape}. Usando acción 1 (default).")
                        action_index = 1
                    except Exception as e:
                        logger.error(f"Error inesperado en select_action '{gain}': {e}. Usando acción 1 (default).", exc_info=True)
                        action_index = 1
                # else: logger.debug(f"No se pudieron obtener índices para '{gain}' en select_action. Usando acción 1.") # Log si falla obtención de índice
            # else: logger.debug(f"No existe tabla Q para '{gain}'. Usando acción 1.") # Log si no hay tabla

            actions[gain] = action_index

        #logger.debug(f"Select Action -> Epsilon: {self._epsilon:.4f}, Explore: {perform_exploration}, Actions: {actions}")
        return actions


    def learn(self,
              current_agent_state_dict: Dict[str, Any], # S
              actions_dict: Dict[str, int],             # A = {kp: a_kp, ki: a_ki, kd: a_kd}
              reward_info: Union[float, Tuple[float, float], Dict[str, float]], # R_info (crudo)
              next_agent_state_dict: Dict[str, Any],    # S'
              controller: Controller,                   # Necesario para RewardStrategy
              done: bool):
        """Actualiza Q-tables usando la experiencia y la RewardStrategy."""

        # 1.6: Parsear reward_info (la lógica se mantiene, es robusta)
        interval_reward: float = np.nan; avg_w_stab: float = np.nan; reward_dict_echo: Optional[Dict[str, float]] = None
        if isinstance(reward_info, tuple) and len(reward_info) == 2:
            interval_reward, avg_w_stab = reward_info
        elif isinstance(reward_info, dict):
            reward_dict_echo = reward_info
            # R_real y w_stab podrían no estar disponibles o ser necesarios para Echo
            interval_reward = np.nan # Marcar como no directamente disponible
            avg_w_stab = np.nan
        elif isinstance(reward_info, (float, int)):
            interval_reward = float(reward_info)
            avg_w_stab = 1.0 # Asumir w_stab=1 si solo se pasa R_real
        else:
            logger.error(f"Learn: reward_info tipo inesperado: {type(reward_info)}. Saltando learn.")
            return

        # Validar valores parseados (reemplazar NaN/inf con defaults si es necesario para la strategy)
        # La strategy es responsable de manejar NaN si lo necesita de forma especial
        if pd.isna(interval_reward) or not np.isfinite(interval_reward): interval_reward = 0.0 # Default R_real a 0
        if pd.isna(avg_w_stab) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0 # Default w_stab a 1

        self._last_td_errors = {gain: np.nan for gain in self.gain_variables} # Resetear TD errors

        # 2. Iterar por cada ganancia para actualizar su Q-table
        for gain in self.gain_variables:
            if gain not in self.q_tables_np: continue # Saltar si no hay tabla Q

            try:
                # 3. Obtener índices discretos para S y S' (devuelven None si fallan)
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)

                # Fail-Fast si no se pueden obtener índices para la tabla actual
                if current_state_indices is None:
                    logger.warning(f"Learn: No se pudieron obtener índices S para '{gain}'. Saltando actualización.")
                    continue
                # Es aceptable que next_state_indices sea None si 'done' es True

                # 4. Obtener acción A_g tomada para esta ganancia
                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not (0 <= action_taken_idx < self.num_actions):
                    logger.warning(f"Learn: Índice de acción inválido ({action_taken_idx}) para '{gain}'. Saltando.")
                    continue

                # 5. Calcular R_learn usando la ESTRATEGIA INYECTADA
                # La estrategia recibe toda la info y decide qué usar.
                reward_for_q_update = self.reward_strategy.compute_reward_for_learning(
                    gain=gain, agent=self, controller=controller, # Pasar dependencias
                    current_agent_state_dict=current_agent_state_dict,
                    current_state_indices=current_state_indices,
                    actions_dict=actions_dict, action_taken_idx=action_taken_idx,
                    interval_reward=interval_reward, # Pasar R_real (o default 0)
                    avg_w_stab=avg_w_stab,           # Pasar w_stab (o default 1)
                    reward_dict=reward_dict_echo     # Pasar R_diff dict (o None)
                )

                # Validar R_learn devuelto (debe ser finito)
                if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
                    logger.warning(f"Learn: RewardStrategy devolvió R_learn inválido ({reward_for_q_update}) para '{gain}'. Usando 0.")
                    reward_for_q_update = 0.0

                # 6. Calcular TD Target y TD Error
                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]

                if done:
                    td_target = reward_for_q_update # No hay Q(S') si es terminal
                else:
                    # Necesitamos S' para calcular max Q(S', a')
                    if next_state_indices is None:
                         logger.warning(f"Learn: Episodio no 'done' pero no se pudieron obtener índices S' para '{gain}'. Usando Q(S')=0.")
                         max_next_q = 0.0
                    else:
                         next_q_values = self.q_tables_np[gain][next_state_indices]
                         # Usar nanmax para robustez ante acciones no exploradas en S'
                         max_next_q = np.nanmax(next_q_values)
                         # Si todos los Q(S',a') son NaN, tratar como 0
                         if pd.isna(max_next_q) or not np.isfinite(max_next_q):
                             max_next_q = 0.0
                    td_target = reward_for_q_update + self.discount_factor * max_next_q

                td_error = td_target - current_q
                self._last_td_errors[gain] = float(td_error) # Guardar TD error

                # 7. Actualizar Q-Table
                new_q_value = current_q + self._learning_rate * td_error
                # Actualizar solo si el nuevo valor es finito
                if pd.notna(new_q_value) and np.isfinite(new_q_value):
                    self.q_tables_np[gain][full_index_current] = new_q_value
                else:
                    logger.warning(f"Learn: Nuevo Q-value inválido ({new_q_value}) para '{gain}' en {full_index_current}. No se actualizó Q.")

                # 8. Actualizar Contador de Visitas N(s, a)
                self.visit_counts_np[gain][full_index_current] += 1

            except IndexError as e:
                # Error de índice probablemente en acceso a Q, N o B
                q_shape = self.q_tables_np.get(gain, np.array([])).shape
                logger.error(f"IndexError en learn '{gain}'. Índices S: {current_state_indices}, Acción: {action_taken_idx}. Q Shape: {q_shape}. Error: {e}", exc_info=True)
            except KeyError as e: # Error si la estrategia intenta acceder a algo no existente
                logger.error(f"KeyError durante learn '{gain}' (posiblemente en RewardStrategy): {e}.", exc_info=True)
            except Exception as e: # Capturar otros errores inesperados
                logger.error(f"Error inesperado durante learn para '{gain}': {e}.", exc_info=True)

    def reset_agent(self):
        """Actualiza epsilon y learning rate al final de un episodio."""
        if self.use_epsilon_decay:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        if self.use_learning_rate_decay:
            self._learning_rate = max(self._learning_rate_min, self._learning_rate * self._learning_rate_decay)
        self._last_td_errors = {gain: np.nan for gain in self.gain_variables}
        # logger.debug(f"Agent reset: Epsilon={self.epsilon:.4f}, LR={self.learning_rate:.4f}")

    def build_agent_state(self, raw_state_vector: Any, controller: Controller, state_config_for_build: Dict) -> Dict[str, Any]:
        """Construye el diccionario de estado del agente desde el estado crudo y el controlador."""
        # logger.debug(f"Building agent state from raw: {raw_state_vector}")
        agent_state: Dict[str, Any] = {}
        # 1.7: Validar input raw_state_vector y state_config
        if not isinstance(state_config_for_build, dict):
             logger.error("build_agent_state: state_config_for_build no es un diccionario.")
             return {} # Devolver dict vacío si la config es inválida

        # Mapeo de nombres estándar a índices en vector de estado del péndulo
        state_vector_map = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        try:
            # Obtener ganancias actuales del controlador
            current_gains = controller.get_params()
        except Exception as e:
            logger.error(f"Error obteniendo ganancias del controlador en build_agent_state: {e}")
            current_gains = {} # Continuar con dict vacío

        # Iterar sobre la configuración de estado proporcionada
        for var_name, config in state_config_for_build.items():
            if isinstance(config, dict) and config.get('enabled', False):
                value = np.nan # Default a NaN
                # Obtener valor desde el vector de estado si corresponde
                if var_name in state_vector_map:
                    idx = state_vector_map[var_name]
                    # Validar tipo y longitud del vector crudo
                    if isinstance(raw_state_vector, (list, tuple, np.ndarray)) and len(raw_state_vector) > idx:
                         val_raw = raw_state_vector[idx]
                         # Asignar solo si es finito
                         if isinstance(val_raw, (int, float)) and np.isfinite(val_raw):
                             value = float(val_raw)
                    # else: logger.warning(f"build_agent_state: Raw state inválido o corto para '{var_name}'.")

                # Obtener valor desde las ganancias del controlador si corresponde
                elif var_name in current_gains:
                    val_gain = current_gains.get(var_name) # Usar .get por seguridad
                    if isinstance(val_gain, (int, float)) and np.isfinite(val_gain):
                        value = float(val_gain)

                # Añadir al diccionario de estado del agente SOLO si se encontró un valor finito
                if not pd.isna(value): # pd.isna maneja NaN y None
                    agent_state[var_name] = value
                # else: logger.debug(f"build_agent_state: No se encontró valor válido para variable habilitada '{var_name}'.")

        #logger.debug(f"Agent state built: {agent_state}")
        return agent_state

    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """
        Prepara el estado interno (tablas Q, N, B) para guardado.
        Representa cada estado discreto usando los puntos representativos
        calculados con np.linspace(min, max, bins).
        """
        logger.info("Estructurando estado del agente para guardado")
        agent_state_save: Dict[str, Any] = {
            "q_tables": {}, "visit_counts": {}, "baseline_tables": {}
        }
        float_precision = 6 # Precisión para valores de estado representativos

        for gain, ordered_vars in self.ordered_state_vars_for_gain.items():
            #logger.debug(f"Estructurando datos para ganancia '{gain}' con estado {ordered_vars}...")
            q_list = []; visit_list = []; baseline_list = []

            if gain not in self.q_tables_np or gain not in self.visit_counts_np:
                 logger.warning(f"Faltan tablas Q o N para '{gain}' al guardar estado. Saltando.")
                 continue
            q_table = self.q_tables_np[gain]
            visit_table = self.visit_counts_np[gain]
            baseline_table = self.baseline_tables_np.get(gain)

            state_shape = q_table.shape[:-1]
            num_actions = q_table.shape[-1]

            # --- Calcular los puntos representativos para cada variable ---
            state_points_cache = {}
            valid_cache = True
            for var_name in ordered_vars:
                try:
                    cfg = self.state_config[var_name]
                    min_v, max_v, bins = cfg['min'], cfg['max'], cfg['bins']
                    if bins > 0:
                         # Calcular los 'bins' puntos incluyendo extremos
                         state_points_cache[var_name] = np.linspace(min_v, max_v, bins)
                    else:
                         raise ValueError(f"Bins inválidos ({bins}) para '{var_name}'")
                except Exception as e:
                    logger.error(f"Error calculando puntos representativos para '{var_name}' al guardar: {e}")
                    valid_cache = False; break
            if not valid_cache:
                 logger.warning(f"No se pudieron calcular puntos representativos para '{gain}'. Saltando guardado.")
                 continue

            # Iterar sobre todos los índices de estado posibles en la tabla
            total_states_processed = 0
            for state_indices in np.ndindex(state_shape):
                # --- Construir representación usando puntos representativos ---
                state_repr_dict = {}
                valid_repr = True
                try:
                    for i, var_name in enumerate(ordered_vars):
                        idx = state_indices[i] # Índice del punto (0 a bins-1)
                        # Obtener el punto representativo correspondiente al índice
                        repr_val = state_points_cache[var_name][idx]
                        state_repr_dict[var_name] = round(repr_val, float_precision)
                except IndexError as e:
                     logger.warning(f"Error generando representación de estado (punto linspace) para índices {state_indices} de '{gain}': {e}. Saltando este estado.")
                     valid_repr = False
                except Exception as e:
                     logger.error(f"Error inesperado generando repr. de estado para {state_indices}, gain '{gain}': {e}")
                     valid_repr = False

                if not valid_repr: continue # Saltar al siguiente estado

                # --- Guardar filas Q, N, B (sin cambios en esta parte) ---
                try:
                    q_row = state_repr_dict.copy()
                    q_values = q_table[state_indices]
                    for a in range(num_actions): q_row[f"action_{a}"] = float(q_values[a])
                    q_list.append(q_row)
                except IndexError: logger.error(...); continue

                try:
                    visit_row = state_repr_dict.copy()
                    visit_counts = visit_table[state_indices]
                    for a in range(num_actions): visit_row[f"action_{a}"] = int(visit_counts[a])
                    visit_list.append(visit_row)
                except IndexError: logger.error(...); continue

                if baseline_table is not None:
                    try:
                        baseline_row = state_repr_dict.copy()
                        baseline_row["baseline_value"] = float(baseline_table[state_indices])
                        baseline_list.append(baseline_row)
                    except IndexError: logger.error(...)

                total_states_processed += 1

            agent_state_save["q_tables"][gain] = q_list
            agent_state_save["visit_counts"][gain] = visit_list
            if baseline_list: agent_state_save["baseline_tables"][gain] = baseline_list
            #ogger.debug(f"Estructuración '{gain}' completa. {total_states_processed} estados procesados.")

        if not agent_state_save["q_tables"]:
             logger.warning("No se generaron datos para ninguna tabla Q al guardar el estado.")

        logger.info("Estructuración estado agente completa.")
        return agent_state_save

    # --- Métodos Helper para Logging (Interfaz) ---

    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Obtiene Q-values[acciones] para el estado dado para ganancias con tabla Q."""
        q_values: Dict[str, np.ndarray] = {}
        for gain in self.gain_variables:
            # Devolver array de NaNs si no se puede obtener Q-values
            q_vals_for_gain = np.full(self.num_actions, np.nan, dtype=np.float32)
            if gain in self.q_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        q_vals_for_gain = self.q_tables_np[gain][state_indices].astype(np.float32)
                    except IndexError: pass # Ignorar error de índice, devolver NaNs
                    except Exception as e: logger.warning(f"Error get_q_values '{gain}': {e}") # Loguear otros errores
            q_values[gain] = q_vals_for_gain
        return q_values

    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Obtiene N(s,a)[acciones] para el estado dado."""
        visit_counts: Dict[str, np.ndarray] = {}
        for gain in self.gain_variables:
            visits_for_gain = np.full(self.num_actions, -1, dtype=np.int32) # Default -1
            if gain in self.visit_counts_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        visits_for_gain = self.visit_counts_np[gain][state_indices].astype(np.int32)
                    except IndexError: pass
                    except Exception as e: logger.warning(f"Error get_visit_counts '{gain}': {e}")
            visit_counts[gain] = visits_for_gain
        return visit_counts

    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """Obtiene B(s) para el estado dado."""
        baselines: Dict[str, float] = {}
        for gain in self.gain_variables:
            baseline_val = np.nan # Default NaN
            if gain in self.baseline_tables_np: # Solo si existe la tabla
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        b_val = self.baseline_tables_np[gain][state_indices]
                        baseline_val = float(b_val) if pd.notna(b_val) and np.isfinite(b_val) else np.nan
                    except IndexError: pass
                    except Exception as e: logger.warning(f"Error get_baseline_value '{gain}': {e}")
            baselines[gain] = baseline_val
        return baselines

    def get_last_td_errors(self) -> Dict[str, float]:
        """Devuelve los TD errors del último paso de learn."""
        # Asegurar que los valores devueltos sean floats estándar o NaN
        return {k: (float(v) if pd.notna(v) and np.isfinite(v) else np.nan)
                for k, v in self._last_td_errors.items()}