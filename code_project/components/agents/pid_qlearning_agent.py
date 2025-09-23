# components/agents/pid_qlearning_agent.py
import numpy as np
import pandas as pd
import logging
from interfaces.rl_agent import RLAgent # Importar Interfaz
from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from interfaces.controller import Controller # Para type hint en build_agent_state
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List

# Importar Shadow strategy sólo para isinstance check (opcional pero claro)
from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PIDQLearningAgent(RLAgent): # Implementar Interfaz RLAgent
    """
    Agente Q-Learning para ajustar ganancias PID (Kp, Ki, Kd).
    Utiliza discretización del espacio de estados y tablas Q separadas por ganancia.
    La lógica de cálculo de recompensa para el aprendizaje se delega a una RewardStrategy inyectada.
    """
    def __init__(self,
                 # --- Dependencia Inyectada ---
                 reward_strategy: RewardStrategy,        # Estrategia de recompensa INYECTADA
                 # --- Configuración del Agente (desde config) ---
                 state_config: Dict[str, Dict[str, Any]], # Configuración de discretización del estado
                 num_actions: int,                       # Número de acciones (e.g., 3)
                 gain_step: Union[float, Dict[str, float]], # Tamaño del paso para ajustar ganancias
                 variable_step: bool,                   # Usar gain_step diferente por ganancia?
                 # --- Parámetros de Aprendizaje (desde config) ---
                 discount_factor: float = 0.98,          # Gamma
                 epsilon: float = 1.0,                   # Epsilon inicial
                 epsilon_min: float = 0.1,               # Epsilon mínimo
                 epsilon_decay: float = 0.99954,         # Decaimiento de epsilon por episodio
                 learning_rate: float = 1.0,             # Alpha inicial
                 learning_rate_min: float = 0.01,        # Alpha mínimo
                 learning_rate_decay: float = 0.999425,  # Decaimiento de alpha por episodio
                 use_epsilon_decay: bool = True,         # Habilitar decaimiento de epsilon?
                 use_learning_rate_decay: bool = True,   # Habilitar decaimiento de alpha?
                 # --- Inicialización de Tablas (desde config) ---
                 q_init_value: float = 0.0,              # Valor inicial para Q(s,a)
                 visit_init_value: int = 0,              # Valor inicial para N(s,a)
                 # --- Parámetros Opcionales (inyectados si la estrategia los necesita) ---
                 shadow_baseline_params: Optional[Dict] = None # e.g., {'baseline_init_value': 0.0}
                 ):
        """
        Inicializa el agente PID Q-Learning. Recibe RewardStrategy como dependencia.
        (Docstring sin cambios respecto a Paso 1)
        """
        logger.info("Inicializando PIDQLearningAgent...")
        # --- Almacenamiento de Dependencias y Parámetros ---
        if not isinstance(reward_strategy, RewardStrategy):
             raise TypeError("El argumento 'reward_strategy' debe ser una instancia de RewardStrategy.")
        self.reward_strategy = reward_strategy
        logger.info(f"Usando Reward Strategy: {type(self.reward_strategy).__name__}")

        try:
             self.state_config = self._validate_and_prepare_state_config(state_config)
        except ValueError as e:
             logger.error(f"Configuración de estado inválida: {e}", exc_info=True)
             raise

        self.num_actions = num_actions
        self.q_init_value = q_init_value
        self.visit_init_value = visit_init_value

        self.gain_step = gain_step
        self.variable_step = variable_step
        if variable_step:
            if not isinstance(gain_step, dict): raise ValueError("variable_step=True, pero gain_step no es dict.")
        elif not isinstance(gain_step, (float, int)): raise ValueError("variable_step=False, pero gain_step no es numérico.")

        self.discount_factor = discount_factor
        self._initial_epsilon, self._initial_learning_rate = epsilon, learning_rate
        self._epsilon, self._epsilon_min, self._epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self._learning_rate, self._learning_rate_min, self._learning_rate_decay = learning_rate, learning_rate_min, learning_rate_decay
        self.use_epsilon_decay, self.use_learning_rate_decay = use_epsilon_decay, use_learning_rate_decay

        # --- Configuración Baseline (Shadow Mode) ---
        self.is_shadow_mode = isinstance(self.reward_strategy, ShadowBaselineRewardStrategy)
        self.baseline_init_value = 0.0
        if self.is_shadow_mode:
             if shadow_baseline_params and isinstance(shadow_baseline_params, dict):
                  self.baseline_init_value = shadow_baseline_params.get('baseline_init_value', 0.0)
                  logger.info(f"Shadow Baseline detectado. B(s) init: {self.baseline_init_value}")
             else: logger.warning("Shadow Baseline detectado, pero 'shadow_baseline_params' no válido. Usando B(s)=0.0.")

        # --- Inicialización de Estructuras Internas ---
        self.ordered_state_vars_for_gain: Dict[str, List[str]] = {}
        self.gain_variables = ['kp', 'ki', 'kd']
        self.q_tables_np: Dict[str, np.ndarray] = {}
        self.visit_counts_np: Dict[str, np.ndarray] = {}
        self.baseline_tables_np: Dict[str, np.ndarray] = {}
        self._last_td_errors: Dict[str, float] = {gain: np.nan for gain in self.gain_variables}

        # --- Crear e Inicializar Tablas NumPy ---
        for gain in self.gain_variables:
            ordered_vars_list = self._get_ordered_vars_for_gain(gain)
            if not ordered_vars_list:
                 logger.info(f"No hay variables de estado habilitadas para ganancia '{gain}'. No se crearán tablas.")
                 continue
            logger.info(f"Inicializando tablas NumPy para ganancia '{gain}'...")
            self.ordered_state_vars_for_gain[gain] = ordered_vars_list
            try:
                state_dims = [self.state_config[var]['bins'] for var in ordered_vars_list]
                if not state_dims: raise ValueError("No se encontraron dimensiones de estado.")
                q_visit_shape = tuple(state_dims + [self.num_actions])
                baseline_shape = tuple(state_dims)
                logger.debug(f"  - Orden vars '{gain}': {ordered_vars_list}")
                logger.debug(f"  - Shape Q/Visit '{gain}': {q_visit_shape}, Shape Baseline '{gain}': {baseline_shape}")
                self.q_tables_np[gain] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
                self.visit_counts_np[gain] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)
                self.baseline_tables_np[gain] = np.full(baseline_shape, self.baseline_init_value, dtype=np.float32)
            except KeyError as e:
                logger.error(f"Error init tablas '{gain}': Falta 'bins' para variable {e}. Saltando."); self._cleanup_failed_gain_init(gain)
            except ValueError as e:
                 logger.error(f"Error init tablas '{gain}': {e}. Saltando."); self._cleanup_failed_gain_init(gain)
            except Exception as e:
                 logger.error(f"Error inesperado init tablas '{gain}': {e}", exc_info=True); self._cleanup_failed_gain_init(gain)

        logger.info("PIDQLearningAgent inicializado exitosamente.")
        if not self.q_tables_np: logger.warning("¡Ninguna tabla Q fue inicializada! Verificar 'state_config'.")

    def _cleanup_failed_gain_init(self, gain: str):
        """Helper para limpiar tablas si la inicialización falla."""
        if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
        if gain in self.q_tables_np: del self.q_tables_np[gain]
        if gain in self.visit_counts_np: del self.visit_counts_np[gain]
        if gain in self.baseline_tables_np: del self.baseline_tables_np[gain]

    # --- Implementación de Propiedades de la Interfaz ---
    @property
    def epsilon(self) -> float: return self._epsilon
    @property
    def learning_rate(self) -> float: return self._learning_rate

    # --- Métodos Helper Internos (Discretización, Validación) ---
    def _validate_and_prepare_state_config(self, config: Dict) -> Dict:
        """Valida la estructura y contenido de state_config."""
        logger.debug("Validando state_config...")
        validated_config = {}
        required_keys_enabled = ['min', 'max', 'bins']
        if not isinstance(config, dict): raise ValueError("state_config debe ser dict.")
        for var, cfg in config.items():
            if not isinstance(cfg, dict): raise ValueError(f"Config para '{var}' debe ser dict.")
            if 'enabled' not in cfg: raise ValueError(f"Config para '{var}' falta 'enabled'.")
            validated_config[var] = cfg.copy()
            if cfg['enabled']:
                missing_keys = [key for key in required_keys_enabled if key not in cfg]
                if missing_keys: raise ValueError(f"Config habilitada '{var}' faltan claves: {missing_keys}.")
                if not all(isinstance(cfg[k], (int, float)) for k in ['min', 'max']): raise ValueError(f"Config '{var}': 'min'/'max' deben ser numéricos.")
                if cfg['min'] >= cfg['max']: raise ValueError(f"Config '{var}': 'min' ({cfg['min']}) debe ser < 'max' ({cfg['max']}).")
                if not isinstance(cfg['bins'], int) or cfg['bins'] <= 0: raise ValueError(f"Config '{var}': 'bins' ({cfg['bins']}) debe ser entero positivo.")
                logger.debug(f" - Var '{var}': Habilitada, Min={cfg['min']}, Max={cfg['max']}, Bins={cfg['bins']}")
        logger.debug("state_config validado.")
        return validated_config

    def _get_ordered_vars_for_gain(self, gain: str) -> List[str]:
        """Obtiene la lista ordenada de variables de estado *habilitadas* relevantes para una ganancia."""
        ordered_vars = OrderedDict()
        # Añadir variables base habilitadas
        for var, cfg in self.state_config.items():
            if cfg.get('enabled', False) and var not in self.gain_variables:
                ordered_vars[var] = True
        # Añadir la propia ganancia si está habilitada como estado
        if gain in self.state_config and self.state_config[gain].get('enabled', False):
            ordered_vars[gain] = True
        return list(ordered_vars.keys())

    def _discretize_value(self, value: float, var_name: str) -> int:
        """Discretiza un valor continuo según la configuración de la variable."""
        if var_name not in self.state_config or not self.state_config[var_name].get('enabled'):
             return 0 # Índice por defecto si no hay config o no está habilitada
        config = self.state_config[var_name]; bins = config['bins']; min_val = config['min']; max_val = config['max']
        if bins <= 0 or max_val <= min_val: return 0
        # Clip value
        clipped_value = np.clip(value, min_val, max_val)
        # Handle edge case where bins=1
        if bins == 1: return 0
        # Calculate bin size, avoid division by zero if max_val == min_val (caught earlier)
        bin_size = (max_val - min_val) / bins
        if bin_size < 1e-9: return 0 # Avoid division by zero/precision issues
        # Handle max edge case explicitly
        if clipped_value >= max_val: return bins - 1
        # Calculate index
        index = int(np.floor((clipped_value - min_val) / bin_size))
        # Final clip for safety
        return int(np.clip(index, 0, bins - 1))

    def get_discrete_state_indices_tuple(self, agent_state_dict: Optional[Dict[str, Any]], gain_variable: str) -> Optional[tuple]:
         """Convierte el diccionario de estado del agente en una tupla de índices discretos para una ganancia.
            Devuelve None si el estado es inválido (falta clave o valor es NaN).
         """
         if agent_state_dict is None:
              # Mantener este error ya que recibir None es inesperado
              logger.error(f"get_discrete_state_indices_tuple recibió agent_state_dict=None para ganancia '{gain_variable}'.")
              return None
         if gain_variable not in self.ordered_state_vars_for_gain:
             return None # No hay vars para esta ganancia

         ordered_vars = self.ordered_state_vars_for_gain[gain_variable]
         indices = []
         try:
             for var_name in ordered_vars:
                 # --- CORRECCIÓN: No levantar KeyError si falta la clave ---
                 if var_name not in agent_state_dict:
                     # Si el dict no está vacío, es un problema real. Si está vacío (como al inicio), es esperado.
                     if agent_state_dict: # Solo loguear advertencia si el dict NO estaba vacío
                         logger.warning(f"Variable habilitada '{var_name}' no encontrada en agent_state_dict no vacío (claves: {list(agent_state_dict.keys())}) al obtener índices para ganancia '{gain_variable}'.")
                     # else: # Es normal al inicio si el dict es vacío, no loguear nada.
                     #    pass
                     return None # Devolver None si falta una clave requerida
                 # --- FIN CORRECCIÓN ---

                 value = agent_state_dict[var_name] # Acceder ahora que sabemos que existe

                 # --- Chequeo de NaN (mantenido de la corrección anterior) ---
                 if pd.isna(value):
                     # Loguear esto sí puede ser útil, indica un estado inválido
                     logger.warning(f"Valor NaN encontrado para variable habilitada '{var_name}' al obtener índices para ganancia '{gain_variable}'. Estado inválido para indexar tablas.")
                     return None # Si cualquier componente es NaN, el índice de estado es inválido
                 # --- FIN Chequeo de NaN ---

                 # Si no es NaN, debe ser numérico. Discretizar.
                 index = self._discretize_value(value, var_name)
                 indices.append(index)

             return tuple(indices) # Devolver tupla solo si todos los componentes son válidos

         # except KeyError: # Ya no debería ocurrir por falta de clave
             # pass # Eliminar este bloque o mantenerlo por si acaso? Mejor quitarlo.
         except Exception as e: # Captura errores de _discretize_value si value no fuera float, u otros
             logger.error(f"Error inesperado obteniendo índices para '{gain_variable}': {e}", exc_info=True)
             return None

    # --- Implementación de Métodos Principales de la Interfaz RLAgent ---

    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """Selecciona una acción para cada ganancia (Kp, Ki, Kd) usando epsilon-greedy."""
        actions: Dict[str, int] = {}
        perform_exploration = np.random.rand() < self._epsilon
        for gain in self.gain_variables:
            action_index = 1 # Default: maintain
            if gain in self.q_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        if perform_exploration:
                            action_index = np.random.randint(self.num_actions)
                        else:
                            q_values_for_state = self.q_tables_np[gain][state_indices]
                            action_index = int(np.argmax(q_values_for_state))
                    except IndexError: logger.error(f"IndexError select_action '{gain}'. Índices: {state_indices}. Shape Q: {self.q_tables_np[gain].shape}. Usando acción 1."); action_index = 1
                    except Exception as e: logger.error(f"Error select_action '{gain}': {e}. Usando acción 1.", exc_info=True); action_index = 1
                else: logger.warning(f"No se pudieron obtener índices para '{gain}' en select_action. Usando acción 1."); action_index = 1
            actions[gain] = action_index

        #logger.debug(f"PIDQLearningAgent -> select_action -> Acciones: {actions} (Explorando: {perform_exploration}, Epsilon: {self._epsilon:.3f})")

        return actions

    def learn(self,
              current_agent_state_dict: Dict[str, Any], # S
              actions_dict: Dict[str, int],             # A
              reward_info: Union[float, Tuple[float, float], Dict[str, float]], # R (info pasada por SimManager)
              next_agent_state_dict: Dict[str, Any],    # S'
              controller: Controller,
              done: bool):                               # Done flag
        """Actualiza las Q-tables usando la experiencia (S, A, R, S') y la RewardStrategy inyectada."""

        # 1. Parsear reward_info para obtener R_real y w_stab (Echo pasa dict)
        interval_reward: float = 0.0; avg_w_stab: float = 1.0; reward_dict_echo: Optional[Dict[str, float]] = None
        if isinstance(reward_info, tuple) and len(reward_info) == 2: # Formato para Shadow/Global desde SimMan
            interval_reward, avg_w_stab = reward_info
            if not isinstance(interval_reward, (float, int)) or not np.isfinite(interval_reward): interval_reward = 0.0
            if not isinstance(avg_w_stab, (float, int)) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0 # Default w_stab
        elif isinstance(reward_info, dict): # Formato para Echo desde SimMan
            reward_dict_echo = reward_info # El dict contiene R_diff precalculados
            # Estimar R_real si es necesario (o la strategy puede no necesitarlo)
            # Por ahora, no lo necesitamos explícitamente si la estrategia es Echo
            interval_reward = 0.0 # Placeholder, Echo usa R_diff del dict
        elif isinstance(reward_info, (float, int)): # Podría ser solo R_real si w_stab no aplica
             interval_reward = float(reward_info) if np.isfinite(reward_info) else 0.0
             avg_w_stab = 1.0 # Asumir w_stab no aplica
        else:
            logger.error(f"Learn recibió reward_info tipo inesperado: {type(reward_info)}. Saltando learn.")
            return

        self._last_td_errors = {gain: np.nan for gain in self.gain_variables} # Resetear

        # 2. Iterar por cada ganancia controlada
        for gain in self.gain_variables:
            if gain not in self.q_tables_np: continue # Saltar si no hay tabla Q para esta ganancia

            try:
                # 3. Obtener índices discretos para S y S'
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)
                if current_state_indices is None or next_state_indices is None:
                    logger.warning(f"No se pudieron obtener índices S o S' para '{gain}' en learn. Saltando actualización.")
                    continue

                # 4. Obtener acción A_g tomada para esta ganancia
                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not (0 <= action_taken_idx < self.num_actions):
                    logger.warning(f"Índice de acción inválido ({action_taken_idx}) para '{gain}' en learn. Saltando.")
                    continue

                # 5. Calcular R_learn usando la ESTRATEGIA INYECTADA
                # Pasamos toda la info necesaria, la estrategia decide qué usar.
                reward_for_q_update = self.reward_strategy.compute_reward_for_learning(
                    gain=gain, agent=self,
                    current_agent_state_dict=current_agent_state_dict, # Podría ser útil para estrategias futuras
                    current_state_indices=current_state_indices,
                    actions_dict=actions_dict,
                    action_taken_idx=action_taken_idx,
                    interval_reward=interval_reward, # R_real
                    avg_w_stab=avg_w_stab,           # w_stab promedio
                    reward_dict=reward_dict_echo,     # R_diff dict (solo para Echo)
                    controller=controller
                )

                # Validar R_learn devuelto por la estrategia
                if not isinstance(reward_for_q_update, (float, int)) or not np.isfinite(reward_for_q_update):
                     logger.warning(f"RewardStrategy devolvió R_learn inválido ({reward_for_q_update}) para '{gain}'. Usando 0.")
                     reward_for_q_update = 0.0

                # 6. Calcular TD Target y TD Error
                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]

                if done:
                    td_target = reward_for_q_update
                else:
                    # Usar nanmax para manejar posibles NaNs en Q-values si no se han visitado acciones
                    next_q_values = self.q_tables_np[gain][next_state_indices]
                    max_next_q = np.nanmax(next_q_values)
                    # Si todas las acciones futuras tienen Q=NaN, max_next_q será NaN. Tratar como 0.
                    if pd.isna(max_next_q) or not np.isfinite(max_next_q): max_next_q = 0.0
                    td_target = reward_for_q_update + self.discount_factor * max_next_q

                td_error = td_target - current_q
                self._last_td_errors[gain] = float(td_error) # Guardar TD error

                # 7. Actualizar Q-Table
                new_q_value = current_q + self._learning_rate * td_error
                if pd.notna(new_q_value) and np.isfinite(new_q_value):
                     self.q_tables_np[gain][full_index_current] = new_q_value
                # else: logger.warning(f"Nuevo Q-value inválido ({new_q_value}) para '{gain}'. No se actualizó.")

                # 8. Actualizar Contador de Visitas N(s, a)
                self.visit_counts_np[gain][full_index_current] += 1

            except IndexError as e:
                q_shape = self.q_tables_np.get(gain, np.array([])).shape
                b_shape = self.baseline_tables_np.get(gain, np.array([])).shape
                logger.error(f"IndexError en learn '{gain}'. Índices S: {current_state_indices}, Acción: {action_taken_idx}. Q Shape: {q_shape}, B Shape: {b_shape}. Error: {e}")
            except KeyError as e: logger.error(f"KeyError durante learn '{gain}': {e}. Verificar dicts o lógica estrategia.")
            except Exception as e: logger.error(f"Error inesperado durante learn '{gain}': {e}.", exc_info=True)

    def reset_agent(self):
        """Actualiza epsilon y learning rate al final de un episodio."""
        if self.use_epsilon_decay:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        if self.use_learning_rate_decay:
            self._learning_rate = max(self._learning_rate_min, self._learning_rate * self._learning_rate_decay)
        self._last_td_errors = {gain: np.nan for gain in self.gain_variables}
        # logger.debug(f"Agent reset: Epsilon={self.epsilon:.4f}, LR={self.learning_rate:.4f}")

    def build_agent_state(self, raw_state_vector: Any, controller: Controller, state_config_for_build: Dict) -> Dict[str, Any]:
        """Construye el diccionario de estado del agente (DEBUGGING AGRESIVO)."""
        logger.debug(f"--- ENTERING build_agent_state ---")
        logger.debug(f"Received state_config keys: {list(state_config_for_build.keys())}")
        enabled_vars_in_config = {k:v for k, v in state_config_for_build.items() if isinstance(v, dict) and v.get('enabled')}
        logger.debug(f"Enabled vars in received config: {list(enabled_vars_in_config.keys())}")

        agent_state: Dict[str, Any] = {}

        state_vector_map = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        try:
            current_gains = controller.get_params()
            logger.debug(f"Controller get_params() returned: {current_gains}")
        except Exception as e:
            logger.error(f"Error get_params: {e}."); current_gains = {} # Devolver vacío si falla

        # --- Bucle para rellenar valores (si existen) ---
        for var_name, config in state_config_for_build.items():
            if config.get('enabled', False): # Solo procesar habilitadas
                value = np.nan # Default
                if var_name in state_vector_map:
                    idx = state_vector_map[var_name]
                    try:
                        if isinstance(raw_state_vector, (list, np.ndarray)) and len(raw_state_vector) > idx:
                            val_raw = raw_state_vector[idx]
                            if isinstance(val_raw, (int, float)) and np.isfinite(val_raw): 
                                value = float(val_raw)
                    except Exception: pass # Ignorar errores de estado crudo por ahora
                elif var_name in current_gains:
                    val_gain = current_gains.get(var_name) # Usar .get por si acaso
                    if isinstance(val_gain, (int, float)) and np.isfinite(val_gain): value = float(val_gain)

                # Sobrescribir el NaN inicial solo si se encontró un valor válido
                if not pd.isna(value):
                    agent_state[var_name] = value
                # else: logger.debug(f"Value for '{var_name}' remains NaN.")

        logger.debug(f"--- EXITING build_agent_state --- Returning dict with keys: {list(agent_state.keys())}")
        return agent_state

    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """Prepara el estado interno del agente para guardado."""
        structured_q_tables = {}; structured_visit_counts = {}; structured_baseline_tables = {}
        logger.info("Estructurando estado del agente para guardado (formato Pandas)...")
        float_precision_for_keys = 6
        processed_gains = 0

        for gain in self.gain_variables:
            if gain in self.q_tables_np and gain in self.ordered_state_vars_for_gain:
                logger.debug(f"Estructurando datos para ganancia '{gain}'...")
                q_table_list = []; visit_count_list = []; baseline_list = []
                ordered_vars = self.ordered_state_vars_for_gain[gain]
                np_q_table = self.q_tables_np[gain]; np_visits = self.visit_counts_np[gain]; np_baseline = self.baseline_tables_np.get(gain)
                if np_q_table is None or np_visits is None: logger.warning(f"Tabla Q/Visitas es None para '{gain}'. Saltando."); continue

                state_shape = np_q_table.shape[:-1]; action_dim = np_q_table.shape[-1]
                discrete_points_cache = {}; valid_cache = True
                for var_name in ordered_vars:
                     try:
                          config = self.state_config[var_name]; min_val, max_val, bins = config['min'], config['max'], config['bins']
                          if bins > 1: discrete_points_cache[var_name] = np.linspace(min_val, max_val, bins)
                          elif bins == 1: discrete_points_cache[var_name] = np.array([(min_val + max_val) / 2.0])
                          else: discrete_points_cache[var_name] = np.array([min_val]) # Bins <= 0
                     except KeyError as e: logger.error(f"Falta config ('{e}') para '{var_name}' precalculando puntos. Saltando '{gain}'."); valid_cache = False; break
                if not valid_cache: continue

                total_states_in_table = np.prod(state_shape)
                for state_indices_tuple in np.ndindex(state_shape):
                    state_repr_dict = {}
                    try:
                        for i, var_name in enumerate(ordered_vars):
                            current_index = state_indices_tuple[i]
                            if current_index >= len(discrete_points_cache[var_name]): raise IndexError(f"Índice {current_index} fuera rango pts '{var_name}' (tamaño {len(discrete_points_cache[var_name])}).")
                            state_repr_dict[var_name] = round(discrete_points_cache[var_name][current_index], float_precision_for_keys)
                    except (IndexError, KeyError, Exception) as e: logger.error(f"Error valor representativo estado {state_indices_tuple}, var '{var_name}': {e}. Saltando estado."); continue

                    try: # Q-Table row
                         q_values = np_q_table[state_indices_tuple]; q_row = state_repr_dict.copy()
                         for action_idx in range(action_dim): q_row[f"action_{action_idx}"] = float(q_values[action_idx])
                         q_table_list.append(q_row)
                    except IndexError: logger.error(f"IndexError fila Q estado {state_indices_tuple}. Saltando Q."); continue

                    try: # Visit Counts row
                         visit_counts = np_visits[state_indices_tuple]; visit_row = state_repr_dict.copy()
                         for action_idx in range(action_dim): visit_row[f"action_{action_idx}"] = int(visit_counts[action_idx])
                         visit_count_list.append(visit_row)
                    except IndexError: logger.error(f"IndexError fila Visitas estado {state_indices_tuple}. Saltando Visitas.")

                    if np_baseline is not None: # Baseline row
                        try:
                             baseline_row = state_repr_dict.copy(); baseline_row['baseline_value'] = float(np_baseline[state_indices_tuple])
                             baseline_list.append(baseline_row)
                        except IndexError: logger.error(f"IndexError fila Baseline estado {state_indices_tuple}. Saltando Baseline.")

                structured_q_tables[gain] = q_table_list; structured_visit_counts[gain] = visit_count_list
                if np_baseline is not None: structured_baseline_tables[gain] = baseline_list
                logger.debug(f"Estructuración '{gain}' completa. {len(q_table_list)} estados.")
                processed_gains += 1
            elif gain in self.gain_variables: logger.debug(f"Saltando estructuración '{gain}' (tablas no inicializadas/sin vars).")

        if processed_gains == 0: logger.warning("No se procesó ninguna ganancia al estructurar estado.")
        logger.info("Estructuración estado agente completa.")
        return {"q_tables": structured_q_tables, "visit_counts": structured_visit_counts, "baseline_tables": structured_baseline_tables}

    # --- Implementación Métodos Helper para Logging ---
    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
         """Obtiene Q-values[acciones] para el estado dado para ganancias con tabla Q."""
         q_values: Dict[str, np.ndarray] = {}
         for gain in self.gain_variables:
             q_vals_for_gain = np.full(self.num_actions, np.nan, dtype=np.float32)
             if gain in self.q_tables_np:
                 state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                 if state_indices is not None:
                     try: q_vals_for_gain = self.q_tables_np[gain][state_indices].astype(np.float32)
                     except IndexError: pass # logger.warning(f"IndexError log Q ({gain}, {state_indices}).")
                     except Exception: pass # logger.warning(f"Error log Q ({gain}): {e}")
             q_values[gain] = q_vals_for_gain
         return q_values

    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Obtiene N(s,a)[acciones] para el estado dado para ganancias con tabla de visitas."""
        visit_counts: Dict[str, np.ndarray] = {}
        for gain in self.gain_variables:
            visits_for_gain = np.full(self.num_actions, -1, dtype=np.int32) # Default -1 para indicar error/no tabla
            if gain in self.visit_counts_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try: visits_for_gain = self.visit_counts_np[gain][state_indices].astype(np.int32)
                    except IndexError: pass # logger.warning(f"IndexError log Visits ({gain}, {state_indices}).")
                    except Exception: pass # logger.warning(f"Error log Visits ({gain}): {e}")
            visit_counts[gain] = visits_for_gain
        return visit_counts

    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """Obtiene B(s) para el estado dado para ganancias con tabla Baseline."""
        baselines: Dict[str, float] = {}
        for gain in self.gain_variables:
            baseline_val = np.nan
            if gain in self.baseline_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        b_val = self.baseline_tables_np[gain][state_indices]
                        baseline_val = float(b_val) if pd.notna(b_val) and np.isfinite(b_val) else np.nan
                    except IndexError: pass # logger.warning(f"IndexError log Baseline ({gain}, {state_indices}).")
                    except Exception: pass # logger.warning(f"Error log Baseline ({gain}): {e}")
            baselines[gain] = baseline_val
        return baselines

    def get_last_td_errors(self) -> Dict[str, float]:
         """Devuelve una copia del diccionario de TD errors del último paso de learn."""
         return {k: (float(v) if pd.notna(v) and np.isfinite(v) else np.nan)
                 for k, v in self._last_td_errors.items()}