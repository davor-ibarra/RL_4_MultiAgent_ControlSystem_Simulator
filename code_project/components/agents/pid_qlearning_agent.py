import numpy as np
import pandas as pd
import logging
from interfaces.rl_agent import RLAgent # Importar Interfaz
from interfaces.reward_strategy import RewardStrategy # Importar Interfaz
from interfaces.controller import Controller # Para type hint en build_agent_state
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List

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

        Args:
            reward_strategy: La instancia de RewardStrategy a usar para calcular R_learn.
            state_config: Configuración para la discretización de cada variable de estado.
            num_actions: Número de acciones discretas por ganancia.
            gain_step: Magnitud del cambio de ganancia por acción (float global o dict por ganancia).
            variable_step: Indica si gain_step es un diccionario por ganancia.
            discount_factor: Factor de descuento gamma.
            epsilon: Tasa de exploración inicial.
            epsilon_min: Tasa de exploración mínima.
            epsilon_decay: Factor de decaimiento de epsilon.
            learning_rate: Tasa de aprendizaje alfa inicial.
            learning_rate_min: Tasa de aprendizaje mínima.
            learning_rate_decay: Factor de decaimiento de alfa.
            use_epsilon_decay: Activar/desactivar decaimiento de epsilon.
            use_learning_rate_decay: Activar/desactivar decaimiento de alfa.
            q_init_value: Valor con el que se inicializan las entradas de las Q-tables.
            visit_init_value: Valor con el que se inicializan los contadores de visita.
            shadow_baseline_params: Parámetros adicionales para Shadow Baseline (e.g., baseline_init_value).
        """
        logger.info("Inicializando PIDQLearningAgent...")
        # --- Almacenamiento de Dependencias y Parámetros ---
        if not isinstance(reward_strategy, RewardStrategy):
             # Esta validación es útil aquí aunque AgentFactory también valida
             raise TypeError("El argumento 'reward_strategy' debe ser una instancia de RewardStrategy.")
        self.reward_strategy = reward_strategy
        logger.info(f"Usando Reward Strategy: {type(self.reward_strategy).__name__}")

        # Validar y almacenar configuración de estado
        try:
             self.state_config = self._validate_and_prepare_state_config(state_config)
        except ValueError as e:
             logger.error(f"Configuración de estado inválida: {e}", exc_info=True)
             raise # Relanzar para detener la creación

        self.num_actions = num_actions
        self.q_init_value = q_init_value
        self.visit_init_value = visit_init_value

        # Validar y almacenar gain_step/variable_step
        self.gain_step = gain_step
        self.variable_step = variable_step
        if variable_step:
            if not isinstance(gain_step, dict):
                raise ValueError("Se especificó variable_step=True, pero gain_step no es un diccionario.")
            # Opcional: Validar que las claves de gain_step sean 'kp', 'ki', 'kd'?
        elif not isinstance(gain_step, (float, int)):
            raise ValueError("Se especificó variable_step=False, pero gain_step no es numérico.")

        # Almacenar parámetros de aprendizaje
        self.discount_factor = discount_factor
        self._initial_epsilon = epsilon # Guardar valor inicial
        self._initial_learning_rate = learning_rate # Guardar valor inicial
        self._epsilon, self._epsilon_min, self._epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self._learning_rate, self._learning_rate_min, self._learning_rate_decay = learning_rate, learning_rate_min, learning_rate_decay
        self.use_epsilon_decay, self.use_learning_rate_decay = use_epsilon_decay, use_learning_rate_decay

        # --- Configuración específica de Baseline (Shadow Mode) ---
        # Determinar si la estrategia inyectada es Shadow Baseline
        self.is_shadow_mode = isinstance(self.reward_strategy, ShadowBaselineRewardStrategy)

        self.baseline_init_value = 0.0
        if self.is_shadow_mode:
             if shadow_baseline_params and isinstance(shadow_baseline_params, dict):
                  self.baseline_init_value = shadow_baseline_params.get('baseline_init_value', 0.0)
                  logger.info(f"Shadow Baseline detectado. B(s) init: {self.baseline_init_value}")
             else:
                  logger.warning("Shadow Baseline detectado, pero 'shadow_baseline_params' no válido. Usando B(s)=0.0.")

        # --- Inicialización de Estructuras Internas ---
        self.ordered_state_vars_for_gain: Dict[str, List[str]] = {}
        self.gain_variables = ['kp', 'ki', 'kd'] # Ganancias controladas

        # Tablas NumPy para eficiencia
        self.q_tables_np: Dict[str, np.ndarray] = {}
        self.visit_counts_np: Dict[str, np.ndarray] = {}
        self.baseline_tables_np: Dict[str, np.ndarray] = {} # Siempre inicializar

        # Últimos TD errors (para logging)
        self._last_td_errors: Dict[str, float] = {gain: np.nan for gain in self.gain_variables}

        # --- Crear e Inicializar Tablas NumPy ---
        # (Lógica mantenida como estaba, ya que es interna al agente)
        for gain in self.gain_variables:
            ordered_vars_list = self._get_ordered_vars_for_gain(gain)
            if not ordered_vars_list:
                 logger.info(f"No hay variables de estado habilitadas definidas para la ganancia '{gain}'. No se crearán tablas.")
                 continue

            logger.info(f"Inicializando tablas NumPy para ganancia '{gain}'...")
            self.ordered_state_vars_for_gain[gain] = ordered_vars_list
            try:
                state_dims = [self.state_config[var]['bins'] for var in ordered_vars_list]
                if not state_dims: raise ValueError("No se encontraron dimensiones de estado para crear las tablas.")

                q_visit_shape = tuple(state_dims + [self.num_actions])
                baseline_shape = tuple(state_dims)
                logger.debug(f"  - Orden vars para '{gain}': {ordered_vars_list}")
                logger.debug(f"  - Dimensiones estado: {state_dims}")
                logger.debug(f"  - Shape Q/Visit para '{gain}': {q_visit_shape}")
                logger.debug(f"  - Shape Baseline para '{gain}': {baseline_shape}")

                self.q_tables_np[gain] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
                self.visit_counts_np[gain] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)
                # Inicializar Baseline table siempre
                self.baseline_tables_np[gain] = np.full(baseline_shape, self.baseline_init_value, dtype=np.float32)
                logger.debug(f"  - Tablas NumPy creadas para '{gain}'.")

            except KeyError as e:
                logger.error(f"Error obteniendo 'bins' para variable {e} (ganancia '{gain}'). Saltando inicialización.")
                if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
                # Limpiar otras tablas si falló
                if gain in self.q_tables_np: del self.q_tables_np[gain]
                if gain in self.visit_counts_np: del self.visit_counts_np[gain]
                if gain in self.baseline_tables_np: del self.baseline_tables_np[gain]
            except ValueError as e:
                 logger.error(f"Error en dimensiones/shape al crear tablas para ganancia '{gain}': {e}. Saltando inicialización.")
                 if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]

        logger.info("PIDQLearningAgent inicializado exitosamente.")
        if not self.q_tables_np:
             logger.warning("¡Ninguna tabla Q fue inicializada! El agente no aprenderá. Verificar 'state_config'.")

    # --- Implementación de Propiedades de la Interfaz ---
    @property
    def epsilon(self) -> float:
        """Current exploration rate (epsilon)."""
        return self._epsilon

    @property
    def learning_rate(self) -> float:
        """Current learning rate (alpha)."""
        return self._learning_rate

    # --- Métodos Helper Internos (Discretización, Validación) ---
    # (Mantenidos como estaban, son lógica interna del agente)
    def _validate_and_prepare_state_config(self, config: Dict) -> Dict:
        """Valida la estructura y contenido de state_config."""
        # ... (código sin cambios) ...
        logger.debug("Validando state_config...")
        validated_config = {}
        required_keys = ['enabled', 'min', 'max', 'bins']
        if not isinstance(config, dict):
             raise ValueError("state_config debe ser un diccionario.")

        for var, cfg in config.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Configuración de estado para '{var}' debe ser un diccionario.")
            if 'enabled' not in cfg:
                # Permitir omitir 'enabled' si no está, asumiendo False? No, mejor requerirlo.
                raise ValueError(f"Configuración de estado para '{var}' falta la clave 'enabled'.")

            validated_config[var] = cfg.copy() # Guardar copia

            if cfg['enabled']:
                missing_keys = [key for key in required_keys if key not in cfg]
                if missing_keys:
                    raise ValueError(f"Configuración de estado habilitada para '{var}' faltan claves: {missing_keys}.")
                if not isinstance(cfg['min'], (int, float)) or not isinstance(cfg['max'], (int, float)):
                    raise ValueError(f"Configuración '{var}': 'min' y 'max' deben ser numéricos.")
                if cfg['min'] >= cfg['max']:
                    raise ValueError(f"Configuración '{var}': 'min' ({cfg['min']}) debe ser < 'max' ({cfg['max']}).")
                if not isinstance(cfg['bins'], int) or cfg['bins'] <= 0:
                    raise ValueError(f"Configuración '{var}': 'bins' ({cfg['bins']}) debe ser entero positivo.")
                logger.debug(f" - Variable '{var}': Habilitada, Min={cfg['min']}, Max={cfg['max']}, Bins={cfg['bins']}")

        logger.debug("state_config validado.")
        return validated_config

    def _get_ordered_vars_for_gain(self, gain: str) -> List[str]:
        """Obtiene la lista ordenada de variables de estado *habilitadas* relevantes para una ganancia."""
        # ... (código sin cambios) ...
        ordered_vars = OrderedDict()
        for var, cfg in self.state_config.items():
            if cfg.get('enabled', False) and var not in self.gain_variables:
                ordered_vars[var] = True
        if gain in self.state_config and self.state_config[gain].get('enabled', False):
            ordered_vars[gain] = True
        return list(ordered_vars.keys())


    def _discretize_value(self, value: float, var_name: str) -> int:
        """Discretiza un valor continuo según la configuración de la variable."""
        # ... (código sin cambios) ...
        if var_name not in self.state_config or not self.state_config[var_name].get('enabled'):
             # logger.warning(f"Intento de discretizar variable '{var_name}' sin config o deshabilitada.")
             return 0 # Índice por defecto si no hay config o no está habilitada

        config = self.state_config[var_name]
        bins = config['bins']
        min_val = config['min']
        max_val = config['max']

        if bins <= 0 or max_val <= min_val: return 0

        clipped_value = np.clip(value, min_val, max_val)
        bin_size = (max_val - min_val) / bins
        if bin_size <= 1e-9: return 0

        if clipped_value >= max_val: index = bins - 1
        else: index = int(np.floor((clipped_value - min_val) / bin_size))

        return int(np.clip(index, 0, bins - 1))


    def get_discrete_state_indices_tuple(self, agent_state_dict: Optional[Dict[str, Any]], gain_variable: str) -> Optional[tuple]: # <--- Aceptar Optional
         """Convierte el diccionario de estado del agente en una tupla de índices discretos para una ganancia."""
         # --- NUEVA VALIDACIÓN ---
         if agent_state_dict is None:
              logger.error(f"Se recibió agent_state_dict=None al intentar obtener índices para ganancia '{gain_variable}'.")
              return None # Devolver None si el diccionario es None
         # -----------------------

         if gain_variable not in self.ordered_state_vars_for_gain:
             # logger.debug(f"No hay vars ordenadas para ganancia '{gain_variable}'.")
             return None

         ordered_vars = self.ordered_state_vars_for_gain[gain_variable]
         indices = []
         try:
             for var_name in ordered_vars:
                 # --- CORRECCIÓN: Verificar existencia *antes* de acceder ---
                 if var_name not in agent_state_dict:
                     # Este log ya existía, mantenerlo.
                     raise KeyError(f"Variable requerida '{var_name}' para ganancia '{gain_variable}' no en agent_state_dict: {list(agent_state_dict.keys())}")

                 value = agent_state_dict[var_name]
                 # Asegurar que value es numérico antes de discretizar
                 if not isinstance(value, (int, float)) or pd.isna(value):
                      logger.warning(f"Valor no numérico o NaN ({value}) para '{var_name}' en discretización '{gain_variable}'. Usando índice 0.")
                      index = 0
                 else:
                      index = self._discretize_value(float(value), var_name)
                 indices.append(index)
             return tuple(indices)

         except KeyError as e:
             # El error ahora es más informativo porque sabemos que agent_state_dict no era None
             logger.error(f"Error de clave al discretizar estado para '{gain_variable}': {e}")
             return None
         except Exception as e:
             logger.error(f"Error inesperado al discretizar estado para '{gain_variable}' (dict no era None): {e}", exc_info=True)
             return None


    # --- Implementación de Métodos Principales de la Interfaz RLAgent ---

    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """Selecciona una acción para cada ganancia (Kp, Ki, Kd) usando epsilon-greedy."""
        # ... (código sin cambios) ...
        actions: Dict[str, int] = {}
        exploration_decisions = {} # Para logging
        perform_exploration = np.random.rand() < self._epsilon

        for gain in self.gain_variables:
            action_index = 1 # Acción por defecto: mantener ganancia
            if gain in self.q_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        if perform_exploration:
                            action_index = np.random.randint(self.num_actions)
                            exploration_decisions[gain] = True
                        else:
                            q_values_for_state = self.q_tables_np[gain][state_indices]
                            action_index = int(np.argmax(q_values_for_state))
                            exploration_decisions[gain] = False
                    except IndexError:
                        logger.error(f"IndexError seleccionando acción para '{gain}'. Índices: {state_indices}. Shape Q: {self.q_tables_np[gain].shape}. Usando acción 1.")
                        action_index = 1
                    except Exception as e:
                        logger.error(f"Error inesperado seleccionando acción para '{gain}': {e}. Usando acción 1.", exc_info=True)
                        action_index = 1
                else:
                    logger.warning(f"No se pudieron obtener índices para '{gain}' en select_action. Usando acción 1.")
                    action_index = 1
            # else: logger.debug(f"No hay tabla Q para ganancia '{gain}'. Usando acción 1.")
            actions[gain] = action_index
        # logger.debug(f"Acciones: {actions} (Explorando: {exploration_decisions}, Epsilon: {self._epsilon:.3f})")
        return actions


    def learn(self,
              current_agent_state_dict: Dict[str, Any], # S
              actions_dict: Dict[str, int],             # A
              reward_info: Union[float, Tuple[float, float], Dict[str, float]], # R
              next_agent_state_dict: Dict[str, Any],    # S'
              done: bool):                               # Done flag
        """Actualiza las Q-tables usando la experiencia (S, A, R, S') y la RewardStrategy inyectada."""
        # ... (código sin cambios funcionales, pero usando self.reward_strategy) ...

        # 1. Parsear reward_info (como antes)
        interval_reward: float = 0.0; avg_w_stab: float = 1.0; reward_dict: Optional[Dict[str, float]] = None
        if isinstance(reward_info, tuple) and len(reward_info) == 2:
            interval_reward, avg_w_stab = reward_info
            if pd.isna(avg_w_stab) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0
        elif isinstance(reward_info, dict):
            reward_dict = reward_info
            valid_rewards = [r for r in reward_info.values() if pd.notna(r) and np.isfinite(r)]
            interval_reward = np.mean(valid_rewards) if valid_rewards else 0.0
        elif isinstance(reward_info, (float, int)):
            interval_reward = float(reward_info)
        else:
            logger.error(f"Learn recibió reward_info de tipo inesperado: {type(reward_info)}. Saltando learn.")
            return

        # Resetear TD errors
        self._last_td_errors = {gain: np.nan for gain in self.gain_variables}

        # 2. Iterar por cada ganancia
        for gain in self.gain_variables:
            if gain not in self.q_tables_np: continue # Saltar si no hay tabla

            try:
                # 3. Obtener índices S y S'
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)
                if current_state_indices is None or next_state_indices is None:
                    logger.warning(f"No se pudieron obtener índices para '{gain}' en learn. Saltando.")
                    continue

                # 4. Obtener acción A_g
                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not (0 <= action_taken_idx < self.num_actions):
                    logger.warning(f"Índice de acción inválido '{action_taken_idx}' para '{gain}' en learn. Saltando.")
                    continue

                # 5. Calcular R_learn usando la ESTRATEGIA INYECTADA
                reward_for_q_update = self.reward_strategy.compute_reward_for_learning(
                    gain=gain, agent=self, # Pasar self (agente)
                    current_agent_state_dict=current_agent_state_dict,
                    current_state_indices=current_state_indices,
                    actions_dict=actions_dict,
                    action_taken_idx=action_taken_idx,
                    interval_reward=interval_reward,
                    avg_w_stab=avg_w_stab,
                    reward_dict=reward_dict
                )

                # Validar R_learn
                if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
                     logger.warning(f"RewardStrategy devolvió R_learn inválido ({reward_for_q_update}) para '{gain}'. Usando 0.")
                     reward_for_q_update = 0.0

                # 6. Calcular TD Target y TD Error
                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]

                if done:
                    td_target = reward_for_q_update
                else:
                    next_q_values = self.q_tables_np[gain][next_state_indices]
                    max_next_q = np.nanmax(next_q_values)
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
                q_shape = self.q_tables_np[gain].shape if gain in self.q_tables_np else 'N/A'
                b_shape = self.baseline_tables_np.get(gain, np.array([])).shape # Safe get
                logger.error(f"IndexError en learn '{gain}'. Índices S: {current_state_indices}, Acción: {action_taken_idx}. Q Shape: {q_shape}, B Shape: {b_shape}. Error: {e}")
            except KeyError as e:
                logger.error(f"KeyError durante learn '{gain}': {e}. Verificar dicts o lógica estrategia.")
            except Exception as e:
                logger.error(f"Error inesperado durante learn '{gain}': {e}.", exc_info=True)


    def reset_agent(self):
        """Actualiza epsilon y learning rate al final de un episodio."""
        if self.use_epsilon_decay:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        if self.use_learning_rate_decay:
            self._learning_rate = max(self._learning_rate_min, self._learning_rate * self._learning_rate_decay)
        # Resetear TD errors
        self._last_td_errors = {gain: np.nan for gain in self.gain_variables}
        # logger.debug(f"Agent reset: Epsilon={self.epsilon:.4f}, LR={self.learning_rate:.4f}")


    def build_agent_state(self, raw_state_vector: Any, controller: Controller, state_config_for_build: Dict) -> Dict[str, Any]:
        """Construye el diccionario de estado del agente a partir del estado crudo y el controlador."""
        # ... (código sin cambios funcionales, solo asegurar uso de interfaz Controller) ...
        agent_state = {}
        state_vector_map = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        try: current_gains = controller.get_params()
        except Exception as e: logger.error(f"Error get_params en build_agent_state: {e}."); current_gains = {'kp': np.nan, 'ki': np.nan, 'kd': np.nan}

        for var_name, config in state_config_for_build.items():
            if config.get('enabled', False):
                value = np.nan
                if var_name in state_vector_map:
                    idx = state_vector_map[var_name]
                    try:
                        if isinstance(raw_state_vector, (list, np.ndarray)) and len(raw_state_vector) > idx: value = raw_state_vector[idx]
                        else: logger.warning(f"raw_state_vector inválido ({raw_state_vector}) para índice {idx} de '{var_name}'.")
                    except Exception as e: logger.warning(f"Error acceso índice {idx} para '{var_name}': {e}")
                elif var_name in current_gains: value = current_gains[var_name]
                else: logger.warning(f"Variable habilitada '{var_name}' no encontrada.")

                try: agent_state[var_name] = float(value) if pd.notna(value) else np.nan
                except (TypeError, ValueError): logger.warning(f"No se pudo convertir valor ({value}) a float para '{var_name}'."); agent_state[var_name] = np.nan
        # logger.debug(f"Agent state construido: {agent_state}")
        return agent_state


    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """
        Prepara el estado interno del agente (tablas Q, Visit, Baseline) para guardado en JSON/Excel.
        Los valores de estado en las tablas representan puntos discretos equiespaciados
        que incluyen los límites min/max definidos en la configuración.
        """
        structured_q_tables = {}
        structured_visit_counts = {}
        structured_baseline_tables = {}
        logger.info("Estructurando estado del agente para guardado (formato Pandas)...")

        # Definir precisión para guardar valores flotantes de estado representativo
        float_precision_for_keys = 6 # Decimales a redondear

        processed_gains = 0
        for gain in self.gain_variables:
            # Comprobar si existen tablas para esta ganancia
            if gain in self.q_tables_np and gain in self.ordered_state_vars_for_gain:
                logger.debug(f"Estructurando datos para ganancia '{gain}'...")
                q_table_list = []
                visit_count_list = []
                baseline_list = [] # Lista para B(s)

                ordered_vars = self.ordered_state_vars_for_gain[gain]
                np_q_table = self.q_tables_np[gain]
                np_visits = self.visit_counts_np[gain]
                # Acceder a baseline table (puede no existir si hubo error en init)
                np_baseline = self.baseline_tables_np.get(gain) # Use .get for safety

                if np_q_table is None or np_visits is None:
                     logger.warning(f"Tabla Q o de Visitas es None para ganancia '{gain}'. Saltando estructuración.")
                     continue

                state_shape = np_q_table.shape[:-1] # Shape sin la dimensión de acción
                action_dim = np_q_table.shape[-1]

                # Precalcular los puntos discretos para cada variable de esta ganancia
                discrete_points_cache = {}
                valid_cache = True
                for var_name in ordered_vars:
                     try:
                          config = self.state_config[var_name]
                          min_val, max_val, bins = config['min'], config['max'], config['bins']
                          if bins > 1:
                               discrete_points_cache[var_name] = np.linspace(min_val, max_val, bins)
                          elif bins == 1:
                               discrete_points_cache[var_name] = np.array([(min_val + max_val) / 2.0])
                          else: # bins <= 0 (inválido, pero manejar)
                               logger.warning(f"Bins inválido ({bins}) para '{var_name}' en get_agent_state_for_saving. Usando min_val.")
                               discrete_points_cache[var_name] = np.array([min_val])
                     except KeyError as e:
                          logger.error(f"Falta configuración ('{e}') para variable '{var_name}' al precalcular puntos discretos. Saltando ganancia '{gain}'.")
                          valid_cache = False
                          break # Salir del bucle de variables si falta config
                if not valid_cache:
                     continue # Saltar a la siguiente ganancia si falla el precalculo

                # Iterar sobre todas las combinaciones de índices de estado discretos
                total_states_in_table = np.prod(state_shape)
                processed_states = 0
                for state_indices_tuple in np.ndindex(state_shape):
                    processed_states += 1
                    # Log progreso opcionalmente (puede ser lento)
                    # if processed_states % 10000 == 0:
                    #      logger.debug(f"  Gain '{gain}': Procesando estado {processed_states}/{total_states_in_table}")

                    # --- Crear diccionario que representa el estado (valores discretos exactos) ---
                    state_repr_dict = {}
                    try:
                        for i, var_name in enumerate(ordered_vars):
                            # Obtener el valor precalculado usando el índice discreto
                            current_index = state_indices_tuple[i]
                            # Validar índice contra tamaño de puntos cacheados
                            if current_index >= len(discrete_points_cache[var_name]):
                                 raise IndexError(f"Índice {current_index} fuera de rango para puntos discretos de '{var_name}' (tamaño {len(discrete_points_cache[var_name])}).")
                            repr_value = discrete_points_cache[var_name][current_index]
                            # Redondear para evitar problemas de precisión flotante en claves JSON/Excel
                            state_repr_dict[var_name] = round(repr_value, float_precision_for_keys)

                    except IndexError as e: # Índice fuera de rango
                         logger.error(f"Error de índice al obtener punto discreto para estado {state_indices_tuple}, var '{var_name}' (índice {i}): {e}. Saltando estado.")
                         continue # Saltar al siguiente estado si hay error de índice
                    except KeyError as e: # Variable no encontrada en cache (no debería pasar por precalculo)
                         logger.error(f"Error de clave '{e}' al buscar puntos discretos cacheados para estado {state_indices_tuple}, var '{var_name}'. Saltando estado.")
                         continue
                    except Exception as e:
                         logger.error(f"Error inesperado calculando valor representativo para estado {state_indices_tuple}, var '{var_name}': {e}. Saltando estado.", exc_info=True)
                         continue

                    # --- Fila para Q-Table ---
                    try:
                         q_values = np_q_table[state_indices_tuple]
                         q_row = state_repr_dict.copy()
                         for action_idx in range(action_dim):
                              # Convertir a float estándar de Python
                              q_row[f"action_{action_idx}"] = float(q_values[action_idx])
                         q_table_list.append(q_row)
                    except IndexError:
                         logger.error(f"Error de índice accediendo a Q-table para estado {state_indices_tuple}. Saltando fila Q.")
                         continue # Saltar si falla Q-table

                    # --- Fila para Visit Counts ---
                    try:
                         visit_counts = np_visits[state_indices_tuple]
                         visit_row = state_repr_dict.copy()
                         for action_idx in range(action_dim):
                              # Convertir a int estándar de Python
                              visit_row[f"action_{action_idx}"] = int(visit_counts[action_idx])
                         visit_count_list.append(visit_row)
                    except IndexError:
                         logger.error(f"Error de índice accediendo a Visit counts para estado {state_indices_tuple}. Saltando fila Visitas.")
                         # Podríamos decidir continuar y guardar Q/Baseline si existen
                         # continue

                    # --- Fila para Baseline Table (si existe tabla) ---
                    if np_baseline is not None:
                        try:
                             baseline_value = np_baseline[state_indices_tuple]
                             baseline_row = state_repr_dict.copy()
                             baseline_row['baseline_value'] = float(baseline_value)
                             baseline_list.append(baseline_row)
                        except IndexError:
                             logger.error(f"Error de índice accediendo a Baseline table para estado {state_indices_tuple}. Saltando fila Baseline.")
                             # Continuar guardando Q/Visits si se pudo

                # Almacenar las listas para esta ganancia
                structured_q_tables[gain] = q_table_list
                structured_visit_counts[gain] = visit_count_list
                if np_baseline is not None:
                    structured_baseline_tables[gain] = baseline_list

                logger.debug(f"Estructuración para ganancia '{gain}' completa. {len(q_table_list)} estados procesados.")
                processed_gains += 1
            else:
                 # Loguear solo si se esperaba (está en gain_variables pero no en q_tables_np o ordered_state_vars)
                 if gain in self.gain_variables:
                      logger.debug(f"Saltando estructuración para ganancia '{gain}' (tablas no inicializadas o sin variables de estado asociadas).")

        if processed_gains == 0:
             logger.warning("No se procesó ninguna ganancia al estructurar estado del agente. Verificar inicialización y configuración.")

        logger.info("Estructuración del estado del agente completa.")
        # Devolver siempre las tres claves principales, aunque estén vacías
        return {
            "q_tables": structured_q_tables,
            "visit_counts": structured_visit_counts,
            "baseline_tables": structured_baseline_tables
        }


    # --- Implementación Métodos Helper para Logging ---

    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
         """Obtiene Q-values[acciones] para el estado dado para ganancias con tabla Q."""
         # ... (código sin cambios) ...
         q_values: Dict[str, np.ndarray] = {}
         for gain in self.gain_variables:
             q_vals_for_gain = np.full(self.num_actions, np.nan, dtype=np.float32)
             if gain in self.q_tables_np:
                 state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                 if state_indices is not None:
                     try: q_vals_for_gain = self.q_tables_np[gain][state_indices].astype(np.float32)
                     except IndexError: logger.warning(f"IndexError Q-values log ({gain}, {state_indices}).")
                     except Exception as e: logger.warning(f"Error Q-values log ({gain}): {e}")
             q_values[gain] = q_vals_for_gain
         return q_values

    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Obtiene N(s,a)[acciones] para el estado dado para ganancias con tabla de visitas."""
        # ... (código sin cambios) ...
        visit_counts: Dict[str, np.ndarray] = {}
        for gain in self.gain_variables:
            visits_for_gain = np.full(self.num_actions, -1, dtype=np.int32)
            if gain in self.visit_counts_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try: visits_for_gain = self.visit_counts_np[gain][state_indices].astype(np.int32)
                    except IndexError: logger.warning(f"IndexError Visit Counts log ({gain}, {state_indices}).")
                    except Exception as e: logger.warning(f"Error Visit Counts log ({gain}): {e}")
            visit_counts[gain] = visits_for_gain
        return visit_counts

    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """Obtiene B(s) para el estado dado para ganancias con tabla Baseline."""
        # ... (código sin cambios) ...
        baselines: Dict[str, float] = {}
        for gain in self.gain_variables:
            baseline_val = np.nan
            if gain in self.baseline_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        baseline_val = float(self.baseline_tables_np[gain][state_indices])
                        if not np.isfinite(baseline_val): baseline_val = np.nan
                    except IndexError: logger.warning(f"IndexError Baseline log ({gain}, {state_indices}).")
                    except Exception as e: logger.warning(f"Error Baseline log ({gain}): {e}")
            baselines[gain] = baseline_val
        return baselines

    def get_last_td_errors(self) -> Dict[str, float]:
         """Devuelve una copia del diccionario de TD errors del último paso de learn."""
         # ... (código sin cambios) ...
         return {k: (float(v) if pd.notna(v) and np.isfinite(v) else np.nan)
                 for k, v in self._last_td_errors.items()}