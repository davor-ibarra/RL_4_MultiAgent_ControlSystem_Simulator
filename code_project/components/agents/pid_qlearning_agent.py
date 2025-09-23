import numpy as np
import pandas as pd
import logging
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Import Reward Strategy Interface
from interfaces.controller import Controller # Para type hint en build_agent_state
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PIDQLearningAgent(RLAgent):
    """
    Agente Q-Learning para ajustar ganancias PID (Kp, Ki, Kd).
    Utiliza discretización del espacio de estados y tablas Q separadas por ganancia.
    La lógica de cálculo de recompensa para el aprendizaje se delega a una RewardStrategy inyectada.
    """
    def __init__(self,
                 # --- Configuración del Agente ---
                 state_config: Dict[str, Dict[str, Any]], # Configuración de discretización del estado
                 num_actions: int,                       # Número de acciones (e.g., 3: decrease, maintain, increase)
                 gain_step: Union[float, Dict[str, float]], # Tamaño del paso para ajustar ganancias
                 variable_step: bool,                   # Usar gain_step diferente por ganancia?
                 # --- Parámetros de Aprendizaje ---
                 reward_strategy: RewardStrategy,        # Estrategia de recompensa INYECTADA
                 discount_factor: float = 0.98,          # Gamma
                 epsilon: float = 1.0,                   # Epsilon inicial
                 epsilon_min: float = 0.1,               # Epsilon mínimo
                 epsilon_decay: float = 0.99954,         # Decaimiento de epsilon por episodio
                 learning_rate: float = 1.0,             # Alpha inicial
                 learning_rate_min: float = 0.01,        # Alpha mínimo
                 learning_rate_decay: float = 0.999425,  # Decaimiento de alpha por episodio
                 use_epsilon_decay: bool = True,         # Habilitar decaimiento de epsilon?
                 use_learning_rate_decay: bool = True,   # Habilitar decaimiento de alpha?
                 # --- Inicialización de Tablas ---
                 q_init_value: float = 0.0,              # Valor inicial para Q(s,a)
                 visit_init_value: int = 0,              # Valor inicial para N(s,a)
                 # --- Parámetros Opcionales (inyectados si la estrategia los necesita) ---
                 shadow_baseline_params: Optional[Dict] = None # e.g., {'baseline_init_value': 0.0}
                 ):
        """
        Inicializa el agente PID Q-Learning.

        Args:
            state_config: Configuración para la discretización de cada variable de estado.
            num_actions: Número de acciones discretas por ganancia.
            gain_step: Magnitud del cambio de ganancia por acción (float global o dict por ganancia).
            variable_step: Indica si gain_step es un diccionario por ganancia.
            reward_strategy: La instancia de RewardStrategy a usar para calcular R_learn.
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
        # --- Validación y Almacenamiento de Parámetros ---
        if not isinstance(reward_strategy, RewardStrategy):
             # Esta validación también ocurre en AgentFactory, pero doble check aquí
             raise TypeError("El argumento 'reward_strategy' debe ser una instancia de RewardStrategy.")
        self.reward_strategy = reward_strategy
        logger.info(f"Usando Reward Strategy: {type(self.reward_strategy).__name__}")

        try: # Validar state_config al inicio
             self.state_config = self._validate_and_prepare_state_config(state_config)
        except ValueError as e:
             logger.error(f"Configuración de estado inválida: {e}")
             raise

        self.num_actions = num_actions
        self.q_init_value = q_init_value
        self.visit_init_value = visit_init_value

        # Manejo de gain_step (global o variable)
        self.gain_step = gain_step
        self.variable_step = variable_step
        if variable_step and not isinstance(gain_step, dict):
             raise ValueError("Se especificó variable_step=True, pero gain_step no es un diccionario.")
        if not variable_step and not isinstance(gain_step, (float, int)):
             raise ValueError("Se especificó variable_step=False, pero gain_step no es numérico.")


        self.discount_factor = discount_factor
        # Guardar valores iniciales para posible reset completo (no implementado actualmente)
        self._initial_epsilon = epsilon
        self._initial_learning_rate = learning_rate
        # Valores actuales que decaen
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.learning_rate, self.learning_rate_min, self.learning_rate_decay = learning_rate, learning_rate_min, learning_rate_decay
        self.use_epsilon_decay, self.use_learning_rate_decay = use_epsilon_decay, use_learning_rate_decay

        # --- Configuración específica de Baseline ---
        self.is_shadow_mode = "ShadowBaseline" in type(self.reward_strategy).__name__
        # Obtener valor inicial de baseline si aplica (desde shadow_baseline_params)
        self.baseline_init_value = 0.0
        if self.is_shadow_mode:
             if shadow_baseline_params and isinstance(shadow_baseline_params, dict):
                  self.baseline_init_value = shadow_baseline_params.get('baseline_init_value', 0.0)
                  logger.info(f"Shadow Baseline detectado. Valor inicial de Baseline B(s): {self.baseline_init_value}")
             else:
                  logger.warning("Shadow Baseline detectado, pero 'shadow_baseline_params' no proporcionado o inválido. Usando baseline_init_value=0.0 por defecto.")

        # --- Inicialización de Estructuras Internas ---
        self.ordered_state_vars_for_gain: Dict[str, List[str]] = {} # Orden de variables por ganancia
        self.gain_variables = ['kp', 'ki', 'kd'] # Ganancias controladas

        # Usar NumPy arrays para eficiencia
        self.q_tables_np: Dict[str, np.ndarray] = {}
        self.visit_counts_np: Dict[str, np.ndarray] = {}
        self.baseline_tables_np: Dict[str, np.ndarray] = {} # Siempre inicializar, usar si es shadow mode

        # Almacenar últimos errores TD calculados para logging
        self._last_td_errors: Dict[str, float] = {}

        # --- Crear e Inicializar Tablas NumPy ---
        for gain in self.gain_variables:
            # Comprobar si la *ganancia específica* está habilitada como variable de estado
            # Si una ganancia no es variable de estado, no se discretiza y no necesita tabla.
            # La necesidad de tabla depende de si las *otras* variables habilitadas forman el estado.
            # --> Cambiar lógica: Crear tabla si *alguna* variable de estado está habilitada para esa ganancia.
            ordered_vars_list = self._get_ordered_vars_for_gain(gain)
            if not ordered_vars_list:
                 logger.info(f"No hay variables de estado habilitadas definidas para la ganancia '{gain}'. No se crearán tablas Q/Visit/Baseline.")
                 continue # Saltar a la siguiente ganancia si no hay estado definido

            logger.info(f"Inicializando tablas NumPy para ganancia '{gain}'...")
            self.ordered_state_vars_for_gain[gain] = ordered_vars_list
            try:
                state_dims = [self.state_config[var]['bins'] for var in ordered_vars_list]
                if not state_dims: # Si la lista está vacía por alguna razón
                     raise ValueError("No se encontraron dimensiones de estado para crear las tablas.")

                q_visit_shape = tuple(state_dims + [self.num_actions])
                baseline_shape = tuple(state_dims) # Shape para B(s) (sin dimensión de acción)
                logger.debug(f"  - Orden variables estado para '{gain}': {ordered_vars_list}")
                logger.debug(f"  - Dimensiones estado: {state_dims}")
                logger.debug(f"  - Shape Q/Visit para '{gain}': {q_visit_shape}")
                logger.debug(f"  - Shape Baseline para '{gain}': {baseline_shape}")

                # Inicializar tablas con valores por defecto y tipos eficientes
                self.q_tables_np[gain] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
                self.visit_counts_np[gain] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)
                # Inicializar Baseline table siempre, la estrategia decidirá si usarla/actualizarla
                self.baseline_tables_np[gain] = np.full(baseline_shape, self.baseline_init_value, dtype=np.float32)
                logger.debug(f"  - Tablas NumPy creadas para '{gain}'.")

            except KeyError as e:
                logger.error(f"Error obteniendo 'bins' para variable {e} al crear array para ganancia '{gain}'. Comprobar state_config. Saltando inicialización para '{gain}'.")
                # Limpiar estructuras si falló la creación
                if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
                if gain in self.q_tables_np: del self.q_tables_np[gain]
                if gain in self.visit_counts_np: del self.visit_counts_np[gain]
                if gain in self.baseline_tables_np: del self.baseline_tables_np[gain]
                continue # Saltar a la siguiente ganancia
            except ValueError as e:
                 logger.error(f"Error en dimensiones/shape al crear tablas para ganancia '{gain}': {e}. Saltando inicialización.")
                 if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
                 continue


        logger.info("PIDQLearningAgent inicializado exitosamente.")
        if not self.q_tables_np:
             logger.warning("¡Ninguna tabla Q fue inicializada! El agente no aprenderá. Verificar 'state_config' y habilitación de variables.")

    # --- Métodos Helper Internos (Discretización, Validación) ---

    def _validate_and_prepare_state_config(self, config: Dict) -> Dict:
        """Valida la estructura y contenido de state_config."""
        logger.debug("Validando state_config...")
        validated_config = {}
        required_keys = ['enabled', 'min', 'max', 'bins']
        if not isinstance(config, dict):
             raise ValueError("state_config debe ser un diccionario.")

        for var, cfg in config.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Configuración de estado para '{var}' debe ser un diccionario.")
            if 'enabled' not in cfg:
                raise ValueError(f"Configuración de estado para '{var}' falta la clave 'enabled'.")

            # Guardar config incluso si está deshabilitado
            validated_config[var] = cfg.copy() # Guardar copia

            if cfg['enabled']:
                missing_keys = [key for key in required_keys if key not in cfg]
                if missing_keys:
                    raise ValueError(f"Configuración de estado habilitada para '{var}' faltan claves requeridas: {missing_keys}.")
                # Validar tipos y valores
                if not isinstance(cfg['min'], (int, float)) or not isinstance(cfg['max'], (int, float)):
                    raise ValueError(f"Configuración '{var}': 'min' y 'max' deben ser numéricos.")
                if cfg['min'] >= cfg['max']:
                    # Permitir min == max SOLO si bins == 1 ? Por ahora no.
                    raise ValueError(f"Configuración '{var}': 'min' ({cfg['min']}) debe ser estrictamente menor que 'max' ({cfg['max']}).")
                if not isinstance(cfg['bins'], int) or cfg['bins'] <= 0:
                    raise ValueError(f"Configuración '{var}': 'bins' ({cfg['bins']}) debe ser un entero positivo.")
                logger.debug(f" - Variable '{var}': Habilitada, Min={cfg['min']}, Max={cfg['max']}, Bins={cfg['bins']}")
            #else:
            #     logger.debug(f" - Variable '{var}': Deshabilitada.")

        logger.debug("state_config validado.")
        return validated_config

    def _get_ordered_vars_for_gain(self, gain: str) -> List[str]:
        """Obtiene la lista ordenada de variables de estado *habilitadas* relevantes para una ganancia."""
        # El orden es importante para indexar las tablas NumPy consistentemente.
        # Orden sugerido: variables de estado físicas, luego la ganancia específica si está habilitada.
        ordered_vars = OrderedDict()
        # Variables de estado no-ganancia habilitadas
        for var, cfg in self.state_config.items():
            # Incluir si está habilitada Y no es una de las ganancias
            if cfg.get('enabled', False) and var not in self.gain_variables:
                ordered_vars[var] = True
        # Añadir la ganancia actual si está habilitada como variable de estado
        if gain in self.state_config and self.state_config[gain].get('enabled', False):
            ordered_vars[gain] = True

        return list(ordered_vars.keys())

    def _discretize_value(self, value: float, var_name: str) -> int:
        """Discretiza un valor continuo según la configuración de la variable."""
        if var_name not in self.state_config:
             # logger.warning(f"Intento de discretizar variable '{var_name}' sin configuración. Devolviendo índice 0.")
             return 0
        config = self.state_config[var_name]
        # Si la variable está deshabilitada en config, no debería llamarse a esto, pero por si acaso:
        if not config.get('enabled', False):
             # logger.warning(f"Intento de discretizar variable deshabilitada '{var_name}'. Devolviendo índice 0.")
             return 0

        bins = config.get('bins', 1)
        min_val = config.get('min', 0.0)
        max_val = config.get('max', 0.0)

        if bins <= 0: return 0 # Evitar división por cero y comportamiento indefinido
        if max_val <= min_val: return 0 # Rango inválido

        # Asegurar que el valor esté dentro de los límites antes de calcular el índice
        # Usar np.clip para manejar valores fuera de rango
        clipped_value = np.clip(value, min_val, max_val)

        # Calcular tamaño del bin
        bin_size = (max_val - min_val) / bins
        if bin_size <= 1e-9: return 0 # Evitar división por flotante muy pequeño

        # Calcular índice basado en el valor clipeado
        # Usar floor para asignar al bin correcto
        # Manejar caso límite donde value == max_val
        if clipped_value >= max_val:
            index = bins - 1
        else:
            index = int(np.floor((clipped_value - min_val) / bin_size))

        # Asegurar que el índice final esté dentro del rango [0, bins-1]
        return int(np.clip(index, 0, bins - 1))


    def get_discrete_state_indices_tuple(self, agent_state_dict: Dict[str, Any], gain_variable: str) -> Optional[tuple]:
         """Convierte el diccionario de estado del agente en una tupla de índices discretos para una ganancia."""
         if gain_variable not in self.ordered_state_vars_for_gain:
             # Esto puede pasar si no se inicializaron tablas para la ganancia
             # logger.debug(f"No hay variables ordenadas definidas para ganancia '{gain_variable}'. No se puede obtener tupla de índices.")
             return None

         ordered_vars = self.ordered_state_vars_for_gain[gain_variable]
         indices = []
         try:
             for var_name in ordered_vars:
                 # Ya sabemos que var_name está en self.state_config y está habilitado
                 # porque _get_ordered_vars_for_gain solo incluye esas.
                 config = self.state_config[var_name]

                 # Comprobar si la variable existe en el estado del agente proporcionado
                 if var_name not in agent_state_dict:
                     raise KeyError(f"Variable de estado requerida '{var_name}' para ganancia '{gain_variable}' "
                                    f"no encontrada en agent_state_dict. Claves disponibles: {list(agent_state_dict.keys())}")

                 value = agent_state_dict[var_name]
                 # Convertir a float por si acaso viene como int
                 index = self._discretize_value(float(value), var_name)
                 indices.append(index)

             return tuple(indices)

         except KeyError as e:
             logger.error(f"Error de clave al discretizar estado para ganancia '{gain_variable}': {e}")
             return None
         except Exception as e:
             logger.error(f"Error inesperado al discretizar estado para '{gain_variable}': {e}", exc_info=True)
             return None


    # --- Métodos Principales de la Interfaz RLAgent ---

    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        """Selecciona una acción para cada ganancia (Kp, Ki, Kd) usando epsilon-greedy."""
        actions: Dict[str, int] = {}
        exploration_decisions = {} # Para logging

        # Decidir exploración una vez por paso de decisión, no por ganancia
        perform_exploration = np.random.rand() < self.epsilon

        for gain in self.gain_variables:
            action_index = 1 # Acción por defecto: mantener ganancia

            # Solo seleccionar acción si hay una tabla Q (y por tanto estado definido) para esta ganancia
            if gain in self.q_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)

                if state_indices is not None:
                    try:
                        if perform_exploration:
                            # Explorar: elegir acción aleatoria (0, 1, o 2)
                            action_index = np.random.randint(self.num_actions)
                            exploration_decisions[gain] = True
                        else:
                            # Explotar: elegir la mejor acción según Q-table
                            q_values_for_state = self.q_tables_np[gain][state_indices]
                            action_index = int(np.argmax(q_values_for_state)) # Convertir a int estándar
                            exploration_decisions[gain] = False

                    except IndexError:
                        logger.error(f"IndexError seleccionando acción para ganancia '{gain}'. Índices: {state_indices}. Shape Q: {self.q_tables_np[gain].shape}. Usando acción por defecto 1.")
                        action_index = 1 # Fallback
                    except Exception as e:
                        logger.error(f"Error inesperado seleccionando acción para ganancia '{gain}': {e}. Usando acción por defecto 1.", exc_info=True)
                        action_index = 1 # Fallback
                else:
                    # Si no se pudieron obtener índices (error previo)
                    logger.warning(f"No se pudieron obtener índices de estado para ganancia '{gain}' en select_action. Usando acción por defecto 1.")
                    action_index = 1
            else:
                 # Si no hay tabla Q para esta ganancia (estado no definido o error init)
                 # logger.debug(f"No hay tabla Q para ganancia '{gain}'. Usando acción por defecto 1.")
                 action_index = 1

            actions[gain] = action_index

        # logger.debug(f"Acciones seleccionadas: {actions} (Explorando: {exploration_decisions})")
        return actions


    def learn(self,
              current_agent_state_dict: Dict[str, Any], # S
              actions_dict: Dict[str, int],             # A
              reward_info: Union[float, Tuple[float, float], Dict[str, float]], # R
              next_agent_state_dict: Dict[str, Any],    # S'
              done: bool):                               # Done flag
        """Actualiza las Q-tables usando la experiencia (S, A, R, S') y la RewardStrategy."""

        # 1. Parsear reward_info (ya hecho en SimulationManager, pero podemos verificar tipo)
        interval_reward: float = 0.0
        avg_w_stab: float = 1.0
        reward_dict: Optional[Dict[str, float]] = None

        if isinstance(reward_info, tuple) and len(reward_info) == 2: # Shadow Baseline: (R_real, avg_w_stab)
            interval_reward, avg_w_stab = reward_info
            if pd.isna(avg_w_stab) or not np.isfinite(avg_w_stab): avg_w_stab = 1.0
        elif isinstance(reward_info, dict): # Echo Baseline: {'kp': R_diff_kp, ...}
            reward_dict = reward_info
            # Estimar R_real promedio si es necesario (aunque la estrategia Echo no la usa directamente)
            valid_rewards = [r for r in reward_info.values() if pd.notna(r) and np.isfinite(r)]
            interval_reward = np.mean(valid_rewards) if valid_rewards else 0.0
        elif isinstance(reward_info, (float, int)): # Global Reward: R_real
            interval_reward = float(reward_info)
        else:
            logger.error(f"Learn recibió reward_info de tipo inesperado: {type(reward_info)}. Saltando paso de aprendizaje.")
            return

        # Limpiar TD errors anteriores antes de calcular los nuevos
        self._last_td_errors = {}

        # 2. Iterar por cada ganancia para actualizar su Q-table
        for gain in self.gain_variables:
            if gain not in self.q_tables_np:
                # logger.debug(f"Saltando aprendizaje para ganancia '{gain}' (sin tabla Q).")
                continue # Saltar si no hay tabla para esta ganancia

            try:
                # 3. Obtener índices discretos para S y S'
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)

                if current_state_indices is None or next_state_indices is None:
                    logger.warning(f"No se pudieron obtener índices de estado para ganancia '{gain}' durante learn. Saltando actualización.")
                    continue

                # 4. Obtener acción A tomada para esta ganancia específica
                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not isinstance(action_taken_idx, int) or not (0 <= action_taken_idx < self.num_actions):
                    logger.warning(f"Índice de acción inválido '{action_taken_idx}' para ganancia '{gain}' en learn(). Saltando actualización.")
                    continue

                # 5. Calcular la recompensa específica para Q-update (R_learn) usando la estrategia
                #    La estrategia puede también actualizar tablas Baseline B(s) internamente.
                reward_for_q_update = self.reward_strategy.compute_reward_for_learning(
                    gain=gain,
                    agent=self,
                    current_agent_state_dict=current_agent_state_dict,
                    current_state_indices=current_state_indices,
                    actions_dict=actions_dict,
                    action_taken_idx=action_taken_idx,
                    interval_reward=interval_reward, # Pasar R_real
                    avg_w_stab=avg_w_stab,           # Pasar w_stab promedio
                    reward_dict=reward_dict          # Pasar R_diff (para Echo)
                )

                # Verificar si R_learn es válida
                if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
                     logger.warning(f"RewardStrategy devolvió un valor inválido ({reward_for_q_update}) para ganancia '{gain}'. Usando 0 para Q-update.")
                     reward_for_q_update = 0.0


                # 6. Calcular TD Target y TD Error
                # Índice completo para Q(s,a) y N(s,a)
                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]

                if done:
                    # Si es estado terminal, el valor del siguiente estado es 0
                    td_target = reward_for_q_update
                else:
                    # Obtener Q-values del siguiente estado (S') para todas las acciones posibles
                    next_q_values = self.q_tables_np[gain][next_state_indices]
                    # Encontrar el máximo Q-value en el siguiente estado (max_a' Q(s', a'))
                    # Usar nanmax por si algún Q-value se volvió NaN
                    max_next_q = np.nanmax(next_q_values)
                    # Si todos son NaN o no hay next_q_values, tratar como 0
                    if pd.isna(max_next_q) or not np.isfinite(max_next_q):
                         # logger.warning(f"Max next Q-value inválido para ganancia '{gain}', estado S' {next_state_indices}. Usando 0.")
                         max_next_q = 0.0
                    # TD Target = R_learn + gamma * max_a' Q(s', a')
                    td_target = reward_for_q_update + self.discount_factor * max_next_q

                # TD Error = TD Target - Q(s, a)
                td_error = td_target - current_q

                # 7. Actualizar Q-Table
                # Q(s, a) <- Q(s, a) + alpha * TD_Error
                new_q_value = current_q + self.learning_rate * td_error
                # Asegurar que no actualizamos con NaN/inf
                if pd.notna(new_q_value) and np.isfinite(new_q_value):
                     self.q_tables_np[gain][full_index_current] = new_q_value
                # else:
                     # logger.warning(f"Nuevo Q-value calculado es inválido ({new_q_value}) para ganancia '{gain}'. No se actualizó Q-table.")


                # 8. Actualizar Contador de Visitas N(s, a)
                self.visit_counts_np[gain][full_index_current] += 1

                # 9. Guardar TD error para logging
                self._last_td_errors[gain] = float(td_error) if pd.notna(td_error) and np.isfinite(td_error) else np.nan

            except IndexError as e:
                q_shape = self.q_tables_np[gain].shape if gain in self.q_tables_np else 'N/A'
                b_shape = self.baseline_tables_np[gain].shape if gain in self.baseline_tables_np else 'N/A'
                logger.error(f"IndexError actualizando tabla NumPy '{gain}'. Índices S: {current_state_indices}, Acción: {action_taken_idx}. "
                             f"Q Shape: {q_shape}, B Shape: {b_shape}. Error: {e}.")
            except KeyError as e:
                logger.error(f"KeyError durante learn para ganancia '{gain}': {e}. Verificar acceso a dicts (state, actions) o lógica de estrategia.")
            except Exception as e:
                logger.error(f"Error inesperado durante learn para ganancia '{gain}': {e}.", exc_info=True)


    def reset_agent(self):
        """Actualiza epsilon y learning rate al final de un episodio."""
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.use_learning_rate_decay:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay)
        # Resetear TD errors para el nuevo episodio
        self._last_td_errors = {}
        # logger.debug(f"Agent reset: Epsilon={self.epsilon:.4f}, LR={self.learning_rate:.4f}")


    def build_agent_state(self, raw_state_vector: Any, controller: Controller, state_config_for_build: Dict) -> Dict[str, Any]:
        """Construye el diccionario de estado del agente a partir del estado crudo y el controlador."""
        agent_state = {}

        # Mapeo de nombres estándar a índices del vector de estado (asumiendo orden estándar)
        state_vector_map = {
            'cart_position': 0,
            'cart_velocity': 1,
            'angle': 2,           # Usar 'angle' consistentemente
            'angular_velocity': 3 # Usar 'angular_velocity' consistentemente
        }

        # Obtener ganancias actuales del controlador
        try:
             current_gains = controller.get_params()
        except Exception as e:
             logger.error(f"Error obteniendo ganancias del controlador en build_agent_state: {e}. Usando NaN.")
             current_gains = {'kp': np.nan, 'ki': np.nan, 'kd': np.nan}


        for var_name, config in state_config_for_build.items():
            if config.get('enabled', False):
                value = np.nan # Default a NaN
                if var_name in state_vector_map:
                    idx = state_vector_map[var_name]
                    try:
                        # Asegurar que raw_state_vector es indexable y tiene suficientes elementos
                        if isinstance(raw_state_vector, (list, np.ndarray)) and len(raw_state_vector) > idx:
                            value = raw_state_vector[idx]
                        else:
                            logger.warning(f"raw_state_vector inválido o demasiado corto ({raw_state_vector}) para obtener índice {idx} de '{var_name}'.")
                    except Exception as e:
                         logger.warning(f"Error accediendo a índice {idx} de raw_state_vector para '{var_name}': {e}")
                elif var_name in current_gains:
                    value = current_gains[var_name]
                else:
                    logger.warning(f"Variable de estado habilitada '{var_name}' no encontrada en state_vector_map ni en controller gains.")

                # Asegurar que el valor es float para consistencia
                try:
                     agent_state[var_name] = float(value) if pd.notna(value) else np.nan
                except (TypeError, ValueError):
                     logger.warning(f"No se pudo convertir valor ({value}) a float para variable '{var_name}'. Usando NaN.")
                     agent_state[var_name] = np.nan


        # logger.debug(f"Agent state construido: {agent_state}")
        return agent_state


    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        """Prepara el estado interno del agente (tablas Q, Visit, Baseline) para guardado en JSON/Excel."""
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
                np_baseline = self.baseline_tables_np.get(gain)

                state_shape = np_q_table.shape[:-1] # Shape sin la dimensión de acción
                action_dim = np_q_table.shape[-1]

                # Iterar sobre todas las combinaciones de índices de estado discretos
                total_states_in_table = np.prod(state_shape)
                processed_states = 0
                for state_indices_tuple in np.ndindex(state_shape):
                    processed_states += 1
                    # if processed_states % 1000 == 0: # Log progreso
                    #      logger.debug(f"  Gain '{gain}': Procesando estado {processed_states}/{total_states_in_table}")

                    # --- Crear diccionario que representa el estado (valores medios del bin) ---
                    state_repr_dict = {}
                    try:
                        for i, var_name in enumerate(ordered_vars):
                            config = self.state_config[var_name]
                            min_val, max_val, bins = config['min'], config['max'], config['bins']
                            # Calcular valor representativo (centro del bin)
                            bin_size = (max_val - min_val) / bins
                            repr_value = min_val + (state_indices_tuple[i] + 0.5) * bin_size
                            # Redondear para evitar problemas de precisión flotante en claves JSON/Excel
                            state_repr_dict[var_name] = round(repr_value, float_precision_for_keys)
                    except Exception as e:
                         logger.error(f"Error calculando valor representativo para estado {state_indices_tuple}, var '{var_name}' (índice {i}): {e}. Saltando estado.")
                         continue

                    # --- Fila para Q-Table ---
                    q_values = np_q_table[state_indices_tuple]
                    q_row = state_repr_dict.copy()
                    for action_idx in range(action_dim):
                        # Convertir a float estándar de Python
                        q_row[f"action_{action_idx}"] = float(q_values[action_idx])
                    q_table_list.append(q_row)

                    # --- Fila para Visit Counts ---
                    visit_counts = np_visits[state_indices_tuple]
                    visit_row = state_repr_dict.copy()
                    for action_idx in range(action_dim):
                        # Convertir a int estándar de Python
                        visit_row[f"action_{action_idx}"] = int(visit_counts[action_idx])
                    visit_count_list.append(visit_row)

                    # --- Fila para Baseline Table (si existe tabla) ---
                    if np_baseline is not None:
                        baseline_value = np_baseline[state_indices_tuple]
                        baseline_row = state_repr_dict.copy()
                        baseline_row['baseline_value'] = float(baseline_value)
                        baseline_list.append(baseline_row)


                # Almacenar las listas para esta ganancia
                structured_q_tables[gain] = q_table_list
                structured_visit_counts[gain] = visit_count_list
                if np_baseline is not None:
                    structured_baseline_tables[gain] = baseline_list

                logger.debug(f"Estructuración para ganancia '{gain}' completa. {len(q_table_list)} estados procesados.")
                processed_gains += 1
            else:
                 if gain in self.gain_variables: # Loguear solo si se esperaba
                      logger.debug(f"Saltando estructuración para ganancia '{gain}' (tablas no inicializadas).")

        if processed_gains == 0:
             logger.warning("No se procesó ninguna ganancia al estructurar estado del agente. Verificar inicialización.")

        logger.info("Estructuración del estado del agente completa.")
        # Devolver siempre las tres claves principales
        return {
            "q_tables": structured_q_tables,
            "visit_counts": structured_visit_counts,
            "baseline_tables": structured_baseline_tables
        }


    # --- Métodos Helper para Logging (Implementación de la interfaz) ---

    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
         """Obtiene Q-values[acciones] para el estado dado para ganancias con tabla Q."""
         q_values: Dict[str, np.ndarray] = {}
         for gain in self.gain_variables:
             q_vals_for_gain = np.full(self.num_actions, np.nan, dtype=np.float32) # Default NaN
             if gain in self.q_tables_np:
                 state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                 if state_indices is not None:
                     try:
                         q_vals_for_gain = self.q_tables_np[gain][state_indices].astype(np.float32)
                     except IndexError:
                         logger.warning(f"IndexError obteniendo Q-values para log ({gain}, {state_indices}).")
                     except Exception as e:
                          logger.warning(f"Error obteniendo Q-values para log ({gain}): {e}")
             q_values[gain] = q_vals_for_gain
         return q_values

    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """Obtiene N(s,a)[acciones] para el estado dado para ganancias con tabla de visitas."""
        visit_counts: Dict[str, np.ndarray] = {}
        for gain in self.gain_variables:
            visits_for_gain = np.full(self.num_actions, -1, dtype=np.int32) # Default -1 (error)
            if gain in self.visit_counts_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        visits_for_gain = self.visit_counts_np[gain][state_indices].astype(np.int32)
                    except IndexError:
                         logger.warning(f"IndexError obteniendo Visit Counts para log ({gain}, {state_indices}).")
                    except Exception as e:
                         logger.warning(f"Error obteniendo Visit Counts para log ({gain}): {e}")
            visit_counts[gain] = visits_for_gain
        return visit_counts

    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """Obtiene B(s) para el estado dado para ganancias con tabla Baseline."""
        baselines: Dict[str, float] = {}
        for gain in self.gain_variables:
            baseline_val = np.nan # Default NaN
            if gain in self.baseline_tables_np: # Comprobar si la tabla existe
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        # Obtener valor y convertir a float estándar
                        baseline_val = float(self.baseline_tables_np[gain][state_indices])
                        if not np.isfinite(baseline_val): baseline_val = np.nan # Asegurar NaN si es inf
                    except IndexError:
                        logger.warning(f"IndexError obteniendo Baseline para log ({gain}, {state_indices}).")
                    except Exception as e:
                         logger.warning(f"Error obteniendo Baseline para log ({gain}): {e}")
            baselines[gain] = baseline_val
        return baselines

    def get_last_td_errors(self) -> Dict[str, float]:
         """Devuelve una copia del diccionario de TD errors del último paso de learn."""
         # Devolver copia para evitar modificación externa
         # Asegurar que los valores son float o NaN
         return {k: (float(v) if pd.notna(v) and np.isfinite(v) else np.nan)
                 for k, v in self._last_td_errors.items()}