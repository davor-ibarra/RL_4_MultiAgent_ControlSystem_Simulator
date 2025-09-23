# components/agents/pid_qlearning_agent.py
import numpy as np
import pandas as pd
import logging
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy
from interfaces.controller import Controller # Para type hint
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union, List

from components.reward_strategies.shadow_baseline_reward_strategy import ShadowBaselineRewardStrategy

logger = logging.getLogger(__name__) # Logger específico del módulo

class PIDQLearningAgent(RLAgent):
    def __init__(self,
                 reward_strategy: RewardStrategy,
                 state_config: Dict[str, Dict[str, Any]],
                 num_actions: int,
                 gain_step: Union[float, Dict[str, float]],
                 variable_step: bool,
                 discount_factor: float = 0.98,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.99954,
                 learning_rate: float = 1.0,
                 learning_rate_min: float = 0.01,
                 learning_rate_decay: float = 0.999425,
                 use_epsilon_decay: bool = True,
                 use_learning_rate_decay: bool = True,
                 q_init_value: float = 0.0,
                 visit_init_value: int = 0,
                 shadow_baseline_params: Optional[Dict] = None,
                 early_termination_config: Optional[Dict] = None # Nombre corregido
                 ):
        logger.info(f"[PIDQLearningAgent] Initializing with RewardStrategy: {type(reward_strategy).__name__}")
        if not isinstance(reward_strategy, RewardStrategy):
             msg = f"Invalid dependency: 'reward_strategy' must be instance of RewardStrategy, not {type(reward_strategy).__name__}."
             logger.critical(f"[PIDQLearningAgent] {msg}"); raise TypeError(msg)
        self._reward_strategy_instance = reward_strategy 

        try:
             self.state_config = self._validate_and_prepare_state_config(state_config, "enabled_agent")
        except ValueError as e:
             logger.critical(f"[PIDQLearningAgent] Invalid state_config: {e}", exc_info=True); raise

        if not isinstance(num_actions, int) or num_actions <= 0:
             raise ValueError(f"num_actions ({num_actions}) must be a positive integer.")
        self.num_actions = num_actions
        self.q_init_value = float(q_init_value)
        self.visit_init_value = int(visit_init_value)

        self.variable_step = bool(variable_step)
        if self.variable_step:
            if not isinstance(gain_step, dict): raise ValueError("variable_step=True, but gain_step is not a dict.")
        elif not isinstance(gain_step, (float, int)): raise ValueError("variable_step=False, but gain_step is not numeric.")
        self.gain_step = gain_step

        self.discount_factor = float(discount_factor)
        if not (0 <= self.discount_factor <= 1): raise ValueError("discount_factor must be in [0, 1].")
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

        self.auxiliary_tables_info: Dict[str, Dict[str, np.ndarray]] = {}
        self.aux_visit_counts_np: Dict[str, Dict[str, np.ndarray]] = {}

        self.baseline_init_value = 0.0 
        if 'baseline' in self.reward_strategy.required_auxiliary_tables:
            if shadow_baseline_params and isinstance(shadow_baseline_params, dict):
                self.baseline_init_value = float(shadow_baseline_params.get('baseline_init_value', 0.0))
            logger.info(f"[PIDQLearningAgent] Reward strategy requires 'baseline' table. B(s) init value: {self.baseline_init_value}")

        self.agent_defining_vars: List[str] = [
            var for var, cfg in self.state_config.items() if cfg.get("enabled_agent", False)
        ]
        if not self.agent_defining_vars:
            logger.warning("[PIDQLearningAgent] No variables have 'enabled_agent: true'. No Q-tables will be created.")

        self.ordered_state_vars_for_agent_var: Dict[str, List[str]] = {}
        self.q_tables_np: Dict[str, np.ndarray] = {}
        self.visit_counts_np: Dict[str, np.ndarray] = {}
        self._last_td_errors: Dict[str, float] = {adv: np.nan for adv in self.agent_defining_vars}
        
        self._initialize_tables() 
        
        if not self.q_tables_np:
            logger.warning("[PIDQLearningAgent] WARNING: No Q-tables were initialized! Check 'state_config' and 'enabled_agent' flags.")
        
        # --- Inicialización de Early Termination ---
        self._request_early_termination_flags: Dict[str, bool] = {adv: False for adv in self.agent_defining_vars}
        self.early_termination_config_data = early_termination_config if isinstance(early_termination_config, dict) else {}
        self.early_termination_enabled = self.early_termination_config_data.get('enabled', False)

        self.patience_M: Dict[str, int] = {}
        self.no_improvement_counter_c_hat: Dict[str, int] = {}
        self.penalty_beta: Dict[str, float] = {}
        self.last_interval_improvement_metric_m_bar: Dict[str, float] = {}
        self.current_interval_improvement_deltas: Dict[str, List[float]] = {adv: [] for adv in self.agent_defining_vars}
        self.last_avg_episode_improvement_metric: Dict[str, float] = {adv: 0.0 for adv in self.agent_defining_vars}
        self.initial_patience_M0_from_config: Dict[str, int] = {} 
        self.initial_penalty_beta0_from_config: Dict[str, float] = {}

        if self.early_termination_enabled:
            logger.info("[PIDQLearningAgent] Early termination enabled. Initializing parameters...")
            self.improvement_metric_source = self.early_termination_config_data.get('improvement_metric_source', "interval_cumulative_reward")
            if self.improvement_metric_source not in ["interval_cumulative_reward", "interval_avg_stability"]:
                logger.warning(f"Invalid improvement_metric_source: {self.improvement_metric_source}. Defaulting to 'interval_cumulative_reward'.")
                self.improvement_metric_source = "interval_cumulative_reward"

            self.patience_type = self.early_termination_config_data.get('patience_type', 'fixed')
            self.min_patience_M_min = int(self.early_termination_config_data.get('min_patience', 5))
            if self.min_patience_M_min < 1:
                logger.warning(f"min_patience ({self.min_patience_M_min}) < 1. Setting to 1.")
                self.min_patience_M_min = 1
            self.max_patience_M_max = int(self.early_termination_config_data.get('max_patience', 50))
            self.patience_adjustment_rate_eta = float(self.early_termination_config_data.get('patience_adjustment_rate', 1.0))
            self.use_stage_learning_for_base_patience = self.early_termination_config_data.get('use_stage_learning_for_base_patience', False)

            penalty_params = self.early_termination_config_data.get('penalty_reward_params', {})
            self.penalty_enabled = penalty_params.get('enabled', False) if isinstance(penalty_params, dict) else False
            
            default_global_patience = 20
            default_global_beta = 0.9

            # Poblar initial_patience_M0_from_config y initial_penalty_beta0_from_config
            for adv_key in self.agent_defining_vars:
                config_key_patience = f'initial_patience_{adv_key}'
                self.initial_patience_M0_from_config[adv_key] = int(self.early_termination_config_data.get(config_key_patience, default_global_patience))

                config_key_beta = f'penalty_beta_init_{adv_key}'
                # Solo leer de penalty_params si penalty_params es un dict y existe
                if penalty_params and isinstance(penalty_params, dict): 
                     self.initial_penalty_beta0_from_config[adv_key] = float(penalty_params.get(config_key_beta, default_global_beta))
                else: # Si no hay penalty_params o no es dict, usar default global
                     self.initial_penalty_beta0_from_config[adv_key] = default_global_beta
            
            # Ahora usar los diccionarios poblados
            for adv in self.agent_defining_vars:
                self.patience_M[adv] = self.initial_patience_M0_from_config.get(adv, default_global_patience)
                self.no_improvement_counter_c_hat[adv] = 0
                self.penalty_beta[adv] = self.initial_penalty_beta0_from_config.get(adv, default_global_beta)
                self.last_interval_improvement_metric_m_bar[adv] = -np.inf # Correcto: inicializar a muy bajo
            
            self._last_et_metrics = {
                adv: {
                    'patience_M': self.patience_M.get(adv, np.nan),
                    'c_hat': 0,
                    'beta': self.penalty_beta.get(adv, default_global_beta) if self.penalty_enabled else 1.0,
                    'current_metric': np.nan,
                    'last_metric': -np.inf, # Reflejar el estado inicial
                    'requested_et': False
                } for adv in self.agent_defining_vars
            }
        else:
            logger.info("[PIDQLearningAgent] Early termination disabled.")
            # Asegurar que _last_et_metrics se inicialice con NaNs/defaults incluso si ET está deshabilitado
            self._last_et_metrics = {
                adv: { 'patience_M': np.nan, 'c_hat': np.nan, 'beta': 1.0, 
                       'current_metric': np.nan, 'last_metric': np.nan, 'requested_et': False }
                for adv in self.agent_defining_vars
            }
        
        # FIN _init_ PIDQLearningAgent
        logger.info("[PIDQLearningAgent] Initialization complete.")

    # --- MÉTODOS PRIVADOS ---
    
    def _validate_and_prepare_state_config(self, config: Dict, enabled_key_name: str) -> Dict:
        logger.debug(f"[PIDQLearningAgent:_validate_state_cfg] Validating state_config using key '{enabled_key_name}'. Input config keys: {list(config.keys())}")
        validated_config = {}
        if not isinstance(config, dict): 
            logger.error("[PIDQLearningAgent:_validate_state_cfg] Input state_config is not a dictionary.")
            raise ValueError("state_config must be a dictionary.")

        required_keys_if_used = ['min', 'max', 'bins']
        for var_name, cfg_dict in config.items():
            if not isinstance(cfg_dict, dict): 
                logger.error(f"[PIDQLearningAgent:_validate_state_cfg] Config for var '{var_name}' is not a dictionary.")
                raise ValueError(f"Config for '{var_name}' must be a dictionary.")

            current_var_cfg = cfg_dict.copy()
            
            # Original validation logic for min/max/bins if the variable is 'enabled_agent'
            # or part of another agent's 'state_vars'
            is_agent_defining_var = current_var_cfg.get(enabled_key_name, False)
            if not isinstance(is_agent_defining_var, bool): # Check type of enabled_key_name value
                logger.warning(f"[PIDQLearningAgent:_validate_state_cfg] Config '{var_name}': key '{enabled_key_name}' is not boolean ({type(is_agent_defining_var)}). Treating as False.")
                is_agent_defining_var = False

            is_part_of_any_state_def = False
            # Check if this var_name is listed in any 'state_vars' of an 'enabled_agent' variable
            for other_var_iter, other_cfg_iter in config.items():
                if isinstance(other_cfg_iter, dict) and other_cfg_iter.get(enabled_key_name, False):
                    state_vars_for_other = other_cfg_iter.get('state_vars', [])
                    if isinstance(state_vars_for_other, list) and var_name in state_vars_for_other:
                        is_part_of_any_state_def = True
                        break
            
            # A variable needs min/max/bins if it defines an agent OR is part of an enabled agent's state definition
            if is_agent_defining_var or is_part_of_any_state_def:
                missing_keys = [key for key in required_keys_if_used if key not in current_var_cfg]
                if missing_keys: 
                    logger.error(f"[PIDQLearningAgent:_validate_state_cfg] Config for state variable '{var_name}' (used in an agent's state) is missing keys: {missing_keys}.")
                    raise ValueError(f"Config for state variable '{var_name}' (used in an agent's state) is missing keys: {missing_keys}.")
                
                # Validate types of min, max, bins
                min_val, max_val, bins_val = current_var_cfg.get('min'), current_var_cfg.get('max'), current_var_cfg.get('bins')
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    logger.error(f"[PIDQLearningAgent:_validate_state_cfg] Config '{var_name}': 'min'/'max' must be numeric.")
                    raise ValueError(f"Config '{var_name}': 'min'/'max' must be numeric.")
                if min_val >= max_val:
                    logger.error(f"[PIDQLearningAgent:_validate_state_cfg] Config '{var_name}': 'min' ({min_val}) must be < 'max' ({max_val}).")
                    raise ValueError(f"Config '{var_name}': 'min' ({min_val}) must be < 'max' ({max_val}).")
                if not isinstance(bins_val, int) or bins_val <= 0:
                    logger.error(f"[PIDQLearningAgent:_validate_state_cfg] Config '{var_name}': 'bins' ({bins_val}) must be a positive integer.")
                    raise ValueError(f"Config '{var_name}': 'bins' ({bins_val}) must be a positive integer.")
            
            validated_config[var_name] = current_var_cfg
        
        logger.debug(f"[PIDQLearningAgent:_validate_state_cfg] state_config validation passed. Output config keys: {list(validated_config.keys())}")
        return validated_config

    def _initialize_tables(self):
        logger.info("[PIDQLearningAgent:_initialize_tables] Initializing NumPy tables (Q, N, Auxiliary, Aux_N)...") # Log actualizado
        self.ordered_state_vars_for_agent_var = {}
        self.q_tables_np = {}
        self.visit_counts_np = {}
        
        self.auxiliary_tables_info = {} # Reinicializar para evitar duplicados si se llama varias veces
        self.aux_visit_counts_np = {}   # Reinicializar

        for aux_table_name in self.reward_strategy.required_auxiliary_tables:
            self.auxiliary_tables_info[aux_table_name] = {}
            # Usar un nombre distinto para la clave de la tabla de visitas, e.g., 'baseline_visits'
            self.aux_visit_counts_np[f"{aux_table_name}_visits"] = {}


        for agent_defining_var in self.agent_defining_vars:
            # Ordenar variables para cada agente
            ordered_vars_for_this_table = self._get_ordered_vars_for_agent_defining_var(agent_defining_var)
            
            logger.info(f"[PIDQLearningAgent:_initialize_tables] Initializing tables for agent defined by '{agent_defining_var}' with state vars: {ordered_vars_for_this_table}")
            self.ordered_state_vars_for_agent_var[agent_defining_var] = ordered_vars_for_this_table

            state_dims = []
            for var_name_in_state in ordered_vars_for_this_table:
                var_detail_config = self.state_config.get(var_name_in_state)
                bins = var_detail_config.get('bins')
                state_dims.append(bins)

            q_visit_shape = tuple(state_dims + [self.num_actions])
            aux_table_shape = tuple(state_dims) 

            self.q_tables_np[agent_defining_var] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
            self.visit_counts_np[agent_defining_var] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)
            logger.debug(f"[PIDQLearningAgent:_initialize_tables] Q/N Table for '{agent_defining_var}' shape: {q_visit_shape}")

            for aux_table_name in self.reward_strategy.required_auxiliary_tables:
                init_val = 0.0
                if aux_table_name == 'baseline':
                    init_val = self.baseline_init_value
                
                self.auxiliary_tables_info[aux_table_name][agent_defining_var] = np.full(aux_table_shape, init_val, dtype=np.float32)
                logger.debug(f"[PIDQLearningAgent:_initialize_tables] Aux Table '{aux_table_name}' for '{agent_defining_var}' shape: {aux_table_shape}, init_val: {init_val}")

                # Inicializar la tabla de cuentas de visita para esta tabla auxiliar
                visit_key_for_aux = f"{aux_table_name}_visits"
                if visit_key_for_aux in self.aux_visit_counts_np: # Debe existir por la inicialización anterior
                    self.aux_visit_counts_np[visit_key_for_aux][agent_defining_var] = np.full(aux_table_shape, self.visit_init_value, dtype=np.int32) # Usar self.visit_init_value? o siempre 0?
                    logger.debug(f"[PIDQLearningAgent:_initialize_tables] Aux Visit Count Table '{visit_key_for_aux}' for '{agent_defining_var}' shape: {aux_table_shape}, init_val: 0")
                else:
                    logger.error(f"[PIDQLearningAgent:_initialize_tables] LOGIC ERROR: Visit key '{visit_key_for_aux}' not found in self.aux_visit_counts_np. This should not happen.")

    def _get_ordered_vars_for_agent_defining_var(self, agent_defining_var: str) -> List[str]:
        # Su config está en self.state_config[agent_defining_var]
        agent_var_config = self.state_config.get(agent_defining_var)

        ordered_vars = OrderedDict()
        ordered_vars[agent_defining_var] = True # La variable base siempre es parte de su propio estado

        additional_state_vars_names = agent_var_config.get('state_vars', [])

        for var_name_from_state_vars_list in additional_state_vars_names:
            ordered_vars[var_name_from_state_vars_list] = True
        
        final_list = list(ordered_vars.keys())
        return final_list

    def _discretize_value(self, value: float, var_name: str) -> Optional[int]:
        # La lógica de si esta variable forma parte del estado de un agente específico está en otra parte.
        config = self.state_config[var_name]
        min_val = config.get('min')
        max_val = config.get('max')
        bins = config.get('bins')
        
        clipped_value = np.clip(value, min_val, max_val)
        if bins == 1:
            return 0
        if np.isclose(min_val, max_val): # Si min y max son iguales (después de clip, todo valor será igual)
            return 0
        
        # Método original con representative_points (centros de bins implícitos por linspace)
        representative_points = np.linspace(min_val, max_val, bins)
        closest_point_index = int(np.argmin(np.abs(representative_points - clipped_value)))
        
        return closest_point_index

    def get_discrete_state_indices_tuple(self, agent_state_dict: Optional[Dict[str, Any]], agent_defining_var: str) -> Optional[tuple]:
        # agent_defining_var es la variable que define esta Q-table (e.g., 'kp')
        # Obtener la lista de TODAS las variables que componen el estado de esta Q-table
        vars_for_this_table_state = self.ordered_state_vars_for_agent_var[agent_defining_var]
        indices = []
        for var_name_in_state in vars_for_this_table_state:
            value = agent_state_dict[var_name_in_state]
            # _discretize_value usa self.state_config[var_name_in_state] para obtener min/max/bins
            index = self._discretize_value(value, var_name_in_state)
            indices.append(index)
        return tuple(indices)
    
    # --- MÉTODOS PRINCIPALES ---

    def select_action(self, agent_state_dict: Dict[str, Any]) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        perform_exploration = np.random.rand() < self._epsilon

        logger.debug(f"[PIDQLearningAgent:select_action] Epsilon: {self._epsilon:.4f}, Perform Exploration: {perform_exploration}, Num Actions Config: {self.num_actions}")

        for agent_var in self.agent_defining_vars:
            action_index = 1 # Default: maintain
            if agent_var in self.q_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, agent_var)
                if state_indices is not None:
                    if perform_exploration:
                        # Aquí es donde se elige aleatoriamente
                        chosen_random_action = np.random.randint(0, self.num_actions) # Asegurar que el límite superior es exclusivo
                        action_index = chosen_random_action
                    else:
                        q_values_for_state = self.q_tables_np[agent_var][state_indices]
                        max_q = np.nanmax(q_values_for_state)
                        if pd.isna(max_q): 
                            action_index = np.random.randint(0, self.num_actions) # Límite superior exclusivo
                        else:
                            best_actions = np.where(np.isclose(q_values_for_state, max_q))[0]
                            action_index = int(np.random.choice(best_actions))
            
            actions[agent_var] = action_index
        
        logger.debug(f"[PIDQLearningAgent:select_action] Final actions selected: {actions}")
        return actions

    def learn(self,
              current_agent_state_dict: Dict[str, Any],
              actions_dict: Dict[str, int], # actions_dict['kp'], actions_dict['ki'], etc.
              reward_info: Union[float, Tuple[float, float], Dict[str, float]],
              next_agent_state_dict: Dict[str, Any],
              controller: Controller,
              done: bool):

        _interval_reward_val = np.nan; _avg_w_stab_val = np.nan
        if isinstance(reward_info, tuple) and len(reward_info) == 2:
            _interval_reward_val, _avg_w_stab_val = reward_info
            _interval_reward_val = float(_interval_reward_val) if np.isfinite(_interval_reward_val) else 0.0
            _avg_w_stab_val = float(_avg_w_stab_val) if np.isfinite(_avg_w_stab_val) else 1.0
        elif isinstance(reward_info, (float,int)):
             _interval_reward_val = float(reward_info) if np.isfinite(reward_info) else 0.0
             _avg_w_stab_val = 1.0

        self._last_td_errors = {adv: np.nan for adv in self.agent_defining_vars}

        # Primero, calculamos el R_learn original usando la estrategia
        raw_reward_for_q_update_dict: Dict[str, float] = {}
        for agent_var_strat_pass in self.agent_defining_vars:
            if agent_var_strat_pass not in self.q_tables_np: continue
            current_state_indices_strat_pass = self.get_discrete_state_indices_tuple(current_agent_state_dict, agent_var_strat_pass)
            if current_state_indices_strat_pass is None: continue
            action_taken_strat_pass = actions_dict.get(agent_var_strat_pass)
            if action_taken_strat_pass is None: continue
            
            raw_reward_for_q_update_dict[agent_var_strat_pass] = self.reward_strategy.compute_reward_for_learning(
                gain=agent_var_strat_pass, agent=self, controller=controller,
                current_agent_state_dict=current_agent_state_dict, current_state_indices=current_state_indices_strat_pass,
                actions_dict=actions_dict, action_taken_idx=action_taken_strat_pass,
                interval_reward=_interval_reward_val, avg_w_stab=_avg_w_stab_val,
                reward_dict=reward_info if isinstance(reward_info, dict) else None
            )

        # Ahora, aplicar lógica de early termination y penalización
        effective_done_for_agent: Dict[str, bool] = {adv: done for adv in self.agent_defining_vars}

        if self.early_termination_enabled:
            for agent_var_et in self.agent_defining_vars:
                if agent_var_et not in self.q_tables_np: continue

                current_interval_metric_value = 0.0
                if self.improvement_metric_source == "interval_avg_stability":
                    current_interval_metric_value = _avg_w_stab_val
                elif self.improvement_metric_source == "interval_cumulative_reward":
                    current_interval_metric_value = _interval_reward_val
                
                # m_bar_c_n es la métrica de mejora intervalo actual.
                m_bar_c_n = current_interval_metric_value 
                
                # Acumular m_bar_c_n para el promedio del episodio (para ajustar M en reset_agent)
                self.current_interval_improvement_deltas[agent_var_et].append(m_bar_c_n)

                previous_metric_for_comparison = self.last_interval_improvement_metric_m_bar[agent_var_et]

                # Compara con la métrica del intervalo de decisión *anterior*
                if pd.notna(m_bar_c_n) and m_bar_c_n > previous_metric_for_comparison: # *Se considera la métrica de evaluación como mayor a la anterior, ojo con nuevas métricas.
                    self.no_improvement_counter_c_hat[agent_var_et] = 0
                    if self.penalty_enabled: # Resetear beta solo si la penalización está activa
                        self.penalty_beta[agent_var_et] = self.initial_penalty_beta0_from_config.get(agent_var_et, 0.9) # Usar el valor correcto
                    # logger.debug(f"[PIDQLearningAgent:learn ET] Gain {agent_var_et}: Improvement detected. Reset c_hat. Metric: {m_bar_c_n:.4f} > Prev: {previous_metric_for_comparison:.4f}")
                else:
                    if pd.notna(m_bar_c_n):
                        self.no_improvement_counter_c_hat[agent_var_et] += 1
                    # logger.debug(f"[PIDQLearningAgent:learn ET] Gain {agent_var_et}: No improvement or invalid metric. c_hat: {self.no_improvement_counter_c_hat[agent_var_et]}. Metric: {m_bar_c_n:.4f}, Prev: {previous_metric_for_comparison:.4f}")


                self.last_interval_improvement_metric_m_bar[agent_var_et] = m_bar_c_n

                patience_for_gain = self.patience_M.get(agent_var_et, 20)
                if self.no_improvement_counter_c_hat[agent_var_et] >= patience_for_gain:
                    self._request_early_termination_flags[agent_var_et] = True
                    effective_done_for_agent[agent_var_et] = True # Marcar como "done" para este agente
                    logger.info(f"[PIDQLearningAgent:learn ET] Agent {agent_var_et}: Early termination triggered. c_hat ({self.no_improvement_counter_c_hat[agent_var_et]}) >= M ({patience_for_gain}).")
                
                # Aplicar penalización si no se mejoró Y la terminación temprana no se activó AÚN para este agente
                if not self._request_early_termination_flags[agent_var_et] and \
                m_bar_c_n <= self.last_interval_improvement_metric_m_bar[agent_var_et] and \
                self.penalty_enabled:
                    
                    current_beta = self.penalty_beta.get(agent_var_et, self.initial_penalty_beta0_from_config.get(agent_var_et,0.9))
                    # Asegurar que la paciencia no sea cero para evitar división por cero
                    if patience_for_gain > 0 :
                        new_beta_factor = (patience_for_gain - self.no_improvement_counter_c_hat[agent_var_et]) / patience_for_gain
                        self.penalty_beta[agent_var_et] = current_beta * max(0.0, new_beta_factor) # Beta no puede ser negativo
                    else: # Si la paciencia es 0, la penalización podría ser 0 o mantenerse, aquí la mantenemos.
                        self.penalty_beta[agent_var_et] = current_beta

                    original_reward = raw_reward_for_q_update_dict.get(agent_var_et, 0.0)
                    penalized_reward = original_reward * self.penalty_beta[agent_var_et]
                    raw_reward_for_q_update_dict[agent_var_et] = penalized_reward # Sobrescribir con la recompensa penalizada
                    # logger.debug(f"[PIDQLearningAgent:learn ET] Gain {agent_var_et}: Penalized reward. Beta: {self.penalty_beta[agent_var_et]:.3f}. R_learn: {penalized_reward:.4f} (Original: {original_reward:.4f})")
                
                # Actualizar _last_et_metrics
                self._last_et_metrics[agent_var_et] = {
                    'patience_M': patience_for_gain,
                    'c_hat': self.no_improvement_counter_c_hat[agent_var_et],
                    'beta': self.penalty_beta.get(agent_var_et, self.initial_penalty_beta0_from_config.get(agent_var_et, 0.9)) if self.penalty_enabled else 1.0,
                    'current_metric': m_bar_c_n, # El valor de este intervalo
                    'last_metric': previous_metric_for_comparison, # El valor del intervalo anterior con el que se comparó
                    'requested_et': self._request_early_termination_flags[agent_var_et]
                }

        # Iterar sobre cada "agente" (variable con Q-table) para actualizarlo
        for agent_var in self.agent_defining_vars:
            if agent_var not in self.q_tables_np: continue # Si no tiene tabla Q, no aprende
            try:
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, agent_var)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, agent_var)
                if current_state_indices is None:
                    logger.warning(f"[PIDQLearningAgent:learn] Cannot get S indices for agent '{agent_var}'. Skipping update."); continue

                # La acción para ESTE agente (agent_var) se toma de actions_dict
                action_taken_for_this_agent_var = actions_dict.get(agent_var)
                if action_taken_for_this_agent_var is None or not (0 <= action_taken_for_this_agent_var < self.num_actions):
                    logger.warning(f"[PIDQLearningAgent:learn] Invalid action index ({action_taken_for_this_agent_var}) for agent '{agent_var}'. Skipping."); continue

                # Usar la recompensa (posiblemente penalizada) del diccionario
                reward_for_q_update = raw_reward_for_q_update_dict.get(agent_var, 0.0)
                if pd.isna(reward_for_q_update) or not np.isfinite(reward_for_q_update):
                    logger.warning(f"[PIDQLearningAgent:learn] RewardStrategy returned invalid R_learn ({reward_for_q_update}) for agent '{agent_var}'. Using 0.")
                    reward_for_q_update = 0.0

                full_index_current = current_state_indices + (action_taken_for_this_agent_var,)
                current_q = self.q_tables_np[agent_var][full_index_current]

                # Usar effective_done_for_agent[agent_var]
                is_terminal_for_this_agent = effective_done_for_agent.get(agent_var, done)

                max_next_q = 0.0
                if not is_terminal_for_this_agent: # Si no es terminal para ESTE agente
                    if next_state_indices is None:
                        logger.warning(f"[PIDQLearningAgent:learn] Not 'done' for '{agent_var}' but no S' indices. Using Q(S')=0.")
                    else:
                        next_q_values = self.q_tables_np[agent_var][next_state_indices]
                        max_next_q_raw = np.nanmax(next_q_values)
                        max_next_q = 0.0 if pd.isna(max_next_q_raw) or not np.isfinite(max_next_q_raw) else max_next_q_raw

                # Actualización de Q-Learning
                if is_terminal_for_this_agent: # Fórmula para terminación
                    td_target = reward_for_q_update 
                else: # Fórmula estándar
                    td_target = reward_for_q_update + self.discount_factor * max_next_q
                
                td_error = td_target - current_q
                self._last_td_errors[agent_var] = float(td_error)
                new_q_value = current_q + self._learning_rate * td_error

                if pd.notna(new_q_value) and np.isfinite(new_q_value):
                    self.q_tables_np[agent_var][full_index_current] = new_q_value
                else: logger.warning(f"[PIDQLearningAgent:learn] Invalid new Q-value ({new_q_value}) for agent '{agent_var}'. Q not updated.")
                self.visit_counts_np[agent_var][full_index_current] += 1

            except IndexError as e: logger.error(f"[PIDQLearningAgent:learn] IndexError for agent '{agent_var}'. S_idx: {current_state_indices}, A_idx: {action_taken_for_this_agent_var}. Q Shape: {self.q_tables_np.get(agent_var, np.array([])).shape}. Error: {e}", exc_info=True)
            except KeyError as e: logger.error(f"[PIDQLearningAgent:learn] KeyError for agent '{agent_var}' (poss. in RewardStrategy): {e}.", exc_info=True)
            except Exception as e: logger.error(f"[PIDQLearningAgent:learn] Unexpected error for agent '{agent_var}': {e}.", exc_info=True)
    
    def reset_agent(self):
        """Actualiza patience, epsilon y learning rate al final de un episodio."""
        if self.use_epsilon_decay:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        if self.use_learning_rate_decay:
            self._learning_rate = max(self._learning_rate_min, self._learning_rate * self._learning_rate_decay)
        
        self._last_td_errors = {adv: np.nan for adv in self.agent_defining_vars}

        # Inicializar aquí para que esté definida incluso si ET está deshabilitado
        current_episode_avg_improvements: Dict[str, float] = {} 

        if self.early_termination_enabled:
            # Calcular el promedio de mejora del episodio que acaba de terminar.
            for adv in self.agent_defining_vars:
                if self.current_interval_improvement_deltas.get(adv): # Usar .get() para seguridad
                    current_episode_avg_improvements[adv] = np.mean(self.current_interval_improvement_deltas[adv])
                else:
                    current_episode_avg_improvements[adv] = 0.0 

                if self.patience_type == 'adaptive':
                    delta_m_bar_e = current_episode_avg_improvements.get(adv, 0.0) - self.last_avg_episode_improvement_metric.get(adv, 0.0)
                    
                    # Paciencia Adaptativa - avanzada - según stage learning (epsilon-greddy)
                    m_base = self.patience_M[adv] 
                    if self.use_stage_learning_for_base_patience:
                        m_base = self.max_patience_M_max * self.epsilon + self.min_patience_M_min * (1 - self.epsilon)
                    # Paciencia Adaptativa - simple - según mejoría
                    new_m = m_base + self.patience_adjustment_rate_eta * np.sign(delta_m_bar_e)

                    self.patience_M[adv] = int(np.clip(new_m, self.min_patience_M_min, self.max_patience_M_max))
                    logger.debug(f"[PIDQLearningAgent:reset_agent] Gain {adv}: Patience M adjusted to {self.patience_M[adv]} (delta_m_bar_e: {delta_m_bar_e:.4f}, base_M: {m_base:.1f})")

                # Reset para el próximo episodio (esto ocurre tanto para adaptive como para fixed patience type)
                self.no_improvement_counter_c_hat[adv] = 0
                # Solo resetear beta si la penalización estaba habilitada
                self.penalty_beta[adv] = self.initial_penalty_beta0_from_config.get(adv, 0.9) if self.penalty_enabled else 1.0
                
                self.last_interval_improvement_metric_m_bar[adv] = -np.inf
                self.current_interval_improvement_deltas[adv] = [] 
                self.last_avg_episode_improvement_metric[adv] = current_episode_avg_improvements.get(adv, 0.0)
        
        # Resetear los flags de terminación temprana y _last_et_metrics para el nuevo episodio,
        # independientemente de si ET estaba habilitado o no, para mantener un estado consistente.
        self._request_early_termination_flags = {adv: False for adv in self.agent_defining_vars}
        default_beta_val = 1.0 # Si ET está deshabilitado, beta es efectivamente 1.0

        self._last_et_metrics = {
            adv: {'patience_M': self.patience_M.get(adv, np.nan), 
                  'c_hat': 0, 
                  'beta': self.penalty_beta.get(adv, default_beta_val), # Usar el beta reseteado
                  'current_metric': np.nan, 
                  'last_metric': 0.0, 
                  'requested_et': False}
            for adv in self.agent_defining_vars
        }
            
        logger.debug(f"[PIDQLearningAgent:reset_agent] Epsilon={self.epsilon:.4f}, LR={self.learning_rate:.4f}")

    def build_agent_state(self, raw_state_vector: Any, controller: Controller, state_config_for_build: Dict) -> Dict[str, Any]:
        # state_config_for_build es self.state_config (ya validado en __init__)
        agent_s: Dict[str, Any] = {}
        #logger.debug(f"[PIDQLearningAgent:build_agent_state] Building agent state. Input raw_state_vector: {np.round(raw_state_vector,3) if isinstance(raw_state_vector, np.ndarray) else raw_state_vector}")
        #logger.debug(f"[PIDQLearningAgent:build_agent_state] Using self.state_config with keys: {list(self.state_config.keys())}")

        state_vector_map = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        current_controller_gains: Dict[str, float] = {}

        current_controller_gains = controller.get_params()
        #logger.debug(f"[PIDQLearningAgent:build_agent_state] Controller params obtained: {current_controller_gains}")

        for var_name, var_cfg_details in self.state_config.items():
            current_value_for_var = np.nan

            if var_name in state_vector_map:
                state_idx = state_vector_map[var_name]
                if isinstance(raw_state_vector, (list, tuple, np.ndarray)) and len(raw_state_vector) > state_idx:
                    raw_val = raw_state_vector[state_idx]
                    if isinstance(raw_val, (int, float)) and np.isfinite(raw_val):
                        current_value_for_var = float(raw_val)
                        #logger.debug(f"[PIDQLearningAgent:build_agent_state] Got system value for '{var_name}': {current_value_for_var}")
            elif var_name in ['kp', 'ki', 'kd']:
                gain_val = current_controller_gains.get(var_name)
                if isinstance(gain_val, (int, float)) and np.isfinite(gain_val):
                    current_value_for_var = float(gain_val)
                    #logger.debug(f"[PIDQLearningAgent:build_agent_state] Got controller gain value for '{var_name}': {current_value_for_var}")
            # Validación numérica y no NaN
            if pd.notna(current_value_for_var) and np.isfinite(current_value_for_var):
                agent_s[var_name] = current_value_for_var

        logger.debug(f"[PIDQLearningAgent:build_agent_state] Constructed agent_state_dict with Content: {agent_s}")
        return agent_s
    
    
    # --- MÉTODOS SECUNDARIOS ---
    
    @property
    def reward_strategy(self) -> RewardStrategy: return self._reward_strategy_instance
    @property
    def epsilon(self) -> float: return self._epsilon
    @property
    def learning_rate(self) -> float: return self._learning_rate

    def get_agent_defining_vars(self) -> List[str]:
        return self.agent_defining_vars
    
    def should_episode_terminate_early(self) -> bool:
        if not self.early_termination_enabled:
            return False
        # Terminar si CUALQUIER agente lo solicita
        return any(self._request_early_termination_flags.values())
    
    def get_last_early_termination_metrics(self) -> Dict[str, Dict[str, Any]]:
        return self._last_et_metrics

    # --- MÉTODOS AUXILIARES ---

    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        logger.info("[PIDQLearningAgent:get_agent_state_for_saving] Structuring agent state...")
        agent_state_save: Dict[str, Any] = {"q_tables": {}, "visit_counts": {}}
        
        # Añadir espacio para tablas auxiliares y sus visitas
        for aux_table_name_base in self.reward_strategy.required_auxiliary_tables: # ej: 'baseline'
            agent_state_save[f"{aux_table_name_base}_tables"] = {} 
            agent_state_save[f"{aux_table_name_base}_visit_counts"] = {} # ej: "baseline_visit_counts"

        float_precision = 6
        for agent_var, ordered_vars_for_table in self.ordered_state_vars_for_agent_var.items():
            q_list, visit_list = [], []
            # Diccionario para listas de tablas auxiliares para este agent_var
            aux_lists_for_agent_var: Dict[str, List[Dict]] = {
                name: [] for name in self.reward_strategy.required_auxiliary_tables
            }
            # Diccionario para listas de cuentas de visita de tablas auxiliares
            aux_visit_lists_for_agent_var: Dict[str, List[Dict]] = {
                f"{name}_visits": [] for name in self.reward_strategy.required_auxiliary_tables
            }

            if agent_var not in self.q_tables_np or agent_var not in self.visit_counts_np:
                 logger.warning(f"[PIDQLearningAgent:save_state] Missing Q or N tables for agent '{agent_var}'. Skipping."); continue
            
            q_table = self.q_tables_np[agent_var]
            visit_table = self.visit_counts_np[agent_var]
            
            state_shape, num_actions_table = q_table.shape[:-1], q_table.shape[-1]
            state_points_cache = {}
            valid_cache = True
            for var_name_in_state in ordered_vars_for_table: 
                try:
                    cfg = self.state_config[var_name_in_state] 
                    min_v, max_v, bins = cfg['min'], cfg['max'], cfg['bins']
                    if bins > 0: state_points_cache[var_name_in_state] = np.linspace(min_v, max_v, bins)
                    else: raise ValueError(f"Invalid bins ({bins}) for '{var_name_in_state}'")
                except Exception as e: logger.error(f"[PIDQLearningAgent:save_state] Error calculating representative points for '{var_name_in_state}': {e}"); valid_cache = False; break
            if not valid_cache: logger.warning(f"[PIDQLearningAgent:save_state] Cannot calculate representative points for agent '{agent_var}'. Skipping save."); continue


            for state_indices_tuple in np.ndindex(state_shape):
                state_repr_dict = {}
                valid_repr = True
                try:
                    for i, var_name_in_state in enumerate(ordered_vars_for_table):
                        state_repr_dict[var_name_in_state] = round(state_points_cache[var_name_in_state][state_indices_tuple[i]], float_precision)
                except Exception as e: logger.error(f"[PIDQLearningAgent:save_state] Error generating state representation for indices {state_indices_tuple}, agent '{agent_var}': {e}"); valid_repr = False
                if not valid_repr: continue

                try:
                    # Q-table y Visit Counts (para Q-table)
                    q_row = state_repr_dict.copy()
                    for a in range(num_actions_table): q_row[f"action_{a}"] = float(q_table[state_indices_tuple][a])
                    q_list.append(q_row)
                    
                    visit_row = state_repr_dict.copy()
                    for a in range(num_actions_table): visit_row[f"action_{a}"] = int(visit_table[state_indices_tuple][a])
                    visit_list.append(visit_row)

                    # Guardar tablas auxiliares y SUS CUENTAS DE VISITA
                    for aux_table_name_base in self.reward_strategy.required_auxiliary_tables: # ej: 'baseline'
                        # Tabla auxiliar (ej: baseline)
                        if agent_var in self.auxiliary_tables_info.get(aux_table_name_base, {}):
                            aux_table = self.auxiliary_tables_info[aux_table_name_base][agent_var]
                            aux_row = state_repr_dict.copy()
                            aux_row[f"{aux_table_name_base}_value"] = float(aux_table[state_indices_tuple])
                            aux_lists_for_agent_var[aux_table_name_base].append(aux_row)
                        
                        # Cuentas de visita de la tabla auxiliar (ej: baseline_visits)
                        visit_key_for_aux = f"{aux_table_name_base}_visits" # ej: "baseline_visits"
                        if visit_key_for_aux in self.aux_visit_counts_np and \
                           agent_var in self.aux_visit_counts_np[visit_key_for_aux]:
                            
                            aux_visit_table = self.aux_visit_counts_np[visit_key_for_aux][agent_var]
                            aux_visit_row = state_repr_dict.copy()
                            # La tabla de visitas para auxiliares no tiene dimensión de acción
                            aux_visit_row[f"{aux_table_name_base}_visit_count"] = int(aux_visit_table[state_indices_tuple])
                            aux_visit_lists_for_agent_var[visit_key_for_aux].append(aux_visit_row)

                except IndexError: 
                    logger.error(f"[PIDQLearningAgent:save_state] IndexError structuring tables for state {state_indices_tuple}, agent '{agent_var}'."); 
                    continue

            agent_state_save["q_tables"][agent_var] = q_list
            agent_state_save["visit_counts"][agent_var] = visit_list
            
            for aux_table_name_base, data_list in aux_lists_for_agent_var.items():
                if data_list: 
                    agent_state_save[f"{aux_table_name_base}_tables"][agent_var] = data_list
            
            for aux_visit_key, data_list in aux_visit_lists_for_agent_var.items(): # aux_visit_key es "baseline_visits", etc.
                if data_list:
                    # La clave en agent_state_save será, por ejemplo, "baseline_visit_counts"
                    # Esto requiere que creemos `agent_state_save[f"{aux_table_name_base}_visit_counts"]` al inicio.
                    # `aux_visit_key` es `f"{name}_visits"`. El nombre de la tabla en JSON será `f"{name}_visit_counts"`
                    json_key_for_aux_visits = aux_visit_key.replace("_visits", "_visit_counts") # ej: baseline_visits -> baseline_visit_counts
                    agent_state_save[json_key_for_aux_visits][agent_var] = data_list


        if not agent_state_save["q_tables"]: 
            logger.warning("[PIDQLearningAgent:save_state] No Q-table data generated for saving.")
        logger.info("[PIDQLearningAgent:save_state] Agent state structuring complete.")
        return agent_state_save

    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        q_values_output: Dict[str, np.ndarray] = {}
        for agent_var in self.agent_defining_vars:
            q_vals_for_agent = np.full(self.num_actions, np.nan, dtype=np.float32)
            if agent_var in self.q_tables_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, agent_var)
                if state_indices is not None:
                    try: q_vals_for_agent = self.q_tables_np[agent_var][state_indices].astype(np.float32)
                    except IndexError: pass
                    except Exception as e: logger.warning(f"[PIDQLearningAgent:get_q_values] Error for agent '{agent_var}': {e}")
            q_values_output[agent_var] = q_vals_for_agent
        return q_values_output

    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        visit_counts_output: Dict[str, np.ndarray] = {}
        for agent_var in self.agent_defining_vars:
            visits_for_agent = np.full(self.num_actions, -1, dtype=np.int32)
            if agent_var in self.visit_counts_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, agent_var)
                if state_indices is not None:
                    try: visits_for_agent = self.visit_counts_np[agent_var][state_indices].astype(np.int32)
                    except IndexError: pass
                    except Exception as e: logger.warning(f"[PIDQLearningAgent:get_visit_counts] Error for agent '{agent_var}': {e}")
            visit_counts_output[agent_var] = visits_for_agent
        return visit_counts_output

    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        # Este método es un wrapper sobre get_auxiliary_table_value para 'baseline'
        baselines: Dict[str, float] = {}
        for agent_var in self.agent_defining_vars:
            val = self.get_auxiliary_table_value(table_name='baseline', gain=agent_var, state_indices=self.get_discrete_state_indices_tuple(agent_state_dict, agent_var)) # type: ignore
            baselines[agent_var] = val if val is not None else np.nan
        return baselines

    def get_last_td_errors(self) -> Dict[str, float]:
        return {k: (float(v) if pd.notna(v) and np.isfinite(v) else np.nan) for k, v in self._last_td_errors.items()}
    

    # --- Implementación de métodos de interfaz para tablas auxiliares ---
    
    def get_auxiliary_table_value(self, table_name: str, gain: str, state_indices: tuple) -> Optional[float]:
        if table_name not in self.auxiliary_tables_info or \
           gain not in self.auxiliary_tables_info[table_name] or \
           state_indices is None:
            return None
        try:
            val = self.auxiliary_tables_info[table_name][gain][state_indices]
            return float(val) if pd.notna(val) and np.isfinite(val) else None
        except IndexError:
            logger.warning(f"[PIDQLearningAgent:get_aux_val] IndexError for table '{table_name}', gain '{gain}', indices {state_indices}.")
            return None
        except Exception as e:
            logger.error(f"[PIDQLearningAgent:get_aux_val] Error accessing table '{table_name}', gain '{gain}': {e}", exc_info=True)
            return None

    def update_auxiliary_table_value(self, table_name: str, gain: str, state_indices: tuple, value: float):
        visit_key_for_aux = f"{table_name}_visits" # ej: "baseline_visits"

        if table_name not in self.auxiliary_tables_info or \
           gain not in self.auxiliary_tables_info[table_name] or \
           state_indices is None:
            logger.warning(f"[PIDQLearningAgent:update_aux_val] Cannot update table '{table_name}' for gain '{gain}': table or indices invalid.")
            return
        if not (pd.notna(value) and np.isfinite(value)):
            logger.warning(f"[PIDQLearningAgent:update_aux_val] Cannot update table '{table_name}' for gain '{gain}' with invalid value: {value}.")
            return
        try:
            self.auxiliary_tables_info[table_name][gain][state_indices] = value
            
            # Incrementar contador de visitas para esta tabla auxiliar y estado
            if visit_key_for_aux in self.aux_visit_counts_np and \
               gain in self.aux_visit_counts_np[visit_key_for_aux]:
                self.aux_visit_counts_np[visit_key_for_aux][gain][state_indices] += 1
            else:
                logger.warning(f"[PIDQLearningAgent:update_aux_val] Visit count table for '{visit_key_for_aux}', gain '{gain}' not found. Counts not incremented.")

        except IndexError:
            logger.error(f"[PIDQLearningAgent:update_aux_val] IndexError for table '{table_name}' (or its visits), gain '{gain}', indices {state_indices}.")
        except Exception as e:
            logger.error(f"[PIDQLearningAgent:update_aux_val] Error updating table '{table_name}' (or its visits), gain '{gain}': {e}", exc_info=True)

    def get_auxiliary_table_names(self) -> List[str]:
        return list(self.auxiliary_tables_info.keys())