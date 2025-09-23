# components/agents/pid_qlearning_agent.py
import numpy as np
import pandas as pd
import itertools
import logging
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy
from interfaces.controller import Controller
from collections import OrderedDict # Para mantener el orden de las variables de estado
from typing import Dict, Any, Optional, Tuple, Union, List

logger = logging.getLogger(__name__)

class PIDQLearningAgent(RLAgent):
    """
    A Q-learning agent designed to tune the gains (Kp, Ki, Kd) of a PID controller.

    This agent can operate in two modes, determined by the `state_config`:
    1.  **Multi-Agent Mode:** Each PID gain is controlled by an independent sub-agent
        with its own Q-table. This is activated when multiple gains have
        `enabled_agent: true`.
    2.  **Mono-Agent Mode:** A single, unified agent controls all PID gains. It uses
        one large Q-table with a combined state space and a composite action space.
        This is activated when exactly one gain has `enabled_agent: true` and
        includes the other gains in its `state_vars` or include others variables.

    The agent's state representation, action space, and learning logic adapt
    automatically based on the detected mode.
    """
    def __init__(self,
                 reward_strategy: Optional[RewardStrategy], # Ahora puede ser None inicialmente
                 state_config: Dict[str, Dict[str, Any]],
                 num_actions: int,
                 gain_delta: Union[float, Dict[str, float]], # Para logging/referencia
                 per_gain_delta: bool, # Para logging/referencia
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
                 aux_table_init_values: Optional[Dict[str, float]] = None,
                 early_stopping_criteria: Optional[Dict] = None
                 ):
        logger.info(f"[PIDQLearningAgent] Initializing with RewardStrategy: {type(reward_strategy).__name__}")

        # --- 1. Parameter Assignment & Initial Validation ---
        self._reward_strategy_instance = reward_strategy
        self.state_config = self._validate_and_prepare_state_config_structure(state_config)
        self.base_num_actions = int(num_actions)
        self.gain_delta_ref = gain_delta
        self.per_gain_delta_ref = per_gain_delta
        self.discount_factor = float(discount_factor)
        self._initial_epsilon, self._current_epsilon = float(epsilon), float(epsilon)
        self._epsilon_min, self._epsilon_decay = float(epsilon_min), float(epsilon_decay)
        self.use_epsilon_decay = bool(use_epsilon_decay)
        self._initial_learning_rate, self._current_learning_rate = float(learning_rate), float(learning_rate)
        self._learning_rate_min, self._learning_rate_decay = float(learning_rate_min), float(learning_rate_decay)
        self.use_learning_rate_decay = bool(use_learning_rate_decay)
        self.q_init_value, self.visit_init_value = float(q_init_value), int(visit_init_value)
        self.aux_table_init_values = aux_table_init_values if isinstance(aux_table_init_values, dict) else {}

        # --- 2. Mode Detection (Multi-Agent vs. Mono-Agent) ---
        self.agent_defining_vars: List[str] = [
            k for k, v in self.state_config.items() if isinstance(v, dict) and v.get("enabled_agent", False)
        ]
        if not self.agent_defining_vars:
            raise ValueError("[PIDQLearningAgent] No agent enabled in state_config. At least one gain must have 'enabled_agent: true'.")
        
        self.is_mono_agent_mode = len(self.agent_defining_vars) == 1
        
        if self.is_mono_agent_mode:
            logger.info("[PIDQLearningAgent] Operating in MONO-AGENT mode.")
            self.mono_agent_master_gain = self.agent_defining_vars[0]
            self.controlled_gains = self._determine_controlled_gains(self.mono_agent_master_gain)
            self.num_actions = self.base_num_actions ** len(self.controlled_gains)
            self._action_map = self._create_mono_agent_action_map()
            logger.info(f"[PIDQLearningAgent] Mono-agent controls: {self.controlled_gains}. Action space size: {self.num_actions}")
        else:
            logger.info(f"[PIDQLearningAgent] Operating in MULTI-AGENT mode for gains: {self.agent_defining_vars}")
            self.num_actions = self.base_num_actions
            self.controlled_gains = self.agent_defining_vars

        # --- 3. Pre-computation for Performance ---
        self._precompute_discretization_bins()

        # --- 4. Internal State Initialization ---
        self.ordered_state_vars_for_q_table: Dict[str, List[str]] = {}
        self.q_tables: Dict[str, np.ndarray] = {}
        self.visit_counts: Dict[str, np.ndarray] = {}
        self.aux_tables: Dict[str, Dict[str, np.ndarray]] = {}
        self.aux_visit_counts: Dict[str, Dict[str, np.ndarray]] = {}
        self._last_td_errors: Dict[str, float] = {adv: np.nan for adv in self.agent_defining_vars}

        # --- 5. Early Termination Setup ---
        self.early_stopping_config = early_stopping_criteria if isinstance(early_stopping_criteria, dict) else {}
        self._early_termination_enabled_flag = self.early_stopping_config.get('enabled', False)
        self._configure_early_termination_params()

        logger.info(f"[PIDQLearningAgent] Initialization complete. Agent gains: {self.agent_defining_vars}")

    # --- Private Initialization & Helper Methods ---
    
    def set_reward_strategy(self, strategy: RewardStrategy):
        """Implementación para la inyección tardía de la estrategia de recompensa."""
        if self._reward_strategy_instance is not None:
            logger.warning(f"[PIDQLearningAgent] Overwriting existing reward strategy.")
        self._reward_strategy_instance = strategy
        # Una vez que la estrategia está establecida, se pueden inicializar las tablas que dependen de ella.
        logger.info(f"[PIDQLearningAgent] RewardStrategy set to {type(strategy).__name__}. Initializing dependent tables.")
        self._initialize_tables_and_aux()

    def _validate_and_prepare_state_config_structure(self, config_to_val: Dict) -> Dict:
        """Valida la estructura básica de state_config. Lanza error si es inválida."""
        if not isinstance(config_to_val, dict):
            raise TypeError("state_config must be a dictionary.")
        for var_name, var_cfg in config_to_val.items():
            if not isinstance(var_cfg, dict):
                raise TypeError(f"Config for state variable '{var_name}' must be a dictionary.")
            is_agent_var = var_cfg.get("enabled_agent", False)
            is_used_in_state_elsewhere = any(
                var_name in other_agent_cfg.get('state_vars', [])
                for other_agent_cfg in config_to_val.values()
                if isinstance(other_agent_cfg, dict)
            )
            if is_agent_var or is_used_in_state_elsewhere:
                if not all(k in var_cfg for k in ['min', 'max', 'bins']):
                    raise ValueError(f"State variable '{var_name}' is missing 'min', 'max', or 'bins' in state_config.")
        return config_to_val
    
    def _precompute_discretization_bins(self) -> None:
        """
        Pre-calcula los valores discretos exactos que podrá tomar cada variable
        (incluyendo los extremos min y max).
        """
        self._bin_centers: Dict[str, np.ndarray] = {}
        self._n_bins: Dict[str, int] = {}
        self._bin_step: Dict[str, float] = {}
        self._bin_min: Dict[str, float] = {}
        for var_name, cfg in self.state_config.items():
            # Centros que INCLUYEN los extremos
            centers = np.linspace(cfg['min'], cfg['max'], cfg['bins'], dtype=np.float64)
            # Paso constante
            step = centers[1] - centers[0] if cfg['bins'] > 1 else 0.0
            # Guarda en los diccionarios
            self._bin_centers[var_name] = centers
            self._n_bins[var_name]      = cfg['bins']
            self._bin_step[var_name]    = step
            self._bin_min[var_name]     = cfg['min']
    
    def _determine_controlled_gains(self, master_gain: str) -> List[str]:
        gain_config = self.state_config.get(master_gain, {})
        controlled = OrderedDict([(master_gain, True)])
        for var in gain_config.get('state_vars', []):
            if var in ['kp', 'ki', 'kd']:
                controlled[var] = True
        return list(controlled.keys())
    
    def _create_mono_agent_action_map(self) -> Dict[int, Dict[str, int]]:
        action_ranges = [range(self.base_num_actions)] * len(self.controlled_gains)
        combined_actions = list(itertools.product(*action_ranges))
        mono_map_dict = {i: {f'action_{gain}': action for gain, action in zip(self.controlled_gains, actions_tuple)} for i, actions_tuple in enumerate(combined_actions)}
        return mono_map_dict

    def _initialize_tables_and_aux(self):
        """Inicializa Q-tables, N-tables, y tablas auxiliares requeridas por la estrategia."""
        logger.debug("[PIDQAgent:_initialize_tables_and_aux] Initializing all agent tables...")
        # Preparar contenedores para tablas auxiliares
        if self.reward_strategy is None:
            logger.warning("[PIDQAgent:_initialize_tables_and_aux] Attempted to initialize tables but reward strategy is not set. Aborting.")
            return
        for aux_name_req in self.reward_strategy.required_auxiliary_tables:
            self.aux_tables[aux_name_req] = {}
            self.aux_visit_counts[f"{aux_name_req}_visits"] = {}
        # Construir contenedores para tablas auxiliares
        for gain_name in self.agent_defining_vars:
            # Determinar las variables de estado para la Q-table de esta ganancia
            ordered_vars = self._determine_ordered_state_vars(gain_name)
            self.ordered_state_vars_for_q_table[gain_name] = ordered_vars
            # Calcular dimensiones de las tablas
            state_dims = [self.state_config[var_n]['bins'] for var_n in ordered_vars]
            q_table_shape_dims = tuple(state_dims + [self.num_actions])
            aux_table_shape_dims = tuple(state_dims) # Sin la dimensión de acción
            # Crear Q-table y N-table
            self.q_tables[gain_name] = np.full(q_table_shape_dims, self.q_init_value, dtype=np.float32)
            self.visit_counts[gain_name] = np.full(q_table_shape_dims, self.visit_init_value, dtype=np.int32)
            logger.debug(f"[PIDQAgent:_initialize_tables_and_aux] Q/N tables for '{gain_name}' shape: {q_table_shape_dims}, StateVars: {ordered_vars}")
            # Crear tablas auxiliares y sus contadores de visita
            for aux_table_name in self.reward_strategy.required_auxiliary_tables:
                initial_val_for_aux = self.aux_table_init_values.get(aux_table_name, 0.0) # Default a 0.0
                self.aux_tables[aux_table_name][gain_name] = np.full(aux_table_shape_dims, initial_val_for_aux, dtype=np.float32)
                aux_visit_key = f"{aux_table_name}_visits"
                self.aux_visit_counts[aux_visit_key][gain_name] = np.full(aux_table_shape_dims, self.visit_init_value, dtype=np.int32)
                logger.debug(f"[PIDQAgent:_initialize_tables_and_aux] AuxTable '{aux_table_name}' (and visits) for '{gain_name}' shape: {aux_table_shape_dims}, InitVal: {initial_val_for_aux}")
        if not self.q_tables:
            logger.warning("[PIDQAgent:_initialize_tables_and_aux] No Q-tables were initialized. Check 'enabled_agent' flags in state_config.")

    def _determine_ordered_state_vars(self, gain_name_context: str) -> List[str]:
        """Obtiene la lista ordenada de variables de estado para la Q-table de una ganancia."""
        gain_specific_config = self.state_config.get(gain_name_context, {})
        # La ganancia en sí misma es la primera variable de su estado
        ordered_vars_map = OrderedDict([(gain_name_context, True)])
        # Añadir otras variables de estado si están especificadas en 'state_vars'
        additional_state_vars = gain_specific_config.get('state_vars', [])
        for var_name in additional_state_vars:
            if var_name in self.state_config: # Asegurar que la variable adicional esté definida en state_config
                ordered_vars_map[var_name] = True
            else:
                logger.warning(f"[PIDQAgent:_determine_ordered_state_vars] State variable '{var_name}' (listed in state_vars for '{gain_name_context}') not found in global state_config. Ignoring.")
        return list(ordered_vars_map.keys())

    def _discretize_value(self, value: float, var_name: str) -> Optional[int]:
        """Devuelve el índice del centro más próximo, incluyendo min y max como idx extremos."""
        if pd.isna(value) or not np.isfinite(value):
            return None
        v_min   = self._bin_min[var_name]
        step    = self._bin_step[var_name]
        n_bins  = self._n_bins[var_name]
        # Clip para no salir de rango
        v_clipped = np.clip(value, v_min, v_min + step * (n_bins - 1))
        # Índice:   round( (v - min) / step )
        idx = int(round((v_clipped - v_min) / step))
        return max(0, min(idx, n_bins - 1)) # Salvaguarda final (por redondeo flotante)

    def _get_discrete_state_indices_tuple(self, agent_state_dict: Dict[str, Any], gain_context: str) -> Optional[tuple]:
        """ 
        Para cada variable en el contexto `gain_context`, obtiene 
        su índice de bin. Si alguno es None (NaN/Inf), devuelve None.
        """
        indices = []
        for var_name in self.ordered_state_vars_for_q_table[gain_context]:
            value = agent_state_dict.get(var_name)
            if value is None: 
                return None
            idx = self._discretize_value(value, var_name)
            if idx is None: 
                return None
            indices.append(idx)
        return tuple(indices)
    
    # Lógica Interna de Early Termination (simplificada y directa)

    def _configure_early_termination_params(self):
        """Configura los parámetros iniciales para Early Termination."""
        if not self._early_termination_enabled_flag:
            self._et_metrics_snapshot = {}
            return
        
        cfg = self.early_stopping_config
        self.et_improvement_metric_source = cfg.get('improvement_metric_source', "interval_cumulative_reward")
        self.et_patience_type = cfg.get('patience_type', 'fixed')
        self.et_min_patience = int(cfg.get('min_patience', 5))
        self.et_max_patience = int(cfg.get('max_patience', 50))
        self.et_patience_adj_rate_eta = float(cfg.get('patience_adjustment_rate', 1.0))
        
        penalty_cfg = cfg.get('penalty_reward_params', {})
        self.et_is_penalty_enabled = penalty_cfg.get('enabled', False)
        
        self.et_patience_counters, self.et_no_improvement_counters = {}, {}
        self.et_penalty_beta_factors, self.et_last_interval_improvement_metric = {}, {}
        self.et_last_episode_avg_improvement, self.et_initial_penalty_beta = {}, {}
        self._et_requested_flags = {g: False for g in self.agent_defining_vars}
        self.current_episode_improvement_metric_values = {g: [] for g in self.agent_defining_vars}

        for gain in self.agent_defining_vars:
            self.et_patience_counters[gain] = int(cfg.get(f'initial_patience_{gain}', 20))
            self.et_no_improvement_counters[gain] = 0
            self.et_initial_penalty_beta[gain] = float(penalty_cfg.get(f'penalty_beta_init_{gain}', 0.9))
            self.et_penalty_beta_factors[gain] = self.et_initial_penalty_beta[gain]
            self.et_last_interval_improvement_metric[gain] = -np.inf
            self.et_last_episode_avg_improvement[gain] = 0.0

        self._update_et_metrics_snapshot_for_reset() # Inicializar el snapshot

    def _update_et_counters_and_flags(self, gain: str, metric_val: float):
        """Actualiza contadores de no mejora y flags de terminación para una ganancia."""
        last_metric = self.et_last_interval_improvement_metric[gain]
        if pd.notna(metric_val) and metric_val > last_metric: # Hubo mejora
            self.et_no_improvement_counters[gain] = 0
            # Resetear beta de penalización si la penalización está activa y hubo mejora
            if self.et_is_penalty_enabled:
                self.et_penalty_beta_factors[gain] = self.et_initial_penalty_beta[gain]
        elif pd.notna(metric_val): # No hubo mejora (o igualó), y la métrica actual es válida
            self.et_no_improvement_counters[gain] += 1
        # Actualizar la "última métrica conocida" para la próxima comparación
        if pd.notna(metric_val):
            self.et_last_interval_improvement_metric[gain] = metric_val
        # Comprobar si se alcanza la paciencia
        if self.et_no_improvement_counters[gain] >= self.et_patience_counters[gain]:
            self._et_requested_flags[gain] = True
            # logger.info(f"[PIDQAgent ET Update] Agent '{gain}': Early Stop Flag SET. NoImprovement ({self.et_no_improvement_counters[gain]}) >= Patience ({self.et_patience_counters[gain]}).")
    
    def _apply_et_reward_penalty(self, gain: str, metric_val: float, r_learn: float) -> float:
        """Aplica penalización a R_learn si no hay mejora y ET no se activó aún."""
        # Esta función se llama solo si self.et_is_penalty_enabled es True y _et_requested_flags[gain] es False.
        last_metric = self.et_last_interval_improvement_metric[gain]

        if pd.notna(metric_val) and metric_val <= last_metric: # No hubo mejora
            patience = self.et_patience_counters[gain]
            counter = self.et_no_improvement_counters[gain]
            if patience > 0 and counter < patience:
                # Decaer beta proporcionalmente a cuánta paciencia queda
                decay = (patience - counter) / patience
                self.et_penalty_beta_factors[gain] *= max(0.0, decay)
            penalized_r_learn = r_learn * self.et_penalty_beta_factors[gain]
            # logger.debug(f"[PIDQAgent ET Penalty Apply] Agent '{gain}': No improvement. R_learn penalized from {r_learn:.4f} to {penalized_r_learn:.4f} (beta: {self.et_penalty_beta_factors[gain]:.4f})")
            return penalized_r_learn
        return r_learn

    def _update_et_metrics_snapshot_for_reset(self):
        """Actualiza el snapshot de métricas de ET después de un reset_agent."""
        if not self._early_termination_enabled_flag: 
            return # No hacer nada si ET está deshabilitado
        self._et_metrics_snapshot = {
            gain: {
                'patience_M': self.et_patience_counters.get(gain, np.nan),
                'c_hat': self.et_no_improvement_counters.get(gain, 0), # Debería ser 0 después del reset
                'beta': self.et_penalty_beta_factors.get(gain, 1.0), # Beta reseteado
                'requested_et_flag': self._et_requested_flags.get(gain, False) # Debería ser False
            } for gain in self.agent_defining_vars
        }

    # --- Métodos de la Interfaz RLAgent ---

    def build_agent_state(self, env_state_dict: Dict[str, Any], controller: Controller) -> Dict[str, Any]:
        """
        Constructs the agent's state dictionary by selecting relevant variables
        from the environment's state dictionary and the controller's parameters.
        """
        agent_s_dict = {}
        controller_gains = controller.get_params()
        
        # Iterar sobre TODAS las variables que el agente podría necesitar (definidas en su state_config).
        for var_name in self.state_config.keys():
            if var_name in env_state_dict:
                agent_s_dict[var_name] = env_state_dict[var_name]
            elif var_name in controller_gains:
                agent_s_dict[var_name] = controller_gains[var_name]
        # logger.debug(f"[PIDQAgent:build_agent_state] Built state: { {k:f'{v:.3f}' for k,v in agent_s_dict.items()} }")
        return agent_s_dict
    
    def select_action(self, current_agent_state_values: Dict[str, Any]) -> Dict[str, int]:
        perform_exploration = np.random.rand() < self._current_epsilon
        actions_map: Dict[str, int] = {}

        if self.is_mono_agent_mode: # Mono-Agent Mode
            master_gain = self.mono_agent_master_gain
            state_indices = self._get_discrete_state_indices_tuple(current_agent_state_values, master_gain)
            action_idx: int
            if state_indices is None or perform_exploration:
                action_idx = np.random.randint(0, self.num_actions)
            else:
                q_values = self.q_tables[master_gain][state_indices]
                max_q = np.nanmax(q_values)
                if np.isnan(max_q):
                    action_idx = np.random.randint(0, self.num_actions)
                else:
                    best_actions = np.where(np.isclose(q_values, max_q))[0]
                    action_idx = int(np.random.choice(best_actions))
            # logger.debug(f"[PIDQAgent:select_action] Epsilon={self._current_epsilon:.3f}, Explore={perform_exploration}, Actions={action_idx}")
            return self._action_map[action_idx] # ya es dict
        else: # Multi-Agent Mode
            for gain in self.agent_defining_vars:
                state_indices = self._get_discrete_state_indices_tuple(current_agent_state_values, gain)
                action_idx: int
                if state_indices is None or perform_exploration:
                    action_idx = np.random.randint(0, self.num_actions)
                else:
                    q_values = self.q_tables[gain][state_indices]
                    max_q = np.nanmax(q_values)
                    if np.isnan(max_q):
                        action_idx = np.random.randint(0, self.num_actions)
                    else:
                        best_actions = np.where(np.isclose(q_values, max_q))[0]
                        action_idx = int(np.random.choice(best_actions))
                actions_map[f'action_{gain}'] = action_idx
                # logger.debug(f"[PIDQAgent:select_action] Epsilon={self._current_epsilon:.3f}, Explore={perform_exploration}, Actions={actions_map}")
            return actions_map

    def learn(self,
              current_agent_s_dict: Dict[str, Any],     # S
              taken_actions_map: Dict[str, int],        # A
              reward_info: Dict[str, float],
              next_agent_s_prime_dict: Dict[str, Any],  # S'
              current_controller_instance: Controller,
              is_episode_done: bool                     # Flag de terminación del entorno
             ) -> Dict[str, float]:
        if self.reward_strategy is None:
            logger.error("[PIDQLearningAgent] Cannot learn without a reward strategy.")
            return {}
        
        self._last_td_errors.clear()
        learning_metrics = {} # Diccionario para devolver

        real_reward_interval = reward_info.get('interval_reward', 0.0)
        avg_stability_interval = reward_info.get('avg_stability_score_interval', 1.0)
        differential_rewards = reward_info.get('differential_rewards')

        for gain_name in self.agent_defining_vars:
            s_indices = self._get_discrete_state_indices_tuple(current_agent_s_dict, gain_name)
            s_prime_indices = self._get_discrete_state_indices_tuple(next_agent_s_prime_dict, gain_name)

            if self.is_mono_agent_mode:
                actions_tuple = tuple(taken_actions_map[f'action_{g}'] for g in self.controlled_gains)
                try:
                    action_idx = next(idx for idx, mapping in self._action_map.items() if tuple(mapping.values()) == actions_tuple)
                except StopIteration: continue
            else:
                action_idx = taken_actions_map.get(f'action_{gain_name}')
            
            if s_indices is None or action_idx is None: 
                continue # No se puede aprender sin S o A, pero no debería ser
            
            # 1. Calcular R_learn usando la RewardStrategy
            r_learn = self.reward_strategy.compute_reward_for_learning(
                gain_id=gain_name, agent_instance=self, controller_instance=current_controller_instance,
                current_agent_s_dict=current_agent_s_dict, current_s_indices=s_indices,
                actions_taken_map=taken_actions_map, action_idx_for_gain=action_idx,
                real_interval_reward=real_reward_interval,
                avg_interval_stability_score=avg_stability_interval,
                differential_rewards_map=differential_rewards
            )

            # Guardar r_learn para logging
            learning_metrics[f'r_learn_{gain_name}'] = r_learn

            # 2. Aplicar lógica de Early Termination (puede modificar r_learn_from_strategy y determinar 'done' para este agente)
            effective_done = is_episode_done
            if self._early_termination_enabled_flag:
                metric_source = self.et_improvement_metric_source
                metric_val = avg_stability_interval if "stability" in metric_source else real_reward_interval
                self._update_et_counters_and_flags(gain_name, metric_val)
                if self.et_is_penalty_enabled and not self._et_requested_flags[gain_name]:
                    r_learn = self._apply_et_reward_penalty(gain_name, metric_val, r_learn)
                if self._et_requested_flags[gain_name]:
                    effective_done = True # Si el agente pide terminar, es 'done' para Q-update
            
            # 3. Actualización Q-Learning
            q_table = self.q_tables[gain_name]
            current_q = q_table[s_indices + (action_idx,)]
            max_q_next = 0.0
            if not effective_done and s_prime_indices is not None:
                q_values_s_prime = q_table[s_prime_indices]
                if not np.all(np.isnan(q_values_s_prime)):
                    max_q_next = np.nanmax(q_values_s_prime)
            # Bellman equation or last reward in case of early termination
            td_target = r_learn + self.discount_factor * max_q_next if not effective_done else r_learn
            td_error = td_target - current_q
            self._last_td_errors[gain_name] = float(td_error)
            learning_metrics[f'td_error_{gain_name}'] = float(td_error) # Guardar td_error para logging

            # Cálculo de alpha_n = 1 / (n_visitas + 1)
            visit_count = self.visit_counts[gain_name][s_indices + (action_idx,)]
            #alpha_n = 1.0 / (visit_count + 1)
            # Actualización incremental sin sesgo
            #new_q = current_q + alpha_n * td_error

            new_q = current_q + self._current_learning_rate * td_error
            # Update learn
            if pd.notna(new_q) and np.isfinite(new_q):
                q_table[s_indices + (action_idx,)] = new_q
            self.visit_counts[gain_name][s_indices + (action_idx,)] += 1

        return learning_metrics

    def reset_agent(self):
        # Decaimiento de Epsilon y Learning Rate
        if self.use_epsilon_decay:
            self._current_epsilon = max(self._epsilon_min, self._current_epsilon * self._epsilon_decay)
        if self.use_learning_rate_decay:
            self._current_learning_rate = max(self._learning_rate_min, self._current_learning_rate * self._learning_rate_decay)
        
        self._last_td_errors.clear() # Resetear TD errors

        # Resetear y actualizar parámetros de Early Termination para el nuevo episodio
        if self._early_termination_enabled_flag:
            for gain in self.agent_defining_vars:
                metrics = self.current_episode_improvement_metric_values.get(gain, [])
                avg_metric = np.mean(metrics) if metrics else 0.0
                # Actualizar paciencia si es adaptativa
                if self.et_patience_type == 'adaptive':
                    improvement_change = avg_metric - self.et_last_episode_avg_improvement.get(gain, 0.0)
                    base_patience = self.et_patience_counters.get(gain, self.et_min_patience)
                    updated_patience = base_patience + self.et_patience_adj_rate_eta * np.sign(improvement_change)
                    self.et_patience_counters[gain] = int(np.clip(updated_patience, self.et_min_patience, self.et_max_patience))
                # Resetear contadores y flags para el nuevo episodio
                self.et_no_improvement_counters[gain] = 0
                self.et_penalty_beta_factors[gain] = self.et_initial_penalty_beta.get(gain, 0.9)
                self.et_last_interval_improvement_metric[gain] = -np.inf
                self.current_episode_improvement_metric_values[gain] = []
                self.et_last_episode_avg_improvement[gain] = avg_metric
                self._et_requested_flags[gain] = False
            # Actualizar el snapshot de ET para el logging del inicio del nuevo episodio
            self._update_et_metrics_snapshot_for_reset()
        # logger.debug(f"[PIDQAgent:reset_agent] Epsilon={self._current_epsilon:.4f}, LR={self._current_learning_rate:.4f}. ET params reset.")
    
    # --- Propiedades y Getters de Interfaz (Simplificados) ---
    @property
    def reward_strategy(self) -> RewardStrategy: 
        return self._reward_strategy_instance
    @property
    def epsilon(self) -> float: 
        return self._current_epsilon # Usar _current_epsilon
    @property
    def learning_rate(self) -> float: 
        return self._current_learning_rate # Usar _current_learning_rate
    @property
    def early_termination_enabled(self) -> bool: 
        return self._early_termination_enabled_flag

    def get_agent_defining_vars(self) -> List[str]: 
        return self.agent_defining_vars
    
    def should_episode_terminate_early(self) -> bool:
        if not self._early_termination_enabled_flag: return False
        return any(self._et_requested_flags.values())

    # --- Métodos para Tablas Auxiliares (usados por RewardStrategy) ---

    def get_auxiliary_table_names(self) -> List[str]:
        return list(self.aux_tables.keys())

    def get_auxiliary_table_value(self, table_name: str, gain: str, s_indices: Optional[tuple]) -> Optional[float]:
        if table_name not in self.aux_tables or gain not in self.aux_tables[table_name] or s_indices is None:
            return None
        return float(self.aux_tables[table_name][gain][s_indices])
        
    def update_auxiliary_table_value(self, table_name: str, gain: str, s_indices: tuple, value: float):
        self.aux_tables[table_name][gain][s_indices] = value
        self.aux_visit_counts[f"{table_name}_visits"][gain][s_indices] += 1

    # --- Métodos para Logging/Inspección (llamados por MetricsCollector) ---

    def get_agent_state_for_saving(self) -> Dict[str, Any]:
        # logger.debug("[PIDQAgent:get_agent_state_for_saving] Preparing agent state for saving...")
        agent_data: Dict[str, Any] = {"q_tables": {}, "visit_counts": {}}
        # Añadir contenedores para tablas auxiliares y sus contadores
        for aux_name_save in self.reward_strategy.required_auxiliary_tables:
            agent_data[f"{aux_name}_tables"] = {}
            agent_data[f"{aux_name}_visit_counts"] = {}
        
        for gain_name, q_table in self.q_tables.items():
            state_vars = self.ordered_state_vars_for_q_table[gain_name]
            state_bins_reps = [self._bin_centers.get(var, []) for var in state_vars]
            if any(len(bins) == 0 for bins in state_bins_reps): continue

            multi_index = pd.MultiIndex.from_product(state_bins_reps, names=state_vars)
            num_actions_for_gain = self.q_tables[gain_name].shape[-1]
            
            q_df = pd.DataFrame(q_table.reshape(-1, num_actions_for_gain), index=multi_index, columns=[f"action_{i}" for i in range(num_actions_for_gain)])
            agent_data["q_tables"][gain_name] = q_df.reset_index().to_dict(orient='records')
            
            v_df = pd.DataFrame(self.visit_counts[gain_name].reshape(-1, num_actions_for_gain), index=multi_index, columns=[f"action_{i}" for i in range(num_actions_for_gain)])
            agent_data["visit_counts"][gain_name] = v_df.reset_index().to_dict(orient='records')

            # Tablas auxiliares y sus contadores de visita
            for aux_name in self.get_auxiliary_table_names():
                aux_table = self.aux_tables[aux_name][gain_name]
                aux_df = pd.DataFrame(aux_table.flatten(), index=multi_index, columns=[f"{aux_name}_value"])
                agent_data[f"{aux_name}_tables"][gain_name] = aux_df.reset_index().to_dict(orient='records')
                
                aux_v_table = self.aux_visit_counts[f"{aux_name}_visits"][gain_name]
                aux_v_df = pd.DataFrame(aux_v_table.flatten(), index=multi_index, columns=[f"{aux_name}_visit_count"])
                agent_data[f"{aux_name}_visit_counts"][gain_name] = aux_v_df.reset_index().to_dict(orient='records')
        
        # logger.debug("[PIDQAgent:get_agent_state_for_saving] Agent state structuring complete.")
        return agent_data

    def get_q_values_for_state(self, s_dict: Dict) -> Dict[str, np.ndarray]:
        q_map: Dict[str, np.ndarray] = {}
        for gain in self.agent_defining_vars:
            s_indices = self._get_discrete_state_indices_tuple(s_dict, gain)
            num_a = self.q_tables[gain].shape[-1]
            q_vals = np.full(num_a, np.nan, dtype=np.float32)
            if s_indices:
                q_vals = self.q_tables[gain][s_indices].astype(np.float32)
            q_map[gain] = q_vals
        return q_map

    def get_visit_counts_for_state(self, s_dict: Dict) -> Dict[str, np.ndarray]:
        v_map: Dict[str, np.ndarray] = {}
        for gain in self.agent_defining_vars:
            s_indices = self._get_discrete_state_indices_tuple(s_dict, gain)
            num_a = self.visit_counts[gain].shape[-1]
            v_counts = np.full(num_a, -1, dtype=np.int32)
            if s_indices:
                v_counts = self.visit_counts[gain][s_indices].astype(np.int32)
            v_map[gain] = v_counts
        return v_map

    def get_baseline_value_for_state(self, s_dict: Dict) -> Dict[str, float]:
        b_map: Dict[str, float] = {}
        if 'baseline' not in self.get_auxiliary_table_names():
            return {g: np.nan for g in self.agent_defining_vars}
        for gain in self.agent_defining_vars:
            s_indices = self._get_discrete_state_indices_tuple(s_dict, gain)
            val = self.get_auxiliary_table_value('baseline', gain, s_indices)
            b_map[gain] = val if val is not None else np.nan
        return b_map
    
    def get_last_early_termination_metrics(self) -> Dict: 
        return getattr(self, '_et_metrics_snapshot', {})
    
    def get_params_log(self) -> Dict[str, Any]:
        """Implements the loggable parameters interface for the agent."""
        # This method provides the current values for logging at any point in the episode.
        log_params = {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
        # Agregar td_errors si existen
        log_params.update(self._get_last_td_errors())
        return log_params
    
    def _get_last_td_errors(self) -> Dict[str, float]: 
        return self._last_td_errors.copy()