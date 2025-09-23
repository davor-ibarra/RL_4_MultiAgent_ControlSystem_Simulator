# components/analysis/extended_metrics_collector.py
from collections import defaultdict
from interfaces.metrics_collector import MetricsCollector
import numpy as np
import pandas as pd # Para pd.notna/isna
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent # Para type hints en helpers

logger_emc_mod = logging.getLogger(__name__) # Renombrado

class ExtendedMetricsCollector(MetricsCollector):
    _instance_id_counter = 0 # Para diferenciar instancias si se crean múltiples (ej. por tests)

    def __init__(self, allowed_metrics_to_log: List[str]): # Recibe la lista de métricas permitidas
        ExtendedMetricsCollector._instance_id_counter += 1
        self._instance_id = ExtendedMetricsCollector._instance_id_counter
        
        self.collected_data: Dict[str, List[Any]] = defaultdict(list) # Renombrado
        self.current_episode_num: int = -1 # Renombrado

        # Almacenar como un conjunto para búsqueda eficiente.
        # Si allowed_metrics_to_log está vacía, significa que json_history está deshabilitado
        # o no se especificaron métricas, por lo que no se logueará nada para JSON.
        self._allowed_metrics_filter_set: Set[str] = set(allowed_metrics_to_log)
        
        # if not self._allowed_metrics_filter_set:
            # logger_emc_mod.debug(f"[EMC Inst:{self._instance_id}] Initialized with empty allowed_metrics. No detailed history will be stored by this instance.")
        # else:
            # logger_emc_mod.debug(f"[EMC Inst:{self._instance_id}] Initialized. Allowed metrics for JSON: {len(self._allowed_metrics_filter_set)}")

    def log(self, metric_key: str, metric_val: Any): # Renombrados
        # Filtrar en origen: solo almacenar si la métrica está en el conjunto permitido
        if not self._allowed_metrics_filter_set or metric_key not in self._allowed_metrics_filter_set:
            # No loguear nada aquí para evitar spam si muchas métricas se ignoran.
            # El debug en __init__ es suficiente para saber si está activo.
            return

        value_to_store: Any
        if metric_val is None:
            value_to_store = np.nan # Usar NaN de NumPy para consistencia
        elif isinstance(metric_val, (float, int)) and not np.isfinite(metric_val):
            value_to_store = np.nan # Convertir inf/-inf a NaN
        else: # Aceptar strings, bools, y números finitos tal cual
            value_to_store = metric_val
        
        self.collected_data[metric_key].append(value_to_store)

    def get_metrics(self) -> Dict[str, List[Any]]:
        return dict(self.collected_data) # Devolver una copia para evitar modificación externa

    def reset(self, episode_identifier_num: int): # Renombrado
        # logger_emc_mod.debug(f"[EMC Inst:{self._instance_id}] Resetting metrics for episode {episode_identifier_num}")
        self.collected_data.clear()
        self.current_episode_num = episode_identifier_num

    # --- Métodos Helper para loguear datos específicos (opcionales pero convenientes) ---
    # Estos métodos ya no necesitan chequear 'json_history_enabled', 
    # ya que self.log() lo hace internamente basado en _allowed_metrics_filter_set.

    def log_q_values(self, agent_q_log: 'RLAgent', agent_s_dict_q_log: Dict[str, Any]):
        q_values_map = agent_q_log.get_q_values_for_state(agent_s_dict_q_log)
        for gain_key_q, q_vals_arr in q_values_map.items():
            # q_vals_arr puede ser un array de NaNs si el estado no tiene Q-values
            max_q = np.nanmax(q_vals_arr) if isinstance(q_vals_arr, np.ndarray) and q_vals_arr.size > 0 and not np.all(np.isnan(q_vals_arr)) else np.nan
            self.log(f'q_value_max_{gain_key_q}', max_q)

    def log_q_visit_counts(self, agent_v_log: 'RLAgent', agent_s_dict_v_log: Dict[str, Any]):
        visit_counts_map = agent_v_log.get_visit_counts_for_state(agent_s_dict_v_log)
        for gain_key_v, visits_arr in visit_counts_map.items():
            # visits_arr puede tener -1s. Sumar solo los >= 0.
            total_visits = np.nansum(visits_arr[visits_arr >= 0]) if isinstance(visits_arr, np.ndarray) and visits_arr.size > 0 else 0
            self.log(f'q_visit_count_state_{gain_key_v}', int(total_visits))

    def log_baselines(self, agent_b_log: 'RLAgent', agent_s_dict_b_log: Dict[str, Any]):
        # Solo loguear si la estrategia del agente realmente usa/requiere baselines.
        # Esto se puede inferir si 'baseline' está en las tablas auxiliares requeridas.
        if 'baseline' in agent_b_log.reward_strategy.required_auxiliary_tables:
            baselines_map = agent_b_log.get_baseline_value_for_state(agent_s_dict_b_log)
            for gain_key_b, baseline_val_data in baselines_map.items():
                self.log(f'baseline_value_{gain_key_b}', baseline_val_data) # baseline_val_data puede ser NaN

    def log_virtual_rewards(self, virtual_rewards_diff_map: Optional[Dict[str, float]]):
        # Usado por EchoBaseline. virtual_rewards_diff_map contiene R_diff.
        gain_keys_vr = ['kp', 'ki', 'kd'] # Asumimos estas ganancias
        if isinstance(virtual_rewards_diff_map, dict):
            for gk_vr in gain_keys_vr:
                self.log(f'virtual_reward_{gk_vr}', virtual_rewards_diff_map.get(gk_vr, np.nan))
        else: # Si no es dict (ej. None), loguear NaNs
            for gk_vr_none in gain_keys_vr: self.log(f'virtual_reward_{gk_vr_none}', np.nan)

    def log_td_errors(self, td_errors_map_log: Dict[str, float]):
        # td_errors_map_log viene de agent.get_last_td_errors()
        gain_keys_td = ['kp', 'ki', 'kd']
        if isinstance(td_errors_map_log, dict):
            for gk_td in gain_keys_td:
                self.log(f'td_error_{gk_td}', td_errors_map_log.get(gk_td, np.nan))
        # else: logger_emc_mod.warning(f"td_errors_map_log not a valid dict. Logging NaNs for TD errors.")

    def log_adaptive_stats(self, adaptive_stats_dict_log: Dict[str, Dict[str, float]]):
        # adaptive_stats_dict_log es como {'angle': {'mu': x, 'sigma': y}, ...}
        features_for_adapt_log = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
        if not isinstance(adaptive_stats_dict_log, dict):
            # Loguear NaNs si el dict es inválido
            for feat_name_nan in features_for_adapt_log:
                self.log(f'adaptive_mu_{feat_name_nan}', np.nan); self.log(f'adaptive_sigma_{feat_name_nan}', np.nan)
            return

        for feat_name_log in features_for_adapt_log:
            stats_for_feat = adaptive_stats_dict_log.get(feat_name_log, {}) # Default a dict vacío
            self.log(f'adaptive_mu_{feat_name_log}', stats_for_feat.get('mu', np.nan))
            self.log(f'adaptive_sigma_{feat_name_log}', stats_for_feat.get('sigma', np.nan))
    
    def log_early_termination_metrics(self, agent_et_log: 'RLAgent'):
        if not agent_et_log.early_termination_enabled: return # No loguear si ET está desactivado en el agente

        agent_gains_et = agent_et_log.get_agent_defining_vars()
        et_metrics_snapshot = agent_et_log.get_last_early_termination_metrics()
        
        for gain_et_log in agent_gains_et:
            metrics_for_gain = et_metrics_snapshot.get(gain_et_log, {}) # Default a dict vacío
            self.log(f'patience_M_{gain_et_log}', metrics_for_gain.get('patience_M', np.nan))
            self.log(f'no_improvement_counter_c_hat_{gain_et_log}', metrics_for_gain.get('c_hat', np.nan))
            self.log(f'penalty_beta_{gain_et_log}', metrics_for_gain.get('beta', np.nan))
            self.log(f'improvement_metric_value_{gain_et_log}', metrics_for_gain.get('current_metric_value', np.nan))
            self.log(f'last_improvement_metric_value_{gain_et_log}', metrics_for_gain.get('last_metric_value', np.nan))
            self.log(f'requested_early_termination_{gain_et_log}', bool(metrics_for_gain.get('requested_et_flag', False)))