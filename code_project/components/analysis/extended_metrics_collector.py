from collections import defaultdict
from interfaces.metrics_collector import MetricsCollector # Importar interfaz
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

# Evitar importación circular para type hinting del agente
if TYPE_CHECKING:
    # from components.agents.pid_qlearning_agent import PIDQLearningAgent # O usar interfaz
    from interfaces.rl_agent import RLAgent

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class ExtendedMetricsCollector(MetricsCollector): # Implementar Interfaz
    """
    Implementación concreta de MetricsCollector.
    Recolecta métricas detalladas durante la simulación. Incluye métodos helper
    para registrar datos específicos del agente y del entrenamiento.
    """
    _instance_count = 0 # Contador para IDs de instancia (debug)

    def __init__(self):
        """Inicializa el colector de métricas."""
        ExtendedMetricsCollector._instance_count += 1
        # Usar defaultdict para simplificar log
        self.metrics: Dict[str, List[Any]] = defaultdict(list)
        self.episode_id: int = -1 # ID del episodio actual
        self._instance_id = ExtendedMetricsCollector._instance_count
        # logger.debug(f"ExtendedMetricsCollector instance {self._instance_id} created.")

    def log(self, metric_name: str, metric_value: Any):
        """Registra un valor único para una métrica."""
        # ... (lógica mantenida, convierte None/inf a NaN) ...
        value_to_log: Any
        if metric_value is None: 
            value_to_log = np.nan
        elif isinstance(metric_value, (float, int)) and not np.isfinite(metric_value): 
            value_to_log = np.nan
        else: 
            value_to_log = metric_value
            logger.debug(f"ExtendedMetricsCollector -> log -> {metric_name} = {value_to_log}") #Usar solo con unos pocos episodios

        try:
            self.metrics[metric_name].append(value_to_log)
        except Exception as e:
             logger.error(f"(Inst {self._instance_id}) Error inesperado añadiendo métrica '{metric_name}': {e}", exc_info=True)


    def get_metrics(self) -> Dict[str, List[Any]]:
        """Devuelve las métricas recolectadas para el episodio actual."""
        # Devuelve una copia del defaultdict como dict estándar
        current_metrics = dict(self.metrics)
        # El ID del episodio se añade en SimulationManager antes de llamar a summarize
        # No añadirlo aquí para mantener el colector enfocado en las listas de métricas
        return current_metrics


    def reset(self, episode_id: int):
        """Limpia métricas y establece nuevo ID de episodio."""
        # logger.debug(f"(Inst {self._instance_id}) Resetting metrics for episode {episode_id}")
        self.metrics.clear()
        self.episode_id = episode_id


    # --- Métodos específicos para loguear datos del agente/entrenamiento ---
    # (Lógica mantenida como estaba, usando interfaz RLAgent para type hints)

    def log_q_values(self, agent: 'RLAgent', agent_state_dict: Dict):
        """Registra los Q-values máximos para el estado actual."""
        # ... (código sin cambios) ...
        try:
            if not hasattr(agent, 'get_q_values_for_state'): raise AttributeError("No get_q_values_for_state")
            q_values_per_gain = agent.get_q_values_for_state(agent_state_dict)
            for gain, q_vals_array in q_values_per_gain.items():
                max_q = np.nan
                if isinstance(q_vals_array, np.ndarray) and q_vals_array.size > 0:
                    try: 
                        max_q = np.nanmax(q_vals_array)
                    except ValueError: 
                        max_q = np.nan # Handle all-NaN case if nanmax raises error
                    if not np.isfinite(max_q): max_q = np.nan
                self.log(f'q_value_max_{gain}', max_q)
        except AttributeError as e: 
            logger.warning(f"(Inst {self._instance_id}) Agente {type(agent).__name__} sin método '{e}'. No log Q-values."); # Log NaN for consistency
        except Exception as e: 
            logger.warning(f"(Inst {self._instance_id}) Error log Q-values: {e}", exc_info=True); # Log NaN


    def log_q_visit_counts(self, agent: 'RLAgent', agent_state_dict: Dict):
        """Registra la suma de cuentas de visita N(s,a) para el estado actual."""
        # ... (código sin cambios) ...
        try:
            if not hasattr(agent, 'get_visit_counts_for_state'): raise AttributeError("No get_visit_counts_for_state")
            visit_counts_per_gain = agent.get_visit_counts_for_state(agent_state_dict)
            for gain, visits_array in visit_counts_per_gain.items():
                total_visits = np.nan
                if isinstance(visits_array, np.ndarray) and visits_array.size > 0:
                     valid_visits = visits_array[visits_array >= 0]
                     total_visits = np.sum(valid_visits) if valid_visits.size > 0 else 0
                self.log(f'q_visit_count_state_{gain}', total_visits)
        except AttributeError as e: logger.warning(f"(Inst {self._instance_id}) Agente {type(agent).__name__} sin método '{e}'. No log Visit Counts."); # Log NaN
        except Exception as e: logger.warning(f"(Inst {self._instance_id}) Error log Visit Counts: {e}", exc_info=True); # Log NaN


    def log_baselines(self, agent: 'RLAgent', agent_state_dict: Dict):
        """Registra el valor del baseline B(s) para el estado actual."""
        # ... (código sin cambios) ...
        try:
            if not hasattr(agent, 'get_baseline_value_for_state'): raise AttributeError("No get_baseline_value_for_state")
            baselines_per_gain = agent.get_baseline_value_for_state(agent_state_dict)
            for gain, baseline_value in baselines_per_gain.items():
                self.log(f'baseline_value_{gain}', baseline_value if pd.notna(baseline_value) else np.nan)
        except AttributeError as e: logger.warning(f"(Inst {self._instance_id}) Agente {type(agent).__name__} sin método '{e}'. No log Baselines."); # Log NaN
        except Exception as e: logger.warning(f"(Inst {self._instance_id}) Error log Baselines: {e}", exc_info=True); # Log NaN


    def log_virtual_rewards(self, virtual_rewards_dict: Optional[Dict[str, float]]):
        """Registra las recompensas virtuales/diferenciales (Echo Baseline)."""
        # ... (código sin cambios) ...
        gains_to_log = ['kp', 'ki', 'kd']
        if virtual_rewards_dict is not None and isinstance(virtual_rewards_dict, dict):
            for gain in gains_to_log:
                value = virtual_rewards_dict.get(gain, np.nan)
                self.log(f'virtual_reward_{gain}', value if pd.notna(value) else np.nan)
        else: # Log NaN si el dict es None
            for gain in gains_to_log: self.log(f'virtual_reward_{gain}', np.nan)


    def log_td_errors(self, td_errors_dict: Dict[str, float]):
        """Registra los TD errors del último paso de learn."""
        # ... (código sin cambios) ...
        gains_to_log = ['kp', 'ki', 'kd']
        if isinstance(td_errors_dict, dict):
             for gain in gains_to_log:
                  td_error = td_errors_dict.get(gain, np.nan)
                  self.log(f'td_error_{gain}', td_error if pd.notna(td_error) else np.nan)
        else: # Log NaN si no es dict
             for gain in gains_to_log: self.log(f'td_error_{gain}', np.nan)


    def log_adaptive_stats(self, stats_dict: Dict[str, Dict[str, float]]):
        """Registra las estadísticas adaptativas (mu, sigma) del stability calculator."""
        # ... (código sin cambios) ...
        adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
        if not isinstance(stats_dict, dict): return
        for var_name in adaptive_vars:
            mu, sigma = np.nan, np.nan
            if var_name in stats_dict:
                var_stats = stats_dict[var_name]
                if isinstance(var_stats, dict):
                    mu = var_stats.get('mu', np.nan)
                    sigma = var_stats.get('sigma', np.nan)
            self.log(f'adaptive_mu_{var_name}', mu if pd.notna(mu) else np.nan)
            self.log(f'adaptive_sigma_{var_name}', sigma if pd.notna(sigma) else np.nan)