from collections import defaultdict
from interfaces.metrics_collector import MetricsCollector # Importar interfaz
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

# Evitar importación circular para type hinting del agente
if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent # Usar interfaz

# 2.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class ExtendedMetricsCollector(MetricsCollector): # Implementar Interfaz
    """
    Implementación concreta de MetricsCollector.
    Recolecta métricas en un defaultdict. Incluye métodos helper para loguear
    datos específicos del agente (Q-values, visitas, etc.) usando la interfaz RLAgent.
    """
    _instance_count = 0 # Contador para IDs de instancia (debug)

    def __init__(self):
        """Inicializa el colector de métricas."""
        ExtendedMetricsCollector._instance_count += 1
        # Usar defaultdict(list) para simplificar log
        self.metrics: Dict[str, List[Any]] = defaultdict(list)
        self.episode_id: int = -1 # ID del episodio actual
        self._instance_id = ExtendedMetricsCollector._instance_count
        # logger.debug(f"ExtendedMetricsCollector instance {self._instance_id} created.")

    def log(self, metric_name: str, metric_value: Any):
        """Registra un valor único para una métrica, convirtiendo None/inf a NaN."""
        value_to_log: Any
        # 2.2: Convertir None, inf, -inf a np.nan para consistencia numérica
        if metric_value is None:
            value_to_log = np.nan
        elif isinstance(metric_value, (float, int)) and not np.isfinite(metric_value):
            value_to_log = np.nan
        else:
            value_to_log = metric_value

        #logger.debug(f"Log Metric: {metric_name} = {value_to_log}") # Log muy verboso

        # 2.3: Simplificar try-except, defaultdict maneja claves nuevas
        try:
            self.metrics[metric_name].append(value_to_log)
        except MemoryError: # Capturar error específico si la lista crece demasiado
             logger.error(f"MemoryError al añadir métrica '{metric_name}'. Demasiados datos?", exc_info=True)
             # Podría implementarse lógica para truncar o parar aquí
             raise # Relanzar MemoryError
        except Exception as e:
             # Loguear otros errores inesperados, pero intentar continuar
             logger.error(f"(Inst {self._instance_id}) Error inesperado añadiendo métrica '{metric_name}': {e}", exc_info=True)

    def get_metrics(self) -> Dict[str, List[Any]]:
        """Devuelve una copia de las métricas recolectadas como dict estándar."""
        # 2.4: Devolver copia explícita para evitar modificaciones externas
        return dict(self.metrics)

    def reset(self, episode_id: int):
        """Limpia métricas y establece nuevo ID de episodio."""
        # logger.debug(f"(Inst {self._instance_id}) Resetting metrics for episode {episode_id}")
        self.metrics.clear()
        self.episode_id = episode_id
        # Añadir episode_id al dict al resetear? No, SimulationManager lo añade al final.

    # --- Métodos específicos para loguear datos del agente/entrenamiento ---
    # Estos son métodos de conveniencia, no parte de la interfaz MetricsCollector.
    # Usan la interfaz RLAgent para obtener los datos.

    def log_q_values(self, agent: 'RLAgent', agent_state_dict: Dict):
        """Registra los Q-values máximos para el estado actual."""
        # 2.5: Usar interfaz RLAgent.get_q_values_for_state
        try:
            q_values_per_gain = agent.get_q_values_for_state(agent_state_dict)
            for gain, q_vals_array in q_values_per_gain.items():
                # nanmax devuelve NaN si todo el array es NaN
                max_q = np.nanmax(q_vals_array) if isinstance(q_vals_array, np.ndarray) else np.nan
                self.log(f'q_value_max_{gain}', max_q)
        except AttributeError:
             logger.warning(f"Agente {type(agent).__name__} no implementa get_q_values_for_state. No se loguean Q-values.")
             # Loguear NaN por consistencia si el método no existe
             for gain in ['kp', 'ki', 'kd']: self.log(f'q_value_max_{gain}', np.nan)
        except Exception as e:
            logger.warning(f"Error logueando Q-values: {e}", exc_info=True)
            for gain in ['kp', 'ki', 'kd']: self.log(f'q_value_max_{gain}', np.nan)


    def log_q_visit_counts(self, agent: 'RLAgent', agent_state_dict: Dict):
        """Registra la suma de cuentas de visita N(s,a) para el estado actual."""
        # 2.6: Usar interfaz RLAgent.get_visit_counts_for_state
        try:
            visit_counts_per_gain = agent.get_visit_counts_for_state(agent_state_dict)
            for gain, visits_array in visit_counts_per_gain.items():
                total_visits = np.nan # Default a NaN
                if isinstance(visits_array, np.ndarray) and visits_array.size > 0:
                    # Sumar solo visitas válidas (>= 0), nansum maneja NaN si los hubiera
                    valid_visits = visits_array[visits_array >= 0]
                    total_visits = np.nansum(valid_visits)
                self.log(f'q_visit_count_state_{gain}', total_visits)
        except AttributeError:
            logger.warning(f"Agente {type(agent).__name__} no implementa get_visit_counts_for_state. No se loguean Visit Counts.")
            for gain in ['kp', 'ki', 'kd']: self.log(f'q_visit_count_state_{gain}', np.nan)
        except Exception as e:
            logger.warning(f"Error logueando Visit Counts: {e}", exc_info=True)
            for gain in ['kp', 'ki', 'kd']: self.log(f'q_visit_count_state_{gain}', np.nan)

    def log_baselines(self, agent: 'RLAgent', agent_state_dict: Dict):
        """Registra el valor del baseline B(s) para el estado actual."""
        # 2.7: Usar interfaz RLAgent.get_baseline_value_for_state
        try:
            baselines_per_gain = agent.get_baseline_value_for_state(agent_state_dict)
            for gain, baseline_value in baselines_per_gain.items():
                # El método del agente ya debería devolver float o NaN
                self.log(f'baseline_value_{gain}', baseline_value)
        except AttributeError:
            logger.warning(f"Agente {type(agent).__name__} no implementa get_baseline_value_for_state. No se loguean Baselines.")
            for gain in ['kp', 'ki', 'kd']: self.log(f'baseline_value_{gain}', np.nan)
        except Exception as e:
            logger.warning(f"Error logueando Baselines: {e}", exc_info=True)
            for gain in ['kp', 'ki', 'kd']: self.log(f'baseline_value_{gain}', np.nan)

    def log_virtual_rewards(self, virtual_rewards_dict: Optional[Dict[str, float]]):
        """Registra las recompensas virtuales/diferenciales (Echo Baseline)."""
        gains_to_log = ['kp', 'ki', 'kd']
        if virtual_rewards_dict is not None and isinstance(virtual_rewards_dict, dict):
            for gain in gains_to_log:
                value = virtual_rewards_dict.get(gain, np.nan) # Default a NaN si falta la clave
                self.log(f'virtual_reward_{gain}', value) # log maneja NaN
        else: # Log NaN si el dict es None o no es dict
            for gain in gains_to_log: self.log(f'virtual_reward_{gain}', np.nan)

    def log_td_errors(self, td_errors_dict: Dict[str, float]):
        """Registra los TD errors del último paso de learn."""
        # 2.8: Usar interfaz RLAgent.get_last_td_errors
        gains_to_log = ['kp', 'ki', 'kd']
        if isinstance(td_errors_dict, dict):
            for gain in gains_to_log:
                # El método del agente ya devuelve float o NaN
                td_error = td_errors_dict.get(gain, np.nan)
                self.log(f'td_error_{gain}', td_error)
        else: # Log NaN si no es dict
            logger.warning(f"td_errors_dict no es un diccionario ({type(td_errors_dict)}). Logueando NaNs.")
            for gain in gains_to_log: self.log(f'td_error_{gain}', np.nan)

    def log_adaptive_stats(self, stats_dict: Dict[str, Dict[str, float]]):
        """Registra las estadísticas adaptativas (mu, sigma) del stability calculator."""
        # 2.9: Loguear stats si el dict es válido
        adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
        if not isinstance(stats_dict, dict):
            # Loguear NaN si no hay stats (calculador no adaptativo o error)
             for var_name in adaptive_vars:
                 self.log(f'adaptive_mu_{var_name}', np.nan)
                 self.log(f'adaptive_sigma_{var_name}', np.nan)
             return

        for var_name in adaptive_vars:
            mu, sigma = np.nan, np.nan
            var_stats = stats_dict.get(var_name) # Obtener sub-dict
            if isinstance(var_stats, dict):
                mu = var_stats.get('mu', np.nan) # Default a NaN si falta 'mu'
                sigma = var_stats.get('sigma', np.nan) # Default a NaN si falta 'sigma'
            self.log(f'adaptive_mu_{var_name}', mu) # log maneja NaN
            self.log(f'adaptive_sigma_{var_name}', sigma) # log maneja NaN