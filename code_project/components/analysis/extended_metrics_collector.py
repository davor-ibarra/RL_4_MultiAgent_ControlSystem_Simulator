from collections import defaultdict
from interfaces.metrics_collector import MetricsCollector # Importar interfaz
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

# Evitar importación circular para type hinting del agente
if TYPE_CHECKING:
    from components.agents.pid_qlearning_agent import PIDQLearningAgent
    # from interfaces.rl_agent import RLAgent # Alternativa genérica

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class ExtendedMetricsCollector(MetricsCollector):
    """
    Implementación concreta de MetricsCollector.
    Recolecta métricas detalladas durante la simulación, almacenándolas en listas
    dentro de un diccionario. Incluye seguimiento de episodio y métodos helper
    para registrar datos específicos del agente y del entrenamiento.
    """
    def __init__(self):
        """Inicializa el colector de métricas."""
        self.metrics: Dict[str, List[Any]] = defaultdict(list)
        self.episode_id: int = -1 # ID del episodio actual
        # logger.debug("ExtendedMetricsCollector instance created.") # Log de bajo nivel

    def log(self, metric_name: str, metric_value: Any):
        """
        Registra un valor único para una métrica específica en el episodio actual.
        Convierte None a np.nan para consistencia numérica.

        Args:
            metric_name (str): Nombre de la métrica (e.g., 'pendulum_angle').
            metric_value (Any): Valor de la métrica.
        """
        # Convertir None a np.nan para facilitar análisis numérico posterior
        # También convertir inf/-inf a NaN? Podría ser útil.
        value_to_log: Any
        if metric_value is None:
            value_to_log = np.nan
        elif isinstance(metric_value, (float, int)) and not np.isfinite(metric_value):
            # logger.debug(f"Convirtiendo valor infinito/NaN ({metric_value}) a np.nan para métrica '{metric_name}'.")
            value_to_log = np.nan # Convertir inf/-inf a NaN
        else:
            value_to_log = metric_value

        try:
            self.metrics[metric_name].append(value_to_log)
        except Exception as e:
             # Error muy improbable, pero por seguridad
             logger.error(f"Error inesperado añadiendo métrica '{metric_name}' al diccionario: {e}", exc_info=True)

    def get_metrics(self) -> Dict[str, List[Any]]:
        """
        Devuelve una copia de todas las métricas recolectadas para el episodio actual,
        incluyendo el ID del episodio.

        Returns:
            Dict[str, List[Any]]: Diccionario de métricas.
        """
        # Crear una copia para evitar modificación externa
        current_metrics = dict(self.metrics)
        # Asegurar que el ID del episodio esté presente (como lista de un elemento para consistencia?)
        # No, mejor como valor único. summarize_episode lo manejará.
        current_metrics['episode'] = self.episode_id
        return current_metrics

    def reset(self, episode_id: int):
        """
        Limpia todas las métricas acumuladas y establece el ID del nuevo episodio.

        Args:
            episode_id (int): ID del episodio que va a comenzar.
        """
        # logger.debug(f"Reseteando métricas para nuevo episodio: {episode_id}")
        self.metrics.clear()
        self.episode_id = episode_id

    # --- Métodos específicos para loguear datos del agente/entrenamiento ---
    # Estos métodos serán llamados por SimulationManager en los puntos adecuados

    def log_q_values(self, agent: 'PIDQLearningAgent', agent_state_dict: Dict):
        """Registra los Q-values máximos para el estado actual para cada ganancia."""
        # logger.debug(f"Logging Q-values for state: {agent_state_dict.keys()}")
        try:
            # Obtener todos los Q-values para el estado actual
            q_values_per_gain = agent.get_q_values_for_state(agent_state_dict)
            for gain, q_vals_array in q_values_per_gain.items():
                max_q = np.nan # Default a NaN
                if isinstance(q_vals_array, np.ndarray) and q_vals_array.size > 0:
                    # Usar nanmax para ignorar NaNs si los hubiera
                    try:
                         max_q = np.nanmax(q_vals_array)
                         # Si todos son NaN, nanmax devuelve NaN (en versiones recientes de numpy)
                         # Si nanmax da error o devuelve -inf, mantener NaN
                         if not np.isfinite(max_q): max_q = np.nan
                    except ValueError: # Si el array está vacío o contiene solo NaN (versiones antiguas)
                         max_q = np.nan
                # Loguear el Q-value máximo (o NaN si no se pudo calcular)
                self.log(f'q_value_max_{gain}', max_q)
        except AttributeError:
             logger.warning(f"Agente {type(agent).__name__} no tiene método 'get_q_values_for_state'. No se loguean Q-values.")
             for gain in ['kp', 'ki', 'kd']: self.log(f'q_value_max_{gain}', np.nan)
        except Exception as e:
            logger.warning(f"Error inesperado logueando Q-values: {e}", exc_info=True)
            # Asegurar que se loguea NaN para todas las ganancias en caso de error
            for gain in ['kp', 'ki', 'kd']: self.log(f'q_value_max_{gain}', np.nan)


    def log_q_visit_counts(self, agent: 'PIDQLearningAgent', agent_state_dict: Dict):
        """Registra la suma de cuentas de visita para el estado actual para cada ganancia."""
        # logger.debug(f"Logging Visit Counts for state: {agent_state_dict.keys()}")
        try:
            visit_counts_per_gain = agent.get_visit_counts_for_state(agent_state_dict)
            for gain, visits_array in visit_counts_per_gain.items():
                total_visits = np.nan # Default a NaN
                if isinstance(visits_array, np.ndarray) and visits_array.size > 0:
                     # Sumar solo cuentas válidas (>= 0)
                     valid_visits = visits_array[visits_array >= 0]
                     if valid_visits.size > 0:
                          total_visits = np.sum(valid_visits)
                     else:
                          total_visits = 0 # Si no hay visitas válidas, es 0
                # Loguear suma de visitas (o NaN si hubo error)
                self.log(f'q_visit_count_state_{gain}', total_visits)
        except AttributeError:
             logger.warning(f"Agente {type(agent).__name__} no tiene método 'get_visit_counts_for_state'. No se loguean cuentas de visita.")
             for gain in ['kp', 'ki', 'kd']: self.log(f'q_visit_count_state_{gain}', np.nan)
        except Exception as e:
            logger.warning(f"Error inesperado logueando cuentas de visita: {e}", exc_info=True)
            for gain in ['kp', 'ki', 'kd']: self.log(f'q_visit_count_state_{gain}', np.nan)


    def log_baselines(self, agent: 'PIDQLearningAgent', agent_state_dict: Dict):
        """Registra el valor del baseline B(s) para el estado actual para cada ganancia."""
        # logger.debug(f"Logging Baselines for state: {agent_state_dict.keys()}")
        try:
            baselines_per_gain = agent.get_baseline_value_for_state(agent_state_dict)
            for gain, baseline_value in baselines_per_gain.items():
                # El valor ya debería ser float o NaN
                self.log(f'baseline_value_{gain}', baseline_value)
        except AttributeError:
             logger.warning(f"Agente {type(agent).__name__} no tiene método 'get_baseline_value_for_state'. No se loguean baselines.")
             for gain in ['kp', 'ki', 'kd']: self.log(f'baseline_value_{gain}', np.nan)
        except Exception as e:
            logger.warning(f"Error inesperado logueando baselines: {e}", exc_info=True)
            for gain in ['kp', 'ki', 'kd']: self.log(f'baseline_value_{gain}', np.nan)


    def log_virtual_rewards(self, virtual_rewards_dict: Optional[Dict[str, float]]):
        """Registra las recompensas diferenciales/virtuales calculadas (usado por Echo Baseline)."""
        # logger.debug(f"Logging Virtual Rewards: {virtual_rewards_dict}")
        if virtual_rewards_dict is not None:
            for gain, value in virtual_rewards_dict.items():
                # Asegurar que logueamos para las 3 ganancias, con NaN si falta
                if gain in ['kp', 'ki', 'kd']:
                     self.log(f'virtual_reward_{gain}', value if pd.notna(value) else np.nan)
        # Asegurar que existen las claves con NaN si el dict es None o no contiene todas las ganancias
        for gain in ['kp', 'ki', 'kd']:
             if virtual_rewards_dict is None or gain not in virtual_rewards_dict:
                  self.log(f'virtual_reward_{gain}', np.nan)


    def log_td_errors(self, td_errors_dict: Dict[str, float]):
        """Registra los TD errors calculados en el último paso de aprendizaje."""
        # logger.debug(f"Logging TD Errors: {td_errors_dict}")
        for gain in ['kp', 'ki', 'kd']:
            # Usar .get para obtener el valor o NaN si no está en el dict
            td_error = td_errors_dict.get(gain, np.nan)
            self.log(f'td_error_{gain}', td_error if pd.notna(td_error) else np.nan)


    def log_adaptive_stats(self, stats_dict: Dict[str, Dict[str, float]]):
        """Registra las estadísticas adaptativas (mu, sigma) actuales del stability calculator."""
        # logger.debug(f"Logging Adaptive Stats: {stats_dict.keys()}")
        if not isinstance(stats_dict, dict): return # Salir si no es diccionario

        adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
        for var_name in adaptive_vars:
            mu = np.nan
            sigma = np.nan
            if var_name in stats_dict:
                var_stats = stats_dict[var_name]
                if isinstance(var_stats, dict):
                    mu = var_stats.get('mu', np.nan)
                    sigma = var_stats.get('sigma', np.nan)
            # Loguear mu y sigma (serán NaN si no existen)
            self.log(f'adaptive_mu_{var_name}', mu if pd.notna(mu) else np.nan)
            self.log(f'adaptive_sigma_{var_name}', sigma if pd.notna(sigma) else np.nan)