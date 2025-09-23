import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Callable # Añadir Callable

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

def _safe_agg(series: pd.Series, agg_func: Callable) -> Any:
    """
    Aplica una función de agregación a una Serie de Pandas de forma segura.
    Maneja Series vacías, Series con solo NaN, y posibles errores durante la agregación.

    Args:
        series (pd.Series): La serie de datos a agregar.
        agg_func (Callable): La función de agregación (e.g., np.mean, np.std, np.min, np.max).

    Returns:
        Any: El resultado de la agregación (usualmente float o int), o np.nan si la
             serie está vacía, contiene solo NaNs, o si ocurre un error.
    """
    # Comprobar si la serie es None o está vacía
    if series is None or series.empty:
        # logger.debug(f"Serie vacía o None para agregación {agg_func.__name__}. Devolviendo NaN.")
        return np.nan

    try:
        # Intentar convertir a numérico, coercing errores a NaN
        numeric_series = pd.to_numeric(series, errors='coerce')

        # Eliminar NaN explícitamente antes de agregar
        valid_series = numeric_series.dropna()

        # Comprobar si quedan datos válidos
        if valid_series.empty:
            # logger.debug(f"Serie sin valores numéricos válidos tras dropna para {agg_func.__name__}. Devolviendo NaN.")
            return np.nan

        # Aplicar la función de agregación
        result = agg_func(valid_series)

        # Verificar si el resultado es NaN/infinito (algunas agregaciones pueden dar inf)
        if pd.isna(result) or not np.isfinite(result):
             # logger.debug(f"Resultado de agregación {agg_func.__name__} es NaN/inf. Devolviendo NaN.")
             return np.nan

        # Intentar convertir a tipo Python estándar (float o int) si es posible
        try:
            if isinstance(result, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(result)
            elif isinstance(result, (np.float_, np.float16, np.float32, np.float64)):
                # A veces np.mean puede devolver float64 incluso con ints, así que convertimos
                return float(result)
            else:
                # Si es otro tipo numpy (e.g., bool), devolver como está si es necesario
                # O convertir a int/float si tiene sentido (e.g., bool -> 0/1)
                return result # Devolver tal cual si no es numérico estándar
        except Exception:
             # Fallback si la conversión a tipo Python falla
             return result

    except Exception as e:
        # Loguear el error si la agregación falla inesperadamente
        logger.warning(f"Error aplicando agregación '{agg_func.__name__}' a la serie. Error: {e}", exc_info=True)
        return np.nan


def summarize_episode(episode_data: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Genera un diccionario resumen completo para los datos detallados de un solo episodio.
    Calcula estadísticas agregadas (media, std, min, max) para métricas numéricas
    y extrae valores finales o representativos para otras.

    Args:
        episode_data (Dict[str, List[Any]]): Diccionario donde las claves son nombres de métricas
                                             y los valores son listas de mediciones recolectadas
                                             durante el episodio.

    Returns:
        Dict[str, Any]: Diccionario que contiene estadísticas resumen para el episodio.
                        Los valores numéricos son floats o ints estándar de Python, o NaN.
    """
    if not isinstance(episode_data, dict):
         logger.error("summarize_episode recibió datos que no son un diccionario. Devolviendo resumen vacío.")
         return {'episode': -1, 'error': 'Invalid input data format'}

    summary: Dict[str, Any] = {'episode': episode_data.get('episode', -1)} # Usar .get para seguridad

    # --- Campos de Resumen Directo (usualmente último valor o pre-calculado) ---
    # Obtener último valor de listas, o valor único si no es lista, o default

    def get_last_or_value(key: str, default: Any = np.nan) -> Any:
        value = episode_data.get(key)
        if isinstance(value, list):
            return value[-1] if value else default
        elif value is not None:
            return value # Asumir que es un valor único pre-calculado
        else:
            return default

    summary['termination_reason'] = get_last_or_value('termination_reason', 'unknown')
    summary['episode_time'] = get_last_or_value('time', np.nan) # Último timestamp
    summary['total_reward'] = get_last_or_value('cumulative_reward', np.nan) # Asume existe o es calculado
    summary['final_epsilon'] = get_last_or_value('epsilon', np.nan)
    summary['final_learning_rate'] = get_last_or_value('learning_rate', np.nan)
    summary['episode_duration_s'] = get_last_or_value('episode_duration_s', np.nan)
    summary['total_agent_decisions'] = get_last_or_value('total_agent_decisions', 0) # Puede ser calculado y añadido
    # Final gains
    summary['final_kp'] = get_last_or_value('final_kp', np.nan)
    summary['final_ki'] = get_last_or_value('final_ki', np.nan)
    summary['final_kd'] = get_last_or_value('final_kd', np.nan)
    # Avg stability score (puede ser calculado y añadido como valor único)
    summary['avg_stability_score'] = get_last_or_value('avg_stability_score', np.nan)


    # Calcular Performance (manejar división por cero o NaN)
    total_reward = summary.get('total_reward')
    ep_time = summary.get('episode_time')
    if pd.notna(total_reward) and pd.notna(ep_time) and ep_time > 1e-6: # Evitar división por cero
        summary['performance'] = total_reward / ep_time
    else:
        summary['performance'] = np.nan


    # --- Estadísticas Agregadas (Media, Std, Min, Max) usando _safe_agg ---
    # Lista de métricas para las que calcular agregados estándar
    metrics_to_aggregate = [
        # System State & Control
        'cart_position', 'cart_velocity', 'pendulum_angle', 'pendulum_velocity',
        'force', 'error', 'integral_error', 'derivative_error',
        # Controller & Agent Params (pueden variar en el tiempo si hay adaptación)
        'kp', 'ki', 'kd',
        # Agent Internals (pueden ser listas si se loguean por paso)
        'epsilon', 'learning_rate',
        # Actions (pueden ser NaN entre decisiones)
        'action_kp', 'action_ki', 'action_kd',
        # Reward & Stability per step
        'reward', 'stability_score',
        # Training Internals (Q-values, visits, baselines, TD errors logueados por decisión)
        'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
        'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
        'baseline_value_kp', 'baseline_value_ki', 'baseline_value_kd',
        'td_error_kp', 'td_error_ki', 'td_error_kd',
        'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
        # Timing
        'step_duration_ms', # 'learn_select_duration_ms' (si se loguea)
        # Gain Steps (si se loguean, podrían ser constantes o variables)
        'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd'
    ]

    for metric in metrics_to_aggregate:
        values = episode_data.get(metric) # Obtener la lista de valores (o None)

        if values is not None and isinstance(values, list):
            # Convertir lista a Serie de Pandas para usar _safe_agg
            # Asegurarse de manejar tipos mixtos o errores durante conversión inicial
            try:
                 # Intentar convertir todo a numérico, errores -> NaN
                 s = pd.Series(values)
                 # No necesitamos convertir a float aquí, _safe_agg lo hará internamente
            except Exception as e:
                 logger.warning(f"Error creando Serie Pandas para métrica '{metric}' en episodio {summary['episode']}. Saltando agregación. Error: {e}")
                 # Asignar NaNs si la creación de la serie falla
                 summary[f'{metric}_mean'] = np.nan
                 summary[f'{metric}_std'] = np.nan
                 summary[f'{metric}_min'] = np.nan
                 summary[f'{metric}_max'] = np.nan
                 continue # Saltar a la siguiente métrica

            # Calcular agregados usando el helper seguro
            summary[f'{metric}_mean'] = _safe_agg(s, np.mean)
            summary[f'{metric}_std'] = _safe_agg(s, np.std)
            summary[f'{metric}_min'] = _safe_agg(s, np.min)
            summary[f'{metric}_max'] = _safe_agg(s, np.max)
        else:
            # Si la métrica no existe o no es una lista, asignar NaN a los agregados
            # logger.debug(f"Métrica '{metric}' no encontrada o no es lista para resumen en episodio {summary['episode']}.")
            summary[f'{metric}_mean'] = np.nan
            summary[f'{metric}_std'] = np.nan
            summary[f'{metric}_min'] = np.nan
            summary[f'{metric}_max'] = np.nan

    # --- Log Adaptative Stats (mu, sigma) - ya son valores finales ---
    adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
    for var_name in adaptive_vars:
         mu_key = f'adaptive_mu_{var_name}'
         sigma_key = f'adaptive_sigma_{var_name}'
         # Usar get_last_or_value para obtener el último valor logueado (o NaN)
         summary[mu_key] = get_last_or_value(mu_key, np.nan)
         summary[sigma_key] = get_last_or_value(sigma_key, np.nan)

    # --- Limpieza final: Convertir todos los NaN de numpy a None para JSON si es necesario? ---
    # Por ahora, mantenemos NaN de numpy, ya que NumpyEncoder los manejará.
    # O convertir explícitamente a float estándar de Python o None
    # final_summary = {k: (float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else
    #                      (int(v) if isinstance(v, (np.integer, int)) else
    #                       None if pd.isna(v) else v)) # Convertir NaN a None
    #                 for k, v in summary.items()}
    # return final_summary

    # Devolver el resumen con NaNs de numpy donde corresponda
    return summary

# La función save_summary_table se elimina de aquí (movida a ResultHandler)
# def save_summary_table(summary_list: List[Dict], filename: str): ... (ELIMINADA)