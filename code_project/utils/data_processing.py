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
    # (Código sin cambios)
    if series is None or series.empty: return np.nan
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        valid_series = numeric_series.dropna()
        if valid_series.empty: return np.nan
        result = agg_func(valid_series)
        if pd.isna(result) or not np.isfinite(result): return np.nan
        try: # Intentar convertir a tipos Python estándar
            if isinstance(result, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)): return int(result)
            elif isinstance(result, (np.float_, np.float16, np.float32, np.float64)): return float(result)
            else: return result
        except Exception: return result
    except Exception as e:
        # logger.warning(f"Error aplicando agregación '{agg_func.__name__}': {e}", exc_info=True)
        return np.nan

def get_last_or_value(data_dict: Dict[str, List[Any]], key: str, default: Any = np.nan) -> Any:
    """Helper para obtener el último valor de una lista en el dict, o el valor si no es lista."""
    value = data_dict.get(key)
    if isinstance(value, list):
        # Devolver el último elemento si la lista no está vacía, sino default
        return value[-1] if value else default
    elif value is not None:
        # Si no es lista pero existe, devolverlo (asume valor único pre-calculado)
        return value
    else:
        # Si la clave no existe o es None, devolver default
        return default

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

    summary: Dict[str, Any] = {}
    # Obtener episodio ID del diccionario si está presente (SimulationManager debería añadirlo)
    summary['episode'] = get_last_or_value(episode_data, 'episode', -1)

    # --- Campos de Resumen Directo (usualmente último valor o pre-calculado) ---
    summary['termination_reason'] = get_last_or_value(episode_data, 'termination_reason', 'unknown')
    summary['episode_time'] = get_last_or_value(episode_data, 'time', np.nan) # Último timestamp
    # total_reward y avg_stability_score son calculados por SimManager y añadidos a episode_data
    summary['total_reward'] = get_last_or_value(episode_data, 'total_reward', np.nan)
    summary['avg_stability_score'] = get_last_or_value(episode_data, 'avg_stability_score', np.nan)
    summary['final_epsilon'] = get_last_or_value(episode_data, 'epsilon', np.nan)
    summary['final_learning_rate'] = get_last_or_value(episode_data, 'learning_rate', np.nan)
    summary['episode_duration_s'] = get_last_or_value(episode_data, 'episode_duration_s', np.nan)
    summary['total_agent_decisions'] = get_last_or_value(episode_data, 'total_agent_decisions', 0)
    summary['final_kp'] = get_last_or_value(episode_data, 'final_kp', np.nan)
    summary['final_ki'] = get_last_or_value(episode_data, 'final_ki', np.nan)
    summary['final_kd'] = get_last_or_value(episode_data, 'final_kd', np.nan)

    # Calcular Performance
    total_reward = summary.get('total_reward')
    ep_time = summary.get('episode_time')
    if pd.notna(total_reward) and pd.notna(ep_time) and ep_time > 1e-6:
        summary['performance'] = total_reward / ep_time
    else:
        summary['performance'] = np.nan

    # --- Métricas a EXCLUIR de la agregación estadística (mean, std, min, max) ---
    # Se reportará su valor final en su lugar.
    metrics_to_exclude_from_agg = [
        'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
        'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd',
        # Todos los adaptive_mu y adaptive_sigma
        'adaptive_mu_angle', 'adaptive_sigma_angle',
        'adaptive_mu_angular_velocity', 'adaptive_sigma_angular_velocity',
        'adaptive_mu_cart_position', 'adaptive_sigma_cart_position',
        'adaptive_mu_cart_velocity', 'adaptive_sigma_cart_velocity',
        # Otras métricas no numéricas o que no tiene sentido agregar
        'time', 'id_agent_decision', 'termination_reason', 'episode_duration_s',
        'total_agent_decisions', 'final_kp', 'final_ki', 'final_kd',
        'total_reward', 'avg_stability_score', # Ya son agregados
    ]
    # Añadir métricas que no existen en episode_data a excluidas para evitar errores
    metrics_to_exclude_from_agg.extend([m for m in metrics_to_exclude_from_agg if m not in episode_data])

    # --- Estadísticas Agregadas (Media, Std, Min, Max) usando _safe_agg ---
    # Iterar sobre todas las claves en episode_data
    for metric, values in episode_data.items():
        # Saltar las métricas excluidas y las que no son listas
        if metric in metrics_to_exclude_from_agg or not isinstance(values, list):
            continue

        # Convertir lista a Serie de Pandas
        try:
             s = pd.Series(values)
        except Exception as e:
             logger.warning(f"Error creando Serie Pandas para métrica '{metric}' en episodio {summary['episode']}. Saltando agregación. Error: {e}")
             summary[f'{metric}_mean'] = np.nan; summary[f'{metric}_std'] = np.nan
             summary[f'{metric}_min'] = np.nan; summary[f'{metric}_max'] = np.nan
             continue

        # Calcular agregados usando el helper seguro
        summary[f'{metric}_mean'] = _safe_agg(s, np.mean)
        summary[f'{metric}_std'] = _safe_agg(s, np.std)
        summary[f'{metric}_min'] = _safe_agg(s, np.min)
        summary[f'{metric}_max'] = _safe_agg(s, np.max)


    # --- Añadir Valores Finales de las Métricas Excluidas (si existen) ---
    metrics_final_value = [
        'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
        'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd',
        'adaptive_mu_angle', 'adaptive_sigma_angle',
        'adaptive_mu_angular_velocity', 'adaptive_sigma_angular_velocity',
        'adaptive_mu_cart_position', 'adaptive_sigma_cart_position',
        'adaptive_mu_cart_velocity', 'adaptive_sigma_cart_velocity',
        # Incluir también q_value_max aquí si se prefiere el valor final en lugar de agregados
        # 'q_value_max_kp', 'q_value_max_ki', 'q_value_max_kd',
        # Incluir TD errors finales?
        # 'td_error_kp', 'td_error_ki', 'td_error_kd',
    ]
    for metric in metrics_final_value:
         # Usar el helper para obtener el último valor si es lista, o el valor si no lo es
         summary[metric] = get_last_or_value(episode_data, metric, np.nan)


    # Devolver el resumen con NaNs de numpy donde corresponda
    return summary