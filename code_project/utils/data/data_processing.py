# utils/data_processing.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Callable, Optional # Añadir Optional y Callable

# 2.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

def _safe_agg(series: Optional[pd.Series], agg_func: Callable) -> Any:
    """
    Aplica una función de agregación a una Serie de Pandas de forma segura.
    Maneja Series None, vacías, con solo NaN, y errores durante la agregación.

    Args:
        series (Optional[pd.Series]): La serie de datos a agregar.
        agg_func (Callable): La función de agregación (e.g., np.mean, np.std, np.min, np.max).

    Returns:
        Any: El resultado de la agregación (usualmente float o int), o np.nan si la
             serie es inválida o la agregación falla.
    """
    # 2.2: Manejar None input
    if series is None or series.empty:
        return np.nan
    try:
        # Intentar convertir a numérico, coercing errores a NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Eliminar NaN y Infinitos antes de agregar
        valid_series = numeric_series.dropna()[np.isfinite(numeric_series.dropna())]
        if valid_series.empty:
            return np.nan # Devolver NaN si no quedan valores válidos

        result = agg_func(valid_series)

        # Devolver NaN si el resultado de la agregación es NaN o Infinito
        if pd.isna(result) or not np.isfinite(result):
            return np.nan

        # Intentar convertir tipos NumPy a tipos Python estándar si es posible
        try:
            if isinstance(result, (np.int_, np.integer)): return int(result)
            elif isinstance(result, (np.float_, np.floating)): return float(result)
            else: return result # Devolver como está si no es un tipo numpy conocido
        except Exception:
            return result # Devolver resultado original si la conversión falla
    except Exception as e:
        # Loguear advertencia si la agregación falla
        logger.warning(f"Error aplicando agregación '{agg_func.__name__}' a serie (primeros elems: {series.head().tolist()}...): {e}")
        return np.nan

def get_last_or_value(data_dict: Dict[str, List[Any]], key: str, default: Any = np.nan) -> Any:
    """
    Helper para obtener el último valor válido de una lista en el dict,
    o el valor si no es lista, o default si no se encuentra/es inválido.
    Maneja NaN/inf para valores numéricos, y acepta strings/otros tipos.
    """
    value = data_dict.get(key)
    last_valid_value = default

    if isinstance(value, list):
        # Iterar hacia atrás para encontrar el último valor no None
        found = False
        for v in reversed(value):
            if v is not None:
                # --- Corrección: Chequear isfinite SOLO si es numérico ---
                is_num = isinstance(v, (int, float, np.number))
                if is_num:
                    # Si es numérico, chequear finitud
                    if pd.notna(v) and np.isfinite(v):
                        last_valid_value = v
                        found = True
                        break # Encontrado el último numérico válido
                else:
                    # Si no es numérico (e.g., string), considerarlo válido si no es None
                    last_valid_value = v
                    found = True
                    break # Encontrado el último no-None
        # Si el bucle termina sin encontrar nada (lista vacía o solo None), se mantiene el default
    elif value is not None:
        # Si no es lista pero existe, chequear validez
        is_num = isinstance(value, (int, float, np.number))
        if is_num:
            # Chequear finitud si es numérico
            if pd.notna(value) and np.isfinite(value):
                last_valid_value = value
        else:
            # Aceptar si no es numérico (e.g., string 'unknown')
            last_valid_value = value
    # else: Si la clave no existe o es None/NaN/Inf, devolver default

    # Intentar convertir tipos NumPy a Python estándar si aplica (después de encontrar el valor)
    try:
        if isinstance(last_valid_value, (np.int_, np.integer)): return int(last_valid_value)
        elif isinstance(last_valid_value, (np.float_, np.floating)): return float(last_valid_value)
        elif isinstance(last_valid_value, np.bool_): return bool(last_valid_value) # Añadir bool
    except Exception: pass # Ignorar errores de conversión

    return last_valid_value


def summarize_episode(episode_data: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Genera un diccionario resumen completo para los datos detallados de un solo episodio.
    Calcula estadísticas agregadas y extrae valores finales/representativos.

    Args:
        episode_data (Dict[str, List[Any]]): Diccionario de métricas del episodio.

    Returns:
        Dict[str, Any]: Diccionario con estadísticas resumen.
    """
    # 2.3: Validar input básico
    if not isinstance(episode_data, dict):
        logger.error("summarize_episode recibió datos no diccionario. Devolviendo resumen vacío.")
        return {'episode': -1, 'error': 'Invalid input data format'}
    if not episode_data:
         logger.warning("summarize_episode recibió diccionario vacío. Devolviendo resumen vacío.")
         return {'episode': -1, 'warning': 'Empty input data'}

    summary: Dict[str, Any] = {}
    # Obtener episode ID (SimulationManager lo añade antes de llamar)
    # Usar el helper get_last_or_value que maneja listas/valores únicos
    summary['episode'] = get_last_or_value(episode_data, 'episode', -1) # Default -1 si falta

    # --- Campos de Resumen Directo (último valor válido o pre-calculado) ---
    # Usar el helper get_last_or_value
    summary['termination_reason'] = get_last_or_value(episode_data, 'termination_reason', 'unknown')
    summary['episode_time'] = get_last_or_value(episode_data, 'episode_time', np.nan) # Añadido por SimMan
    summary['total_reward'] = get_last_or_value(episode_data, 'total_reward', np.nan) # Añadido por SimMan
    summary['avg_stability_score'] = get_last_or_value(episode_data, 'avg_stability_score', np.nan) # Añadido por SimMan
    summary['avg_w_stab_kp_cf'] = get_last_or_value(episode_data, 'virtual_w_stab_kp_cf', np.nan)
    summary['avg_w_stab_ki_cf'] = get_last_or_value(episode_data, 'virtual_w_stab_ki_cf', np.nan)
    summary['avg_w_stab_kd_cf'] = get_last_or_value(episode_data, 'virtual_w_stab_kd_cf', np.nan)
    summary['final_epsilon'] = get_last_or_value(episode_data, 'epsilon', np.nan)
    summary['final_learning_rate'] = get_last_or_value(episode_data, 'learning_rate', np.nan)
    summary['episode_duration_s'] = get_last_or_value(episode_data, 'episode_duration_s', np.nan) # Añadido por SimMan
    summary['total_agent_decisions'] = get_last_or_value(episode_data, 'total_agent_decisions', 0) # Añadido por SimMan
    summary['final_kp'] = get_last_or_value(episode_data, 'final_kp', np.nan) # Añadido por SimMan
    summary['final_ki'] = get_last_or_value(episode_data, 'final_ki', np.nan) # Añadido por SimMan
    summary['final_kd'] = get_last_or_value(episode_data, 'final_kd', np.nan) # Añadido por SimMan
    summary['performance'] = get_last_or_value(episode_data, 'performance', np.nan) # Añadido por SimMan


    # --- Métricas a EXCLUIR de la agregación estadística (mean, std, min, max) ---
    # 2.4: Definir métricas que *no* deben agregarse estadísticamente
    metrics_to_exclude_from_agg = {
        # Identificadores y descriptores
        'episode', 'time', 'id_agent_decision', 'termination_reason',
        # Valores finales ya extraídos
        'epsilon', 'learning_rate', 'final_kp', 'final_ki', 'final_kd',
        'episode_duration_s', 'total_agent_decisions', 'episode_time',
        # Valores agregados ya calculados
        'total_reward', 'avg_stability_score', 'performance', 'cumulative_reward',
        # Pasos de ganancia (usualmente constantes o logueados al final)
        'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd',
        # Recompensas y Estabilidades virtuales (último valor ya extraído)
        'virtual_reward_kp', 'virtual_reward_ki', 'virtual_reward_kd',
        'avg_w_stab_kp_cf', 'avg_w_stab_ki_cf', 'avg_w_stab_kd_cf',
        # Contadores de visita (su valor final puede ser más útil que la media)
        'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd',
        # Acciones discretas (la media no tiene sentido)
        'action_kp', 'action_ki', 'action_kd',
        # Podrían añadirse otras si su agregación no es informativa
    }
    # No es necesario extender con claves no presentes, la lógica de abajo lo maneja.

    # --- Estadísticas Agregadas (Media, Std, Min, Max) usando _safe_agg ---
    for metric, values in episode_data.items():
        # Saltar si es una métrica excluida o si el valor no es una lista
        if metric in metrics_to_exclude_from_agg or not isinstance(values, list):
            continue

        # Convertir lista a Serie de Pandas para usar _safe_agg
        try:
            s = pd.Series(values)
        except Exception as e:
            logger.warning(f"Error creando Serie Pandas para métrica '{metric}' en ep {summary['episode']}. Saltando agregación. Error: {e}")
            summary[f'{metric}_mean'] = np.nan; summary[f'{metric}_std'] = np.nan
            summary[f'{metric}_min'] = np.nan; summary[f'{metric}_max'] = np.nan
            continue

        # Calcular agregados usando el helper seguro
        summary[f'{metric}_mean'] = _safe_agg(s, np.mean)
        summary[f'{metric}_std'] = _safe_agg(s, np.std)
        summary[f'{metric}_min'] = _safe_agg(s, np.min)
        summary[f'{metric}_max'] = _safe_agg(s, np.max)

    # --- Añadir Valores Finales Representativos (opcional, puede ser redundante) ---
    # Si se desea explícitamente el último valor de algunas métricas no agregadas
    # (aunque get_last_or_value ya lo hizo para las principales)
    metrics_final_value_explicit = [
         'gain_step', 'gain_step_kp', 'gain_step_ki', 'gain_step_kd', # Último valor del paso
         'q_visit_count_state_kp', 'q_visit_count_state_ki', 'q_visit_count_state_kd', # Último conteo
         # ... añadir otras si es necesario ...
    ]
    adaptive_vars = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity']
    for var in adaptive_vars: metrics_final_value_explicit.extend([f'adaptive_mu_{var}', f'adaptive_sigma_{var}'])

    for metric in metrics_final_value_explicit:
        # Solo añadir si no fue agregado como mean/std/min/max
        if f'{metric}_mean' not in summary:
            # Usar el helper para obtener el último valor válido
            summary[metric] = get_last_or_value(episode_data, metric, np.nan)

    # 2.5: Devolver el diccionario de resumen
    return summary