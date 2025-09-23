# utils/data_processing.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple, Set

logger_dp_mod = logging.getLogger(__name__) # Renombrado para evitar colisión con logger global

def _safe_aggregate_series(data_series: Optional[pd.Series], 
                           aggregation_func_call: Callable, # Renombrado para claridad
                           indicator_id: str = "" # Renombrado
                          ) -> Any:
    """Aplica una función de agregación a una pd.Series de forma segura."""
    if data_series is None or data_series.empty:
        return np.nan
    
    # Convertir a numérico, errores a NaN, luego quitar NaNs y no-finitos
    numeric_series_clean = pd.to_numeric(data_series, errors='coerce').dropna()
    numeric_series_clean = numeric_series_clean[np.isfinite(numeric_series_clean)]
        
    if numeric_series_clean.empty:
        return np.nan

    # Manejo especial para percentiles si la función de agregación los espera como string/float
    # (ej. pd.Series.quantile(q=0.25))
    if indicator_id.startswith('p') and indicator_id[1:].isdigit(): # p25, p50, p75
        try:
            quantile_val_float = float(indicator_id[1:]) / 100.0
            # Usar pd.Series.quantile si la función lo permite o pasarla directamente
            if hasattr(aggregation_func_call, '__name__') and aggregation_func_call.__name__ == '<lambda>': # Si es una lambda para quantile
                 result_agg = aggregation_func_call(numeric_series_clean) # La lambda ya tiene el cuantil
            elif callable(getattr(numeric_series_clean, 'quantile', None)): # Si la serie tiene quantile
                 result_agg = numeric_series_clean.quantile(quantile_val_float)
            else: # Fallback a np.percentile si agg_func es np.percentile y se pasó el cuantil
                 result_agg = aggregation_func_call(numeric_series_clean, float(indicator_id[1:]))
        except Exception: # Fallback si el manejo de percentil falla
            result_agg = aggregation_func_call(numeric_series_clean)
    else:
        result_agg = aggregation_func_call(numeric_series_clean)

    if pd.isna(result_agg) or not np.isfinite(result_agg): return np.nan
    
    # Intentar convertir tipos de NumPy a tipos nativos de Python para mejor serialización JSON
    if isinstance(result_agg, (np.integer, np.int_)): return int(result_agg)
    if isinstance(result_agg, (np.floating, np.float64)): return float(result_agg)
    return result_agg


def get_last_valid_value_from_list(
    metrics_data: Dict[str, List[Any]],
    metric_key: str,
    default_val: Any = np.nan # Renombrado
) -> Any:
    """Obtiene el último valor válido de una lista de métricas."""
    metric_list_values = metrics_data.get(metric_key)
    last_valid_val = default_val

    if isinstance(metric_list_values, list):
        for val_item in reversed(metric_list_values): # Iterar desde el final
            if val_item is not None:
                if isinstance(val_item, (int, float, np.number)): # Si es numérico
                    if pd.notna(val_item) and np.isfinite(val_item):
                        last_valid_val = val_item; break
                else: # Si no es numérico (ej. string, bool), tomar el último no-None
                    last_valid_val = val_item; break
    elif metric_list_values is not None: # Si no es lista pero no es None (valor único)
        if isinstance(metric_list_values, (int, float, np.number)):
            if pd.notna(metric_list_values) and np.isfinite(metric_list_values):
                last_valid_val = metric_list_values
        else:
            last_valid_val = metric_list_values
    
    # Conversión de tipos NumPy a Python nativos
    if isinstance(last_valid_val, (np.integer, np.int_)): return int(last_valid_val)
    if isinstance(last_valid_val, (np.floating, np.float64)): return float(last_valid_val)
    if isinstance(last_valid_val, np.bool_): return bool(last_valid_val)
    return last_valid_val

def summarize_episode(
    detailed_metrics: Dict[str, List[Any]],
    summary_directives: Tuple[Set[str], Set[str]],
    global_summary_config: Dict[str, Any]
) -> Dict[str, Any]:
    
    if not global_summary_config.get('enabled_summary', False):
        return {}

    ep_summary: Dict[str, Any] = {}
    
    # --- 1. Usar directivas pre-procesadas ---
    direct_cols_keys, stat_cols_keys = summary_directives

    # --- 2. Extraer valores directos para el resumen ---
    # Para las columnas directas, tomamos el ÚLTIMO valor válido de la serie temporal.
    for col_key in direct_cols_keys:
        ep_summary[col_key] = get_last_valid_value_from_list(detailed_metrics, col_key)

    # --- 3. Calcular métricas derivadas y añadirlas directamente al resumen ---
    times = detailed_metrics.get('time', [])
    if times:
        sim_duration = times[-1]
        # La clave 'time_duration' para el summary se calcula aquí, no se extrae.
        if 'time_duration' in direct_cols_keys:
            ep_summary['time_duration'] = sim_duration
        
        total_reward = np.nansum(detailed_metrics.get('reward', [0.0]))
        ep_summary['total_reward'] = total_reward
        ep_summary['performance'] = total_reward / sim_duration if sim_duration > 1e-9 else 0.0
        
        stability_scores = detailed_metrics.get('stability_score')
        if stability_scores and any(pd.notna(s) for s in stability_scores):
            ep_summary['avg_stability_score'] = np.nanmean(stability_scores)
        else:
            ep_summary['avg_stability_score'] = np.nan

    # --- 4. Calcular y extraer estadísticas agregadas ---
    if global_summary_config.get('enabled_summary_stats', False):
        stat_indicators = global_summary_config.get('statistics_metrics', [])
        agg_funcs: Dict[str, Callable] = {
            '_mean': np.nanmean, '_sigma': np.nanstd, '_min': np.nanmin,
            'p25': lambda s: np.nanpercentile(s, 25) if not s.empty and not s.isnull().all() else np.nan,
            'p50': lambda s: np.nanpercentile(s, 50) if not s.empty and not s.isnull().all() else np.nan,
            'p75': lambda s: np.nanpercentile(s, 75) if not s.empty and not s.isnull().all() else np.nan,
            '_max': np.nanmax
        }
        
        for metric_key in stat_cols_keys:
            if metric_key in detailed_metrics:
                data_series = pd.to_numeric(pd.Series(detailed_metrics[metric_key]), errors='coerce')
                for indicator_id in stat_indicators:
                    if indicator_id in agg_funcs:
                        summary_col_name = f"{metric_key}{indicator_id}"
                        ep_summary[summary_col_name] = _safe_aggregate_series(data_series, agg_funcs[indicator_id], indicator_id)

    # --- 5. Reordenar columnas del resumen final ---
    first_cols = global_summary_config.get('summary_first_cols', [])
    ordered_summary = {col: ep_summary.pop(col) for col in first_cols if col in ep_summary}
    remaining_keys_sorted = sorted(ep_summary.keys())
    for key in remaining_keys_sorted:
        ordered_summary[key] = ep_summary[key]
            
    return ordered_summary