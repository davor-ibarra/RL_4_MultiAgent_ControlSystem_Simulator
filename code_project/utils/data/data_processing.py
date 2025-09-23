# utils/data_processing.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Callable, Optional

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
    detailed_metrics: Dict[str, List[Any]], # Métricas detalladas del episodio
    summary_directives_config: Dict[str, Any] # Directivas de 'processed_data_directives'
) -> Dict[str, Any]:
    """Genera un diccionario resumen para un episodio, usando directivas."""
    # logger_dp_mod.debug(f"Summarizing episode. Directives: {summary_directives_config.keys()}")
    ep_summary: Dict[str, Any] = {}
    
    # Asegurar que 'episode' siempre esté, si está en las métricas detalladas
    ep_summary['episode'] = get_last_valid_value_from_list(detailed_metrics, 'episode', -1)

    # 1. Columnas Directas (último valor válido)
    direct_cols = summary_directives_config.get('summary_direct_columns', [])
    for col_key_direct in direct_cols:
        if col_key_direct == 'episode': continue # Ya añadido
        if col_key_direct in detailed_metrics:
            ep_summary[col_key_direct] = get_last_valid_value_from_list(detailed_metrics, col_key_direct)
        # else: logger_dp_mod.debug(f"Direct summary key '{col_key_direct}' not in detailed_metrics.")

    # 2. Estadísticas Agregadas
    if summary_directives_config.get('summary_stats_enabled', False):
        stat_metric_keys = summary_directives_config.get('summary_stat_columns', [])
        stat_indicators_list = summary_directives_config.get('summary_stat_indicators', [])
        
        # Mapeo de indicadores a funciones (simplificado)
        agg_funcs: Dict[str, Callable] = {
            '_mean': np.nanmean, '_sigma': np.nanstd, '_min': np.nanmin, '_max': np.nanmax,
            'p50': np.nanmedian, # p50 es mediana
            # Lambdas para percentiles que usan np.nanpercentile
            'p25': lambda s: np.nanpercentile(s, 25) if len(s[~np.isnan(s)]) > 0 else np.nan,
            'p75': lambda s: np.nanpercentile(s, 75) if len(s[~np.isnan(s)]) > 0 else np.nan,
        }
        
        for metric_key_for_stat in stat_metric_keys:
            if metric_key_for_stat in detailed_metrics:
                data_list_for_stat = detailed_metrics[metric_key_for_stat]
                if not isinstance(data_list_for_stat, list): continue # Solo procesar listas

                # Crear pd.Series para facilitar agregaciones y manejo de NaNs
                # Convertir a numérico aquí, errores a NaN
                pd_series_data = pd.to_numeric(pd.Series(data_list_for_stat), errors='coerce')

                for indicator_str in stat_indicators_list:
                    agg_function_to_call = agg_funcs.get(indicator_str)
                    if agg_function_to_call:
                        summary_col_name = f"{metric_key_for_stat}{indicator_str}"
                        # _safe_aggregate_series ya maneja NaNs y tipos numéricos
                        ep_summary[summary_col_name] = _safe_aggregate_series(pd_series_data, agg_function_to_call, indicator_str)
                    # else: logger_dp_mod.warning(f"Unknown indicator '{indicator_str}' for metric '{metric_key_for_stat}'.")
            # else: logger_dp_mod.debug(f"Stat column key '{metric_key_for_stat}' not in detailed_metrics.")

    # 3. Procesamiento Adicional (ej. métricas de ET que son directas pero se basan en gain_id)
    #    Se asume que si 'patience_M_kp' está en 'summary_direct_columns', ya se extrajo.
    #    Si se necesitan agregaciones de métricas de ET (ej. promedio de 'improvement_metric_value_kp'),
    #    deben estar en 'summary_stat_columns' y 'summary_stat_indicators'.
    #    El '_agent_defining_vars' puede ser útil si se quiere iterar y construir nombres de métricas dinámicamente.
    #    Por ahora, la lógica de arriba es genérica y se basa en los nombres exactos en las directivas.

    return ep_summary