import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any, Optional

import gc # Import garbage collector

# 1.1: Usar logger específico (inyectado o a nivel de módulo)
logger = logging.getLogger(__name__)

class HeatmapGenerator:
    """
    Genera datos numéricos agregados para heatmaps y los guarda en hojas de Excel.
    Usa 'name' de la config del plot como nombre de hoja Excel.
    Aplica filtrado por 'termination_reason' si está especificado en la config del plot.
    """
    def __init__(self, logger_instance: logging.Logger):
        # 1.2: Usar el logger inyectado
        self._logger = logger_instance
        if not isinstance(logger_instance, logging.Logger):
            # Usar logger global temporalmente para el error si la inyección falla
            logger.error("[HeatmapGenerator] Logger inválido proporcionado en init.")
            raise TypeError("Logger inválido.")
        self._logger.info("[HeatmapGenerator] HeatmapGenerator instance created.")

    def _load_and_prepare_data(self, results_folder: str) -> Optional[pd.DataFrame]:
        """
        Carga datos detallados de todos los archivos JSON de episodios,
        los alinea temporalmente y los combina en un único DataFrame.
        Asegura que la columna 'termination_reason' esté presente si existe.
        """
        all_aligned_data: List[pd.DataFrame] = []
        total_files_processed = 0
        total_episodes_processed = 0
        total_timesteps_loaded = 0

        self._logger.info(f"[HeatmapGenerator_load_and_prepare_data] Buscando archivos simulation_data_ep_*.json en: {results_folder}")
        try:
            # 1.3: Encontrar y ordenar archivos JSON
            files = [f for f in os.listdir(results_folder) if f.startswith("simulation_data_ep_") and f.endswith(".json")]
            if not files:
                self._logger.warning("[HeatmapGenerator_load_and_prepare_data] No se encontraron archivos de datos detallados ('simulation_data_ep_*.json'). No se pueden generar datos de heatmap.")
                return None
            # Ordenar por el número inicial del episodio en el nombre del archivo
            try:
                files.sort(key=lambda name: int(name.split('_ep_')[-1].split('_to_')[0]))
            except (ValueError, IndexError):
                self._logger.warning("[HeatmapGenerator_load_and_prepare_data] No se pudo ordenar los archivos de datos por número de episodio. Procesando en orden del sistema de archivos.")

            self._logger.info(f"[HeatmapGenerator_load_and_prepare_data] Encontrados {len(files)} archivos de datos detallados para cargar.")

            # 1.4: Cargar datos de todos los archivos
            raw_episode_list_from_files: List[Dict] = []
            for filename in files:
                filepath = os.path.join(results_folder, filename)
                self._logger.debug(f"[HeatmapGenerator_load_and_prepare_data] Cargando archivo: {filename}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        # Cada archivo contiene una LISTA de diccionarios de episodios
                        episodes_in_file = json.load(f)
                    if isinstance(episodes_in_file, list):
                        # Validar que la lista contenga diccionarios
                        valid_episodes = [ep for ep in episodes_in_file if isinstance(ep, dict)]
                        if len(valid_episodes) != len(episodes_in_file):
                            self._logger.warning(f"[HeatmapGenerator_load_and_prepare_data] Archivo {filename} contenía elementos que no eran diccionarios.")
                        raw_episode_list_from_files.extend(valid_episodes)
                        total_files_processed += 1
                    else:
                        self._logger.warning(f"[HeatmapGenerator_load_and_prepare_data] Archivo {filename} no contiene una lista de episodios como se esperaba. Saltando archivo.")
                except json.JSONDecodeError as e_json:
                    self._logger.error(f"[HeatmapGenerator_load_and_prepare_data] Error decodificando JSON en {filename}: {e_json}. Saltando archivo.")
                except Exception as e_load:
                    self._logger.error(f"[HeatmapGenerator_load_and_prepare_data] Error cargando o procesando {filename}: {e_load}. Saltando archivo.")

            if not raw_episode_list_from_files:
                self._logger.warning("[HeatmapGenerator_load_and_prepare_data] No se cargaron datos válidos de ningún archivo de episodios.")
                return None

            # 1.5: Procesar y alinear cada episodio en un DataFrame
            for i, episode_dict in enumerate(raw_episode_list_from_files):
                # Extraer ID de episodio (usar el primero si es lista, o default)
                episode_id_val = episode_dict.get('episode')
                if isinstance(episode_id_val, list): episode_id_val = episode_id_val[0] if episode_id_val else f'ep_idx_{i}'
                elif episode_id_val is None: episode_id_val = f'ep_idx_{i}'

                # Usar 'time' como referencia de longitud
                time_values = episode_dict.get('time')
                if not isinstance(time_values, list) or not time_values:
                    self._logger.warning(f"[HeatmapGenerator_load_and_prepare_data] Episodio {episode_id_val} no tiene datos de 'time' válidos. Saltando episodio.")
                    continue
                num_steps = len(time_values)
                reference_index = pd.Index(range(num_steps)) # Índice para alinear

                # Crear DataFrame temporal para el episodio
                temp_data = {'time': pd.Series(time_values, index=reference_index)}

                # Iterar sobre todas las métricas del episodio
                for metric, values in episode_dict.items():
                    if metric == 'time': continue # Ya procesado

                    if isinstance(values, list):
                        # Si es una lista, crear Serie y alinear si es necesario
                        if len(values) == num_steps:
                            temp_data[metric] = pd.Series(values, index=reference_index)
                        else:
                            # Loguear advertencia sobre desalineación
                            self._logger.debug(f"[HeatmapGenerator_load_and_prepare_data] Ep {episode_id_val}: Métrica '{metric}' (len {len(values)}) desalineada con 'time' (len {num_steps}). Alineando...")
                            aligned_series = pd.Series(index=reference_index, dtype=object) # dtype=object para manejar tipos mixtos
                            valid_len = min(len(values), num_steps)
                            aligned_series.iloc[:valid_len] = values[:valid_len] # Copiar datos hasta la longitud mínima
                            temp_data[metric] = aligned_series
                    elif values is not None:
                        # Si no es lista (valor único), replicarlo para toda la longitud
                        temp_data[metric] = pd.Series([values] * num_steps, index=reference_index)
                    # Ignorar métricas con valor None

                # 1.6: Crear DataFrame y añadir ID de episodio
                try:
                    # Convertir columnas relevantes a numérico donde sea posible, manteniendo otras
                    episode_df = pd.DataFrame(temp_data)
                    # Añadir columna de episodio (asegurarse de que no sea lista)
                    episode_df['episode'] = episode_id_val
                    # Asegurar que 'termination_reason' exista, aunque sea con NaNs si no estaba
                    if 'termination_reason' not in episode_df.columns:
                        episode_df['termination_reason'] = pd.NA

                    all_aligned_data.append(episode_df)
                    total_episodes_processed += 1
                except Exception as e_df:
                    self._logger.error(f"[HeatmapGenerator_load_and_prepare_data] Error creando DataFrame para episodio {episode_id_val}: {e_df}")

            if not all_aligned_data:
                self._logger.warning("[HeatmapGenerator_load_and_prepare_data] No se pudieron crear DataFrames alineados para ningún episodio.")
                return None

            del raw_episode_list_from_files # Liberar memoria de la lista cruda
            gc.collect() # Forzar recolección

            if not all_aligned_data:
                self._logger.warning("[HeatmapGenerator_load_and_prepare_data] No se pudieron crear DataFrames alineados para ningún episodio.")
                return None

            # 1.7: Concatenar todos los DataFrames de episodios
            df_combined = pd.concat(all_aligned_data, ignore_index=True)
            del all_aligned_data # Liberar memoria de la lista de dataframes individuales
            gc.collect() # Forzar recolección
            
            total_timesteps_loaded = len(df_combined)
            self._logger.info(f"[HeatmapGenerator_load_and_prepare_data] Datos cargados y aplanados/alineados: {total_timesteps_loaded} timesteps de {total_episodes_processed} episodios en {total_files_processed} archivos.")
            # Loguear columnas disponibles para depuración
            self._logger.debug(f"[HeatmapGenerator_load_and_prepare_data] Columnas disponibles en DataFrame combinado: {df_combined.columns.tolist()}")
            return df_combined

        except OSError as e_os:
            self._logger.error(f"[HeatmapGenerator_load_and_prepare_data] Error de sistema de archivos listando/accediendo a {results_folder}: {e_os}")
            return None
        except Exception as e_main:
            self._logger.error(f"[HeatmapGenerator_load_and_prepare_data] Error inesperado durante carga/preparación de datos detallados: {e_main}", exc_info=True)
            return None

    def _calculate_heatmap_grid(self, df: pd.DataFrame, config: Dict) -> Optional[pd.DataFrame]:
        """Calcula la cuadrícula de valores agregados para un heatmap, aplicando filtrado."""
        plot_name_for_log = config.get('name', f"idx_{config.get('_internal_plot_index','?')}")
        heatmap_table: Optional[pd.DataFrame] = None
        try:
            # 1.8: Validar configuración requerida
            required_cfg_keys = ['x_variable', 'y_variable', 'value_variable']
            if not all(k in config for k in required_cfg_keys):
                missing_keys = [k for k in required_cfg_keys if k not in config]
                self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap config (plot='{plot_name_for_log}') Faltan claves: {missing_keys}.")
                raise ValueError(f"[HeatmapGenerator_calculate_heatmap_grid] Configuración de heatmap incompleta: faltan {missing_keys}")

            x_var, y_var, value_var = config['x_variable'], config['y_variable'], config['value_variable']
            # Leer 'aggregation' y 'bins' desde la sub-config 'config' si existe, o desde el nivel superior
            plot_style_config = config.get('config', {}) # Obtener el sub-diccionario 'config'
            agg_func_name = plot_style_config.get('aggregation', 'count').lower() # Leer de 'config'
            num_bins_cfg = plot_style_config.get('bins', 50)
            # Asegurar que num_bins sea un entero > 0
            try:
                 num_bins = int(num_bins_cfg)
                 if num_bins <= 0:
                      self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): 'bins' debe ser > 0, recibido {num_bins_cfg}. Usando default 50.")
                      num_bins = 50
            except (ValueError, TypeError):
                 self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): 'bins' inválido ({num_bins_cfg}). Usando default 50.")
                 num_bins = 50
            # Leer filtro desde sub-config 'config' también
            filter_reasons = plot_style_config.get('filter_termination_reason')
            # Leer Límites de Configuración
            xmin, xmax = plot_style_config.get('xmin'), plot_style_config.get('xmax') 
            ymin, ymax = plot_style_config.get('ymin'), plot_style_config.get('ymax') 
            logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Límites config: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]") 

            # 1.9: Validar columnas necesarias en el DataFrame
            required_df_cols = list(set([x_var, y_var, value_var]))
            # Añadir 'termination_reason' solo si se va a filtrar
            if filter_reasons and isinstance(filter_reasons, list):
                required_df_cols.append('termination_reason')

            missing_cols = [col for col in required_df_cols if col not in df.columns]
            if missing_cols:
                # Distinguir si falta 'termination_reason' cuando se necesita para filtrar
                if 'termination_reason' in missing_cols and filter_reasons:
                    self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Se especificó filter_termination_reason pero la columna 'termination_reason' falta en los datos detallados. El filtro no se aplicará.")
                    # Quitarla de las requeridas para no fallar, pero el filtro no se aplicará
                    required_df_cols.remove('termination_reason')
                    filter_reasons = None # Desactivar el filtro
                # Comprobar si todavía faltan columnas esenciales (x, y, value)
                missing_essential = [col for col in [x_var, y_var, value_var] if col not in df.columns]
                if missing_essential:
                    self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Faltan columnas esenciales {missing_essential}. Columnas disponibles: {df.columns.tolist()}")
                    raise ValueError(f"[HeatmapGenerator_calculate_heatmap_grid] Columnas esenciales para heatmap no encontradas: {missing_essential}")

            # 1.10: Filtrar por termination_reason ANTES de procesar numéricamente
            df_filtered = df.copy()
            if filter_reasons and isinstance(filter_reasons, list) and 'termination_reason' in df_filtered.columns:
                rows_before_filter = len(df_filtered)
                # Asegurarse de que la columna no tenga NaNs antes de isin
                df_filtered = df_filtered[df_filtered['termination_reason'].isin(filter_reasons)]
                rows_after_filter = len(df_filtered)
                self._logger.info(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Filtrado por {filter_reasons}. Datos: {rows_after_filter}/{rows_before_filter} timesteps.")
                if df_filtered.empty:
                    self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): No quedan datos después de filtrar por termination_reason. No se generará heatmap.")
                    return None
            elif filter_reasons and isinstance(filter_reasons, list) and 'termination_reason' not in df_filtered.columns:
                self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): filter_termination_reason especificado pero la columna no está presente. No se aplicará filtro.")


            # 1.11: Limpiar datos numéricos (solo columnas x, y, value)
            df_numeric = pd.DataFrame(index=df_filtered.index) # Crear DataFrame vacío con el mismo índice
            cols_to_convert = [x_var, y_var, value_var]
            valid_conversion = True
            for col in cols_to_convert:
                if col in df_filtered.columns:
                    # Convertir cada columna individualmente
                    df_numeric[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                else:
                    # Esto no debería pasar si la validación inicial fue correcta, pero por seguridad
                    self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Columna '{col}' inesperadamente ausente antes de conversión numérica.")
                    valid_conversion = False
                    break
            if not valid_conversion: return None

            rows_before_drop = df_numeric.shape[0]
            # Eliminar filas donde CUALQUIERA de las 3 columnas sea NaN o Infinito
            df_numeric.dropna(subset=[x_var, y_var, value_var], inplace=True)
            df_numeric = df_numeric[np.isfinite(df_numeric[[x_var, y_var, value_var]]).all(axis=1)]
            rows_after_drop = df_numeric.shape[0]

            if df_numeric.empty:
                self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Sin datos numéricos válidos después de la limpieza (NaN/inf).")
                return None
            self._logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Datos numéricos válidos {rows_after_drop}/{rows_before_drop} para binning y agregación.")

            # Filtrar Datos por Límites ANTES de Binning
            df_filtered_for_binning = df_numeric.copy() # Usar copia
            rows_before_limit_filter = len(df_filtered_for_binning)
            applied_filters = []
            if xmin is not None: df_filtered_for_binning = df_filtered_for_binning[df_filtered_for_binning[x_var] >= xmin]; applied_filters.append(f"x>={xmin}")
            if xmax is not None: df_filtered_for_binning = df_filtered_for_binning[df_filtered_for_binning[x_var] <= xmax]; applied_filters.append(f"x<={xmax}")
            if ymin is not None: df_filtered_for_binning = df_filtered_for_binning[df_filtered_for_binning[y_var] >= ymin]; applied_filters.append(f"y>={ymin}")
            if ymax is not None: df_filtered_for_binning = df_filtered_for_binning[df_filtered_for_binning[y_var] <= ymax]; applied_filters.append(f"y<={ymax}")
            rows_after_limit_filter = len(df_filtered_for_binning)
            if df_filtered_for_binning.empty:
                logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Sin datos válidos después de filtrar por límites ({', '.join(applied_filters)}).")
                return None
            if applied_filters:
                logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Datos filtrados por límites ({', '.join(applied_filters)}) -> {rows_after_limit_filter}/{rows_before_limit_filter} para binning.")

            # Usar df_filtered_for_binning para el binning a partir de aquí
            x_col_data = df_filtered_for_binning[x_var].values
            y_col_data = df_filtered_for_binning[y_var].values

            # --- Binning ---
            x_bin_edges, y_bin_edges = None, None
            try:
                x_range_min = xmin if xmin is not None else np.min(x_col_data)
                x_range_max = xmax if xmax is not None else np.max(x_col_data)
                y_range_min = ymin if ymin is not None else np.min(y_col_data)
                y_range_max = ymax if ymax is not None else np.max(y_col_data)

                # Validar rangos antes de linspace
                tolerance_range = 1e-9 # Tolerancia pequeña para comparación de flotantes
                if (x_range_max - x_range_min) < tolerance_range:
                    logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Rango X nulo o negativo ({x_range_min} a {x_range_max}) después de aplicar límites/filtrado. No se puede generar heatmap.")
                    return None
                if (y_range_max - y_range_min) < tolerance_range:
                    logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Rango Y nulo o negativo ({y_range_min} a {y_range_max}) después de aplicar límites/filtrado. No se puede generar heatmap.")
                    return None

                # Crear los bordes explícitos usando linspace
                x_bin_edges = np.linspace(x_range_min, x_range_max, num_bins + 1)
                y_bin_edges = np.linspace(y_range_min, y_range_max, num_bins + 1)
                logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Bordes X definidos ({len(x_bin_edges)}): [{x_bin_edges[0]:.3f} ... {x_bin_edges[-1]:.3f}]")
                logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Bordes Y definidos ({len(y_bin_edges)}): [{y_bin_edges[0]:.3f} ... {y_bin_edges[-1]:.3f}]")
                # Usar pd.cut para obtener los índices de los bins y los bordes explícitos
                self._logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Aplicando pd.cut con bordes explícitos (num_bins={num_bins})")
                x_bin_indices = pd.cut(x_col_data, bins=x_bin_edges, labels=False, include_lowest=True, right=True, retbins=False)
                y_bin_indices = pd.cut(y_col_data, bins=y_bin_edges, labels=False, include_lowest=True, right=True, retbins=False)
                # Añadir los índices de bin al DataFrame filtrado
                # Asegurarse de usar el índice correcto del df filtrado
                df_numeric.loc[df_filtered_for_binning.index, f'{x_var}_bin'] = x_bin_indices
                df_numeric.loc[df_filtered_for_binning.index, f'{y_var}_bin'] = y_bin_indices

                # Eliminar filas donde el binning podría haber fallado (raro con include_lowest=True)
                if df_numeric[[f'{x_var}_bin', f'{y_var}_bin']].isnull().any().any():
                    self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Se encontraron NaNs en los índices de bin. Eliminando filas afectadas.")
                    df_numeric.dropna(subset=[f'{x_var}_bin', f'{y_var}_bin'], inplace=True)
                    if df_numeric.empty:
                        self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Sin datos después de eliminar NaNs de bins.")
                        return None
                # Convertir índices de bin a enteros
                df_numeric[f'{x_var}_bin'] = df_numeric[f'{x_var}_bin'].astype(int)
                df_numeric[f'{y_var}_bin'] = df_numeric[f'{y_var}_bin'].astype(int)

            except ValueError as e_cut:
                self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Error durante el binning (pd.cut) para heatmap '{plot_name_for_log}': {e_cut}. Asegurar que 'bins' ({num_bins}) sea válido para el rango de datos.")
                return None
            except Exception as e_bin:
                self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Error inesperado durante el binning para heatmap '{plot_name_for_log}': {e_bin}", exc_info=True)
                return None

            # --- Pivot Table para Agregación ---
            # Mapa de nombres de agregación a funciones/strings de Pandas
            agg_map = {
                'mean': 'mean', 'median': 'median', 'std': 'std', 'var': 'var',
                'sum': 'sum', 'count': 'size', # 'size' cuenta ocurrencias en el grupo
                'min': 'min', 'max': 'max',
                # Podrían añadirse funciones lambda si es necesario, pero es más complejo
            }
            agg_func_pd = agg_map.get(agg_func_name)
            if not agg_func_pd:
                self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Agregación '{agg_func_name}' no reconocida. Usando 'count'.")
                agg_func_pd = 'size'

            # 1.13: Seleccionar la columna de valores correcta para pivot
            # Si la agregación es 'count' (size), la columna 'values' es irrelevante,
            # pero pivot_table requiere una. Usamos x_var como placeholder.
            value_var_for_pivot = x_var if agg_func_pd == 'size' else value_var
 
            self._logger.debug(f"[HeatmapGenerator_calculate_heatmap_grid] Creando pivot_table para '{plot_name_for_log}' (X:{x_var}, Y:{y_var}, Value:{value_var_for_pivot}, Agg:{agg_func_pd}) usando índices de bin...")
            try:
                heatmap_table = pd.pivot_table(
                    df_numeric,
                    values=value_var_for_pivot, # Columna cuyos valores se agregan (o placeholder para 'size')
                    index=f'{y_var}_bin',     # Filas = bins de Y
                    columns=f'{x_var}_bin',   # Columnas = bins de X
                    aggfunc=agg_func_pd,      # Función de agregación
                    fill_value=np.nan         # Rellenar celdas sin datos con NaN
                )
                # Asegurar que los índices/columnas (que son bin indices 0, 1, ...) sean enteros
                heatmap_table.index = heatmap_table.index.astype(int)
                heatmap_table.columns = heatmap_table.columns.astype(int)

            except Exception as e_pivot:
                self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Error durante pd.pivot_table para heatmap '{plot_name_for_log}': {e_pivot}", exc_info=True)
                return None

            # --- Reindexar y Asignar Ejes con Centros de Bins ---
            # Crear índices completos para X e Y (0 a num_bins-1)
            full_x_bin_index = pd.RangeIndex(start=0, stop=num_bins, name=f'{x_var}_bin')
            full_y_bin_index = pd.RangeIndex(start=0, stop=num_bins, name=f'{y_var}_bin')

            # Reindexar la tabla para asegurar que tenga todas las celdas (rellenando con NaN si faltan)
            heatmap_table = heatmap_table.reindex(index=full_y_bin_index, columns=full_x_bin_index, fill_value=np.nan)

            # Calcular centros de los bins
            if x_bin_edges is None or y_bin_edges is None:
                self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Bin edges no están definidos después del binning. No se pueden calcular centros.")
                return None
            try:
                # Centros = promedio de los bordes de cada bin
                x_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
                y_centers = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2

                # Asegurar que tenemos la cantidad correcta de centros
                if len(x_centers) != num_bins or len(y_centers) != num_bins:
                    self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Discrepancia entre num_bins ({num_bins}) y número de centros calculados (X:{len(x_centers)}, Y:{len(y_centers)}). Usando índices numéricos.")
                    # Fallback: dejar los índices como 0, 1, ... si los centros fallan
                    heatmap_table.index.name = y_var # Nombrar los ejes correctamente
                    heatmap_table.columns.name = x_var
                else:
                    # Reemplazar los índices (0..N-1) con los valores de los centros de los bins
                    # Mantener el nombre del eje original (y_var, x_var)
                    heatmap_table.index = pd.Index(y_centers, name=y_var)
                    heatmap_table.columns = pd.Index(x_centers, name=x_var)

            except Exception as e_repr:
                self._logger.warning(f"[HeatmapGenerator_calculate_heatmap_grid] Heatmap ('{plot_name_for_log}'): Error asignando centros de bins como ejes: {e_repr}. Se usarán índices numéricos.", exc_info=True)
                # Nombrar ejes aunque se usen índices numéricos
                heatmap_table.index.name = y_var
                heatmap_table.columns.name = x_var

            self._logger.info(f"[HeatmapGenerator_calculate_heatmap_grid] Cuadrícula heatmap ('{plot_name_for_log}') calculada ({heatmap_table.shape[0]}x{heatmap_table.shape[1]}) usando '{agg_func_pd}'.")
            return heatmap_table

        except ValueError as ve: # Capturar errores de validación específicos
            self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Error de configuración o datos para heatmap '{plot_name_for_log}': {ve}")
            return None
        except Exception as e:
            self._logger.error(f"[HeatmapGenerator_calculate_heatmap_grid] Error inesperado calculando cuadrícula heatmap '{plot_name_for_log}': {e}", exc_info=True)
            return None

    def generate(self, results_folder: str, heatmap_configs: List[Dict], output_excel_filepath: str):
        """
        Genera un archivo Excel con hojas para cada heatmap configurado.
        Carga los datos detallados una sola vez.
        """
        # 1.14: Cargar datos detallados UNA SOLA VEZ
        df_all_data = self._load_and_prepare_data(results_folder)
        if df_all_data is None:
            self._logger.error("[HeatmapGenerator_generate] Fallo al cargar datos detallados. No se pueden generar datos para heatmaps.")
            return

        # 1.15: Preparar escritura a Excel
        writer = None
        generated_count = 0
        engine = None
        try:
            import openpyxl # Intentar importar para usar el engine
            engine = 'openpyxl'
            self._logger.info("[HeatmapGenerator_generate] Usando engine 'openpyxl' para escribir Excel.")
        except ImportError:
            self._logger.warning("[HeatmapGenerator_generate] Biblioteca 'openpyxl' no encontrada. Se usará el engine por defecto de Pandas (puede ser limitado o requerir otras dependencias). Instalar: pip install openpyxl")
            # Pandas usará xlsxwriter o similar si está disponible, o fallará

        sheets_to_write: Dict[str, pd.DataFrame] = {}
        processed_sheet_names: set = set()

        # 1.16: Iterar sobre configuraciones de heatmap
        for config in heatmap_configs:
            # Asegurar que config es un dict y tiene tipo 'heatmap'
            if not isinstance(config, dict) or config.get("type") != "heatmap":
                continue # Saltar si no es una config de heatmap válida

            # Extraer nombre de hoja y manejar colisiones
            plot_index = config.get('_internal_plot_index', '?') # Obtener índice interno
            plot_name = config.get('name')

            if plot_name:
                # Limitar longitud a 31 caracteres para Excel
                sheet_name = str(plot_name)[:31]
            else:
                # Usar nombre por defecto si no se proporciona 'name'
                sheet_name = f"heatmap_idx_{plot_index}"[:31]
                self._logger.warning(f"[HeatmapGenerator_generate] Heatmap config index={plot_index} no tiene 'name'. Usando nombre de hoja por defecto: '{sheet_name}'. Se recomienda añadir un 'name' único en la config YAML.")

            # Manejo de nombres de hoja duplicados
            original_sheet_name = sheet_name
            count = 1
            while sheet_name in processed_sheet_names:
                suffix = f"_{count}"
                max_len = 31 - len(suffix)
                sheet_name = original_sheet_name[:max_len] + suffix
                count += 1
                if count > 10: # Evitar bucle infinito si hay demasiados duplicados
                    self._logger.error(f"[HeatmapGenerator_generate] Demasiadas colisiones de nombre de hoja para '{original_sheet_name}'. Saltando este heatmap.")
                    sheet_name = None
                    break
            if sheet_name is None: continue # Saltar si no se pudo generar nombre único

            processed_sheet_names.add(sheet_name)
            log_name_ref = f"name='{plot_name}'" if plot_name else f"index={plot_index}"
            self._logger.info(f"[HeatmapGenerator_generate] Procesando config heatmap ({log_name_ref}) -> Hoja: '{sheet_name}'")

            # 1.17: Calcular la cuadrícula usando los datos ya cargados
            grid_df: Optional[pd.DataFrame] = None
            try:
                # Pasar una copia del DataFrame para evitar modificaciones accidentales entre cálculos
                grid_df = self._calculate_heatmap_grid(df_all_data.copy(), config)
            except Exception as e_calc:
                # Errores dentro de _calculate_heatmap_grid ya se loguean
                self._logger.error(f"[HeatmapGenerator_generate] Fallo irrecuperable calculando grid para hoja '{sheet_name}'. Error: {e_calc}", exc_info=True)

            # 1.18: Añadir a la lista para escribir si es válido
            if grid_df is not None and not grid_df.empty:
                sheets_to_write[sheet_name] = grid_df
                generated_count += 1
                self._logger.info(f"[HeatmapGenerator_generate] Datos para hoja '{sheet_name}' generados correctamente ({grid_df.shape[0]}x{grid_df.shape[1]}).")
            else:
                self._logger.warning(f"[HeatmapGenerator_generate] No se generaron datos válidos para la hoja '{sheet_name}'. No se incluirá en el archivo Excel.")

        # 1.19: Escribir todas las hojas válidas al archivo Excel
        if generated_count > 0:
            self._logger.info(f"[HeatmapGenerator_generate] Escribiendo {generated_count} hojas de datos heatmap en -> {output_excel_filepath}")
            try:
                # Inicializar ExcelWriter
                writer = pd.ExcelWriter(output_excel_filepath, engine=engine)
                # Escribir cada DataFrame a su hoja
                for sheet_name, df_to_write in sheets_to_write.items():
                    # Guardar CON índice y cabecera (representan los centros de bins)
                    df_to_write.to_excel(writer, sheet_name=sheet_name, index=True, header=True)
                # Cerrar (guardar) el archivo Excel
                # En versiones recientes de pandas, close() puede ser suficiente
                # writer.save() # Descomentar si writer.close() no guarda
                writer.close()
                self._logger.info("[HeatmapGenerator_generate] Archivo Excel con datos de heatmap guardado exitosamente.")
            except Exception as e_write:
                self._logger.error(f"[HeatmapGenerator_generate] Error escribiendo archivo Excel '{output_excel_filepath}': {e_write}", exc_info=True)
                # Ensure writer is closed even if writing fails mid-way
                if writer is not None and hasattr(writer, 'close'):
                    try:
                        # Check if 'close' is the correct method based on pandas version
                        # For newer pandas, close() might save and close. For older, save() might be needed.
                        # Let's assume close() handles it. If not, add writer.save() before close().
                        writer.close()
                        self._logger.info("[HeatmapGenerator_generate] ExcelWriter cerrado tras error de escritura.")
                    except Exception as e_close_err:
                        self._logger.error(f"[HeatmapGenerator_generate] Error cerrando ExcelWriter tras error de escritura: {e_close_err}")
            # finally: # No es necesario finally si usamos with, pero aquí no lo usamos
                # if writer is not None:
                #      try: writer.close()
                #      except: pass
        else:
            self._logger.warning("[HeatmapGenerator_generate] No se generaron hojas de datos heatmap válidas. No se creó el archivo Excel.")