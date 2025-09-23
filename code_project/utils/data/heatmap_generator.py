# utils/data/heatmap_generator.py
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any, Optional

import gc

logger_module_hm = logging.getLogger(__name__) # 'logger_module_hm'

class HeatmapGenerator:
    def __init__(self, injected_logger: logging.Logger): # 'injected_logger'
        self._internal_logger = injected_logger # '_internal_logger'
        if not isinstance(injected_logger, logging.Logger):
            logger_module_hm.error("[HeatmapGenerator] Invalid logger instance provided during initialization.")
            raise TypeError("A valid logging.Logger instance is required for HeatmapGenerator.")
        self._internal_logger.info("[HeatmapGenerator] HeatmapGenerator instance created.")

    def _load_and_prepare_detailed_data(self, output_data_root_path: str) -> Optional[pd.DataFrame]: # 'output_data_root_path'
        """
        Carga datos detallados de todos los archivos JSON de episodios (simulation_data_ep_*.json),
        los alinea temporalmente si es necesario, y los combina en un único DataFrame.
        """
        all_episode_dataframes_list: List[pd.DataFrame] = [] # 'all_episode_dataframes_list'
        num_files_processed = 0 # 'num_files_processed'
        num_episodes_loaded = 0 # 'num_episodes_loaded'
        total_timesteps_concatenated = 0 # 'total_timesteps_concatenated'

        self._internal_logger.info(f"[HeatmapGenerator:_load_and_prepare] Searching for simulation_data_ep_*.json files in: {output_data_root_path}")
        try:
            # Encontrar y ordenar archivos JSON de datos detallados
            # El nombre de archivo sigue siendo simulation_data_ep_...
            json_data_files_list = [f_name for f_name in os.listdir(output_data_root_path) if f_name.startswith("simulation_data_ep_") and f_name.endswith(".json")] # 'json_data_files_list', 'f_name'
            if not json_data_files_list:
                self._internal_logger.warning("[HeatmapGenerator:_load_and_prepare] No detailed episode data files ('simulation_data_ep_*.json') found. Cannot generate heatmap data.")
                return None
            
            try: # Intentar ordenar por el número de episodio en el nombre del archivo
                json_data_files_list.sort(key=lambda name_str: int(name_str.split('_ep_')[-1].split('_to_')[0])) # 'name_str'
            except (ValueError, IndexError):
                self._internal_logger.warning("[HeatmapGenerator:_load_and_prepare] Could not sort data files by episode number. Processing in filesystem order.")

            self._internal_logger.info(f"[HeatmapGenerator:_load_and_prepare] Found {len(json_data_files_list)} detailed data files to load.")

            # Cargar datos de todos los archivos JSON
            raw_episodes_from_all_files: List[Dict] = [] # 'raw_episodes_from_all_files'
            for json_filename_loop in json_data_files_list: # 'json_filename_loop'
                full_json_filepath = os.path.join(output_data_root_path, json_filename_loop) # 'full_json_filepath'
                self._internal_logger.debug(f"[HeatmapGenerator:_load_and_prepare] Loading file: {json_filename_loop}")
                try:
                    with open(full_json_filepath, 'r', encoding='utf-8') as f_json_in: # 'f_json_in'
                        episodes_list_in_file = json.load(f_json_in) # 'episodes_list_in_file'
                    if isinstance(episodes_list_in_file, list):
                        valid_episode_dicts = [ep_dict for ep_dict in episodes_list_in_file if isinstance(ep_dict, dict)] # 'valid_episode_dicts', 'ep_dict'
                        if len(valid_episode_dicts) != len(episodes_list_in_file):
                            self._internal_logger.warning(f"[HeatmapGenerator:_load_and_prepare] File {json_filename_loop} contained elements that were not dictionaries.")
                        raw_episodes_from_all_files.extend(valid_episode_dicts)
                        num_files_processed += 1
                    else:
                        self._internal_logger.warning(f"[HeatmapGenerator:_load_and_prepare] File {json_filename_loop} does not contain a list of episodes as expected. Skipping file.")
                except json.JSONDecodeError as e_json_dec_hm: # 'e_json_dec_hm'
                    self._internal_logger.error(f"[HeatmapGenerator:_load_and_prepare] JSON decoding error in {json_filename_loop}: {e_json_dec_hm}. Skipping file.")
                except Exception as e_load_json_hm: # 'e_load_json_hm'
                    self._internal_logger.error(f"[HeatmapGenerator:_load_and_prepare] Error loading or processing {json_filename_loop}: {e_load_json_hm}. Skipping file.")

            if not raw_episodes_from_all_files:
                self._internal_logger.warning("[HeatmapGenerator:_load_and_prepare] No valid episode data loaded from any files.")
                return None

            # Procesar y alinear cada episodio en un DataFrame
            for ep_idx, episode_raw_data_dict in enumerate(raw_episodes_from_all_files): # 'ep_idx', 'episode_raw_data_dict'
                # Extraer ID de episodio (usar el primero si es lista, o default)
                episode_id_from_data = episode_raw_data_dict.get('episode') # 'episode_id_from_data'
                if isinstance(episode_id_from_data, list): episode_id_from_data = episode_id_from_data[0] if episode_id_from_data else f'ep_idx_{ep_idx}'
                elif episode_id_from_data is None: episode_id_from_data = f'ep_idx_{ep_idx}'

                time_metric_values = episode_raw_data_dict.get('time') # 'time_metric_values'
                if not isinstance(time_metric_values, list) or not time_metric_values:
                    self._internal_logger.warning(f"[HeatmapGenerator:_load_and_prepare] Episode {episode_id_from_data} has no valid 'time' data. Skipping episode.")
                    continue
                
                num_timesteps_in_ep = len(time_metric_values) # 'num_timesteps_in_ep'
                # Usar un índice de referencia para alinear todas las series del episodio
                reference_pd_index = pd.Index(range(num_timesteps_in_ep)) # 'reference_pd_index'

                # Crear DataFrame temporal para el episodio actual
                temp_episode_df_data = {'time': pd.Series(time_metric_values, index=reference_pd_index)} # 'temp_episode_df_data'

                for metric_name_loop, metric_values_list_loop in episode_raw_data_dict.items(): # 'metric_name_loop', 'metric_values_list_loop'
                    if metric_name_loop == 'time': continue # Ya procesado

                    if isinstance(metric_values_list_loop, list):
                        if len(metric_values_list_loop) == num_timesteps_in_ep: # Si la longitud coincide, usar directamente
                            temp_episode_df_data[metric_name_loop] = pd.Series(metric_values_list_loop, index=reference_pd_index)
                        else: # Si no coincide, alinear rellenando con NA
                            self._internal_logger.debug(f"[HeatmapGenerator:_load_and_prepare] Ep {episode_id_from_data}: Metric '{metric_name_loop}' (len {len(metric_values_list_loop)}) misaligned with 'time' (len {num_timesteps_in_ep}). Aligning with NA padding...")
                            aligned_metric_series = pd.Series(index=reference_pd_index, dtype=object) # 'aligned_metric_series', dtype=object para tipos mixtos
                            common_length = min(len(metric_values_list_loop), num_timesteps_in_ep) # 'common_length'
                            aligned_metric_series.iloc[:common_length] = metric_values_list_loop[:common_length]
                            temp_episode_df_data[metric_name_loop] = aligned_metric_series
                    elif metric_values_list_loop is not None: # Si es un valor único (no lista), replicarlo
                        temp_episode_df_data[metric_name_loop] = pd.Series([metric_values_list_loop] * num_timesteps_in_ep, index=reference_pd_index)
                    # Ignorar métricas con valor None (no se añaden)

                try:
                    episode_df_obj = pd.DataFrame(temp_episode_df_data) # 'episode_df_obj'
                    episode_df_obj['episode'] = episode_id_from_data # Añadir columna de ID de episodio
                    # Asegurar que 'termination_reason' exista, aunque sea con NAs
                    if 'termination_reason' not in episode_df_obj.columns:
                        episode_df_obj['termination_reason'] = pd.NA

                    all_episode_dataframes_list.append(episode_df_obj)
                    num_episodes_loaded += 1
                except Exception as e_create_ep_df: # 'e_create_ep_df'
                    self._internal_logger.error(f"[HeatmapGenerator:_load_and_prepare] Error creating DataFrame for episode {episode_id_from_data}: {e_create_ep_df}")

            if not all_episode_dataframes_list:
                self._internal_logger.warning("[HeatmapGenerator:_load_and_prepare] No episode DataFrames could be created after processing files.")
                return None

            del raw_episodes_from_all_files # Liberar memoria
            gc.collect()

            # Concatenar todos los DataFrames de episodios en uno solo
            combined_data_df = pd.concat(all_episode_dataframes_list, ignore_index=True) # 'combined_data_df'
            del all_episode_dataframes_list # Liberar memoria
            gc.collect()
            
            total_timesteps_concatenated = len(combined_data_df)
            self._internal_logger.info(f"[HeatmapGenerator:_load_and_prepare] Data loaded and preprocessed: {total_timesteps_concatenated} timesteps from {num_episodes_loaded} episodes across {num_files_processed} files.")
            self._internal_logger.debug(f"[HeatmapGenerator:_load_and_prepare] Columns in combined DataFrame: {combined_data_df.columns.tolist()}")
            return combined_data_df

        except OSError as e_os_list_dir_hm: # 'e_os_list_dir_hm'
            self._internal_logger.error(f"[HeatmapGenerator:_load_and_prepare] Filesystem error listing/accessing {output_data_root_path}: {e_os_list_dir_hm}")
            return None
        except Exception as e_main_load_hm: # 'e_main_load_hm'
            self._internal_logger.error(f"[HeatmapGenerator:_load_and_prepare] Unexpected error during detailed data load/preparation: {e_main_load_hm}", exc_info=True)
            return None

    def _calculate_heatmap_grid_data(self, source_dataframe: pd.DataFrame, heatmap_plot_config: Dict) -> Optional[pd.DataFrame]: # 'source_dataframe', 'heatmap_plot_config'
        """Calcula la cuadrícula de valores agregados para un heatmap, aplicando filtrado y binning."""
        plot_name_ref = heatmap_plot_config.get('name', f"plot_idx_{heatmap_plot_config.get('_internal_plot_index','?')}") # 'plot_name_ref'
        
        try:
            # Validar configuración requerida para el heatmap
            required_heatmap_keys = ['x_variable', 'y_variable', 'value_variable'] # 'required_heatmap_keys'
            if not all(key_h_cfg in heatmap_plot_config for key_h_cfg in required_heatmap_keys): # 'key_h_cfg'
                missing_heatmap_keys = [key_h_miss for key_h_miss in required_heatmap_keys if key_h_miss not in heatmap_plot_config] # 'missing_heatmap_keys', 'key_h_miss'
                self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Heatmap config (plot='{plot_name_ref}') is missing required keys: {missing_heatmap_keys}.")
                raise ValueError(f"Heatmap configuration incomplete: missing {missing_heatmap_keys}")

            x_axis_var_name, y_axis_var_name, value_var_name = heatmap_plot_config['x_variable'], heatmap_plot_config['y_variable'], heatmap_plot_config['value_variable'] # 'x_axis_var_name', 'y_axis_var_name', 'value_var_name'
            
            # Leer 'aggregation' y 'bins' de la sub-config 'config' (styling)
            heatmap_style_config_dict = heatmap_plot_config.get('config', {}) # 'heatmap_style_config_dict'
            aggregation_method_name = heatmap_style_config_dict.get('aggregation', 'count').lower() # 'aggregation_method_name'
            num_bins_for_axes_cfg = heatmap_style_config_dict.get('bins', 50) # 'num_bins_for_axes_cfg'
            
            try: # Validar num_bins
                 num_bins_int = int(num_bins_for_axes_cfg) # 'num_bins_int'
                 if num_bins_int <= 0:
                      self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): 'bins' must be > 0, received {num_bins_for_axes_cfg}. Using default 50.")
                      num_bins_int = 50
            except (ValueError, TypeError):
                 self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): 'bins' value ({num_bins_for_axes_cfg}) is invalid. Using default 50.")
                 num_bins_int = 50
            
            # Leer filtro de termination_reason
            filter_by_termination_reasons_list = heatmap_style_config_dict.get('filter_termination_reason') # 'filter_by_termination_reasons_list'
            # Leer límites de ejes para filtrado y binning
            x_axis_min_limit, x_axis_max_limit = heatmap_style_config_dict.get('xmin'), heatmap_style_config_dict.get('xmax') # 'x_axis_min_limit', 'x_axis_max_limit'
            y_axis_min_limit, y_axis_max_limit = heatmap_style_config_dict.get('ymin'), heatmap_style_config_dict.get('ymax') # 'y_axis_min_limit', 'y_axis_max_limit'
            self._internal_logger.debug(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Axis limits from config: x=[{x_axis_min_limit}, {x_axis_max_limit}], y=[{y_axis_min_limit}, {y_axis_max_limit}]")

            # Validar columnas necesarias en el DataFrame fuente
            required_df_columns_list = list(set([x_axis_var_name, y_axis_var_name, value_var_name])) # 'required_df_columns_list'
            if filter_by_termination_reasons_list and isinstance(filter_by_termination_reasons_list, list):
                required_df_columns_list.append('termination_reason')

            missing_df_columns = [col_df_name for col_df_name in required_df_columns_list if col_df_name not in source_dataframe.columns] # 'missing_df_columns', 'col_df_name'
            if missing_df_columns:
                if 'termination_reason' in missing_df_columns and filter_by_termination_reasons_list:
                    self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): filter_termination_reason specified, but 'termination_reason' column is missing. Filter will not be applied.")
                    filter_by_termination_reasons_list = None # Desactivar filtro si la columna no existe
                
                essential_cols_still_missing = [col_ess_name for col_ess_name in [x_axis_var_name, y_axis_var_name, value_var_name] if col_ess_name not in source_dataframe.columns] # 'essential_cols_still_missing', 'col_ess_name'
                if essential_cols_still_missing:
                    self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Missing essential columns {essential_cols_still_missing}. Available columns: {source_dataframe.columns.tolist()}")
                    raise ValueError(f"Essential columns for heatmap not found: {essential_cols_still_missing}")

            # Filtrar por termination_reason ANTES de procesar numéricamente
            df_filtered_by_reason = source_dataframe.copy() # 'df_filtered_by_reason'
            if filter_by_termination_reasons_list and isinstance(filter_by_termination_reasons_list, list) and 'termination_reason' in df_filtered_by_reason.columns:
                rows_count_before_reason_filter = len(df_filtered_by_reason) # 'rows_count_before_reason_filter'
                df_filtered_by_reason = df_filtered_by_reason[df_filtered_by_reason['termination_reason'].isin(filter_by_termination_reasons_list)]
                rows_count_after_reason_filter = len(df_filtered_by_reason) # 'rows_count_after_reason_filter'
                self._internal_logger.info(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Filtered by termination_reason ({filter_by_termination_reasons_list}). Data points: {rows_count_after_reason_filter}/{rows_count_before_reason_filter}.")
                if df_filtered_by_reason.empty:
                    self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): No data remains after filtering by termination_reason. Heatmap cannot be generated.")
                    return None
            elif filter_by_termination_reasons_list and 'termination_reason' not in df_filtered_by_reason.columns:
                 self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): filter_termination_reason specified, but 'termination_reason' column not present. Filter not applied.")

            # Limpiar datos numéricos (solo columnas x, y, value)
            numeric_df_for_heatmap = pd.DataFrame(index=df_filtered_by_reason.index) # 'numeric_df_for_heatmap'
            cols_for_numeric_conversion = [x_axis_var_name, y_axis_var_name, value_var_name] # 'cols_for_numeric_conversion'
            is_conversion_successful = True # 'is_conversion_successful'
            for col_to_convert_name in cols_for_numeric_conversion: # 'col_to_convert_name'
                if col_to_convert_name in df_filtered_by_reason.columns:
                    numeric_df_for_heatmap[col_to_convert_name] = pd.to_numeric(df_filtered_by_reason[col_to_convert_name], errors='coerce')
                else: # No debería ocurrir si la validación de columnas fue correcta
                    self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Column '{col_to_convert_name}' unexpectedly absent before numeric conversion.")
                    is_conversion_successful = False; break
            if not is_conversion_successful: return None

            rows_count_before_nan_drop = numeric_df_for_heatmap.shape[0] # 'rows_count_before_nan_drop'
            numeric_df_for_heatmap.dropna(subset=[x_axis_var_name, y_axis_var_name, value_var_name], inplace=True)
            numeric_df_for_heatmap = numeric_df_for_heatmap[np.isfinite(numeric_df_for_heatmap[[x_axis_var_name, y_axis_var_name, value_var_name]]).all(axis=1)]
            rows_count_after_nan_drop = numeric_df_for_heatmap.shape[0] # 'rows_count_after_nan_drop'

            if numeric_df_for_heatmap.empty:
                self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): No valid numeric data remains after NaN/inf cleanup. Heatmap cannot be generated.")
                return None
            self._internal_logger.debug(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Valid numeric data points for binning/aggregation: {rows_count_after_nan_drop}/{rows_count_before_nan_drop}.")

            # Filtrar Datos por Límites de Ejes ANTES del Binning
            df_ready_for_binning = numeric_df_for_heatmap.copy() # 'df_ready_for_binning'
            rows_count_before_axis_limits_filter = len(df_ready_for_binning) # 'rows_count_before_axis_limits_filter'
            applied_axis_filters_log: List[str] = [] # 'applied_axis_filters_log'
            if x_axis_min_limit is not None: df_ready_for_binning = df_ready_for_binning[df_ready_for_binning[x_axis_var_name] >= x_axis_min_limit]; applied_axis_filters_log.append(f"x>={x_axis_min_limit}")
            if x_axis_max_limit is not None: df_ready_for_binning = df_ready_for_binning[df_ready_for_binning[x_axis_var_name] <= x_axis_max_limit]; applied_axis_filters_log.append(f"x<={x_axis_max_limit}")
            if y_axis_min_limit is not None: df_ready_for_binning = df_ready_for_binning[df_ready_for_binning[y_axis_var_name] >= y_axis_min_limit]; applied_axis_filters_log.append(f"y>={y_axis_min_limit}")
            if y_axis_max_limit is not None: df_ready_for_binning = df_ready_for_binning[df_ready_for_binning[y_axis_var_name] <= y_axis_max_limit]; applied_axis_filters_log.append(f"y<={y_axis_max_limit}")
            rows_count_after_axis_limits_filter = len(df_ready_for_binning) # 'rows_count_after_axis_limits_filter'
            
            if df_ready_for_binning.empty:
                self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): No valid data after filtering by axis limits ({', '.join(applied_axis_filters_log)}). Heatmap cannot be generated.")
                return None
            if applied_axis_filters_log:
                self._internal_logger.debug(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Data filtered by axis limits ({', '.join(applied_axis_filters_log)}) -> {rows_count_after_axis_limits_filter}/{rows_count_before_axis_limits_filter} points for binning.")

            x_axis_data_values = df_ready_for_binning[x_axis_var_name].values # 'x_axis_data_values'
            y_axis_data_values = df_ready_for_binning[y_axis_var_name].values # 'y_axis_data_values'

            # --- Binning ---
            x_axis_bin_edges_np, y_axis_bin_edges_np = None, None # 'x_axis_bin_edges_np', 'y_axis_bin_edges_np'
            try:
                # Determinar los rangos para el binning (usar límites de config si provistos, sino de los datos)
                x_range_for_bin_min = x_axis_min_limit if x_axis_min_limit is not None else np.min(x_axis_data_values) # 'x_range_for_bin_min'
                x_range_for_bin_max = x_axis_max_limit if x_axis_max_limit is not None else np.max(x_axis_data_values) # 'x_range_for_bin_max'
                y_range_for_bin_min = y_axis_min_limit if y_axis_min_limit is not None else np.min(y_axis_data_values) # 'y_range_for_bin_min'
                y_range_for_bin_max = y_axis_max_limit if y_axis_max_limit is not None else np.max(y_axis_data_values) # 'y_range_for_bin_max'

                # Validar que los rangos sean válidos (max > min)
                range_tolerance = 1e-9 # 'range_tolerance'
                if (x_range_for_bin_max - x_range_for_bin_min) < range_tolerance:
                    self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): X-axis range is null or negative ({x_range_for_bin_min} to {x_range_for_bin_max}) after applying limits/filtering. Cannot generate heatmap.")
                    return None
                if (y_range_for_bin_max - y_range_for_bin_min) < range_tolerance:
                    self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Y-axis range is null or negative ({y_range_for_bin_min} to {y_range_for_bin_max}) after applying limits/filtering. Cannot generate heatmap.")
                    return None

                # Crear los bordes de los bins usando linspace
                x_axis_bin_edges_np = np.linspace(x_range_for_bin_min, x_range_for_bin_max, num_bins_int + 1)
                y_axis_bin_edges_np = np.linspace(y_range_for_bin_min, y_range_for_bin_max, num_bins_int + 1)
                self._internal_logger.debug(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): X-axis bin edges ({len(x_axis_bin_edges_np)}): [{x_axis_bin_edges_np[0]:.3f} ... {x_axis_bin_edges_np[-1]:.3f}]")
                self._internal_logger.debug(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Y-axis bin edges ({len(y_axis_bin_edges_np)}): [{y_axis_bin_edges_np[0]:.3f} ... {y_axis_bin_edges_np[-1]:.3f}]")
                
                # Asignar datos a los bins
                # Usar pd.cut para obtener los índices de los bins. labels=False devuelve índices enteros.
                # include_lowest=True para incluir el valor mínimo. right=True significa que el bin es (a, b].
                x_bin_indices_series = pd.cut(x_axis_data_values, bins=x_axis_bin_edges_np, labels=False, include_lowest=True, right=True, retbins=False) # 'x_bin_indices_series'
                y_bin_indices_series = pd.cut(y_axis_data_values, bins=y_axis_bin_edges_np, labels=False, include_lowest=True, right=True, retbins=False) # 'y_bin_indices_series'
                
                # Añadir los índices de bin al DataFrame filtrado (df_ready_for_binning)
                # Es importante usar .loc con el índice original para asignar correctamente
                numeric_df_for_heatmap.loc[df_ready_for_binning.index, f'{x_axis_var_name}_bin_idx'] = x_bin_indices_series # '_bin_idx'
                numeric_df_for_heatmap.loc[df_ready_for_binning.index, f'{y_axis_var_name}_bin_idx'] = y_bin_indices_series

                # Eliminar filas donde el binning pudo haber fallado (raro con include_lowest=True)
                if numeric_df_for_heatmap[[f'{x_axis_var_name}_bin_idx', f'{y_axis_var_name}_bin_idx']].isnull().any().any():
                    self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): NaNs found in bin indices. Removing affected rows.")
                    numeric_df_for_heatmap.dropna(subset=[f'{x_axis_var_name}_bin_idx', f'{y_axis_var_name}_bin_idx'], inplace=True)
                    if numeric_df_for_heatmap.empty:
                        self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): No data remains after removing NaN bin indices.")
                        return None
                # Convertir índices de bin a enteros (pd.cut puede devolver float si hay NaNs)
                numeric_df_for_heatmap[f'{x_axis_var_name}_bin_idx'] = numeric_df_for_heatmap[f'{x_axis_var_name}_bin_idx'].astype(int)
                numeric_df_for_heatmap[f'{y_axis_var_name}_bin_idx'] = numeric_df_for_heatmap[f'{y_axis_var_name}_bin_idx'].astype(int)

            except ValueError as e_pd_cut_binning: # 'e_pd_cut_binning'
                self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Error during pd.cut binning for heatmap '{plot_name_ref}': {e_pd_cut_binning}. Ensure 'bins' ({num_bins_int}) is valid for data range.")
                return None
            except Exception as e_generic_binning: # 'e_generic_binning'
                self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Unexpected error during binning for heatmap '{plot_name_ref}': {e_generic_binning}", exc_info=True)
                return None

            # --- Pivot Table para Agregación ---
            aggregation_func_map = { # 'aggregation_func_map'
                'mean': 'mean', 'median': 'median', 'std': 'std', 'var': 'var',
                'sum': 'sum', 'count': 'size', # 'size' cuenta ocurrencias en el grupo (no necesita 'value_var_name')
                'min': 'min', 'max': 'max',
            }
            pandas_agg_func_str = aggregation_func_map.get(aggregation_method_name) # 'pandas_agg_func_str'
            if not pandas_agg_func_str:
                self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Aggregation method '{aggregation_method_name}' not recognized. Using 'count'.")
                pandas_agg_func_str = 'size'

            # Si la agregación es 'count' (size), la columna 'values' es irrelevante para pivot_table,
            # pero una debe ser especificada. Usamos x_axis_var_name como placeholder.
            value_col_for_pivot = x_axis_var_name if pandas_agg_func_str == 'size' else value_var_name # 'value_col_for_pivot'
 
            self._internal_logger.debug(f"[HeatmapGenerator:_calculate_grid] Creating pivot_table for '{plot_name_ref}' (X_bin_idx:{x_axis_var_name}_bin_idx, Y_bin_idx:{y_axis_var_name}_bin_idx, ValueCol:{value_col_for_pivot}, AggFunc:{pandas_agg_func_str})")
            
            final_heatmap_grid_df: Optional[pd.DataFrame] = None # 'final_heatmap_grid_df'
            try:
                final_heatmap_grid_df = pd.pivot_table(
                    numeric_df_for_heatmap, # Usar el DF que tiene las columnas _bin_idx
                    values=value_col_for_pivot,
                    index=f'{y_axis_var_name}_bin_idx', # Filas = bins de Y
                    columns=f'{x_axis_var_name}_bin_idx', # Columnas = bins de X
                    aggfunc=pandas_agg_func_str,
                    fill_value=np.nan # Rellenar celdas sin datos (combinaciones de bins no presentes) con NaN
                )
                # Asegurar que los índices/columnas (que son bin índices 0, 1, ...) sean enteros
                final_heatmap_grid_df.index = final_heatmap_grid_df.index.astype(int)
                final_heatmap_grid_df.columns = final_heatmap_grid_df.columns.astype(int)

            except Exception as e_pivot_table_hm: # 'e_pivot_table_hm'
                self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Error during pd.pivot_table for heatmap '{plot_name_ref}': {e_pivot_table_hm}", exc_info=True)
                return None

            # --- Reindexar y Asignar Ejes con Centros de Bins ---
            # Crear índices completos para X e Y (0 a num_bins_int-1)
            full_x_axis_bin_index = pd.RangeIndex(start=0, stop=num_bins_int, name=f'{x_axis_var_name}_bin_idx') # 'full_x_axis_bin_index'
            full_y_axis_bin_index = pd.RangeIndex(start=0, stop=num_bins_int, name=f'{y_axis_var_name}_bin_idx') # 'full_y_axis_bin_index'

            # Reindexar la tabla para asegurar que tenga todas las celdas (rellenando con NaN si faltan)
            final_heatmap_grid_df = final_heatmap_grid_df.reindex(index=full_y_axis_bin_index, columns=full_x_axis_bin_index, fill_value=np.nan)

            # Calcular centros de los bins para usarlos como etiquetas de ejes
            if x_axis_bin_edges_np is None or y_axis_bin_edges_np is None: # Deberían estar definidos
                self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Bin edges are not defined after binning. Cannot calculate bin centers for axes.")
                return None # No se pueden crear ejes significativos
            try:
                # Centros = promedio de los bordes de cada bin
                x_axis_bin_centers = (x_axis_bin_edges_np[:-1] + x_axis_bin_edges_np[1:]) / 2 # 'x_axis_bin_centers'
                y_axis_bin_centers = (y_axis_bin_edges_np[:-1] + y_axis_bin_edges_np[1:]) / 2 # 'y_axis_bin_centers'

                if len(x_axis_bin_centers) != num_bins_int or len(y_axis_bin_centers) != num_bins_int:
                    self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Discrepancy between num_bins ({num_bins_int}) and number of calculated bin centers (X:{len(x_axis_bin_centers)}, Y:{len(y_axis_bin_centers)}). Using numeric bin indices for axes.")
                    # Fallback: nombrar los ejes con los nombres de las variables originales, pero los índices seguirán siendo 0,1,...
                    final_heatmap_grid_df.index.name = y_axis_var_name
                    final_heatmap_grid_df.columns.name = x_axis_var_name
                else:
                    # Reemplazar los índices numéricos (0..N-1) con los valores de los centros de los bins
                    # Mantener el nombre del eje original (y_axis_var_name, x_axis_var_name)
                    final_heatmap_grid_df.index = pd.Index(y_axis_bin_centers, name=y_axis_var_name)
                    final_heatmap_grid_df.columns = pd.Index(x_axis_bin_centers, name=x_axis_var_name)

            except Exception as e_axes_relabel: # 'e_axes_relabel'
                self._internal_logger.warning(f"[HeatmapGenerator:_calculate_grid] Heatmap ('{plot_name_ref}'): Error assigning bin centers as axes labels: {e_axes_relabel}. Numeric bin indices will be used.", exc_info=True)
                # Nombrar ejes aunque se usen índices numéricos
                final_heatmap_grid_df.index.name = y_axis_var_name
                final_heatmap_grid_df.columns.name = x_axis_var_name

            self._internal_logger.info(f"[HeatmapGenerator:_calculate_grid] Heatmap grid ('{plot_name_ref}') calculated ({final_heatmap_grid_df.shape[0]}x{final_heatmap_grid_df.shape[1]}) using aggregation '{pandas_agg_func_str}'.")
            return final_heatmap_grid_df

        except ValueError as ve_heatmap_cfg: # 've_heatmap_cfg' (Errores de validación de config)
            self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Configuration or data error for heatmap '{plot_name_ref}': {ve_heatmap_cfg}")
            return None
        except Exception as e_calc_grid_unexp: # 'e_calc_grid_unexp'
            self._internal_logger.error(f"[HeatmapGenerator:_calculate_grid] Unexpected error calculating heatmap grid for '{plot_name_ref}': {e_calc_grid_unexp}", exc_info=True)
            return None

    def generate(self, output_root_path_gen: str, heatmap_configs_list: List[Dict], output_excel_target_filepath: str): # 'output_root_path_gen', 'heatmap_configs_list', 'output_excel_target_filepath'
        """
        Genera un archivo Excel con hojas para cada heatmap configurado.
        Carga los datos detallados una sola vez.
        """
        all_detailed_data_df = self._load_and_prepare_detailed_data(output_root_path_gen) # 'all_detailed_data_df'
        if all_detailed_data_df is None:
            self._internal_logger.error("[HeatmapGenerator:generate] Failed to load detailed data. Cannot generate heatmap data sheets.")
            return

        excel_writer_obj = None # 'excel_writer_obj'
        num_sheets_generated = 0 # 'num_sheets_generated'
        excel_engine_to_use = None # 'excel_engine_to_use'
        try:
            import openpyxl # Intentar importar para usar el engine explícitamente
            excel_engine_to_use = 'openpyxl'
            self._internal_logger.info("[HeatmapGenerator:generate] Using 'openpyxl' engine for writing Excel file.")
        except ImportError:
            self._internal_logger.warning("[HeatmapGenerator:generate] 'openpyxl' library not found. Pandas will use its default engine (may be limited or require other dependencies like xlsxwriter). Consider installing: pip install openpyxl")
            # Pandas usará xlsxwriter o similar si está disponible, o podría fallar si no hay ninguno.

        excel_sheets_to_write_map: Dict[str, pd.DataFrame] = {} # 'excel_sheets_to_write_map'
        processed_excel_sheet_names_set: set = set() # 'processed_excel_sheet_names_set'

        for current_heatmap_config in heatmap_configs_list: # 'current_heatmap_config'
            if not isinstance(current_heatmap_config, dict) or current_heatmap_config.get("type") != "heatmap":
                continue # Saltar si no es una config de heatmap válida

            plot_internal_idx = current_heatmap_config.get('_internal_plot_index', '?') # 'plot_internal_idx'
            config_plot_name = current_heatmap_config.get('name') # 'config_plot_name'

            # Determinar nombre de la hoja Excel
            target_sheet_name: Optional[str] = None # 'target_sheet_name'
            if config_plot_name:
                target_sheet_name = str(config_plot_name)[:31] # Limitar longitud a 31 chars para Excel
            else:
                target_sheet_name = f"heatmap_idx_{plot_internal_idx}"[:31]
                self._internal_logger.warning(f"[HeatmapGenerator:generate] Heatmap config (index={plot_internal_idx}) has no 'name' attribute. Using default sheet name: '{target_sheet_name}'. It's recommended to provide a unique 'name' in the YAML plot configuration.")

            # Manejar nombres de hoja duplicados
            original_target_sheet_name = target_sheet_name # 'original_target_sheet_name'
            duplication_counter = 1 # 'duplication_counter'
            while target_sheet_name in processed_excel_sheet_names_set:
                name_suffix = f"_{duplication_counter}" # 'name_suffix'
                max_base_name_len = 31 - len(name_suffix) # 'max_base_name_len'
                target_sheet_name = original_target_sheet_name[:max_base_name_len] + name_suffix
                duplication_counter += 1
                if duplication_counter > 10: # Evitar bucle infinito
                    self._internal_logger.error(f"[HeatmapGenerator:generate] Too many sheet name collisions for base '{original_target_sheet_name}'. Skipping this heatmap.")
                    target_sheet_name = None; break 
            if target_sheet_name is None: continue

            processed_excel_sheet_names_set.add(target_sheet_name)
            log_plot_ref_name = f"name='{config_plot_name}'" if config_plot_name else f"index={plot_internal_idx}" # 'log_plot_ref_name'
            self._internal_logger.info(f"[HeatmapGenerator:generate] Processing heatmap config ({log_plot_ref_name}) -> Target Excel Sheet: '{target_sheet_name}'")

            # Calcular la cuadrícula de datos para el heatmap
            heatmap_grid_dataframe: Optional[pd.DataFrame] = None # 'heatmap_grid_dataframe'
            try:
                # Pasar una copia del DataFrame de todos los datos para evitar modificaciones accidentales
                heatmap_grid_dataframe = self._calculate_heatmap_grid_data(all_detailed_data_df.copy(), current_heatmap_config)
            except Exception as e_calc_heatmap_grid: # 'e_calc_heatmap_grid'
                self._internal_logger.error(f"[HeatmapGenerator:generate] Unrecoverable error calculating grid for sheet '{target_sheet_name}'. Error: {e_calc_heatmap_grid}", exc_info=True)

            if heatmap_grid_dataframe is not None and not heatmap_grid_dataframe.empty:
                excel_sheets_to_write_map[target_sheet_name] = heatmap_grid_dataframe
                num_sheets_generated += 1
                self._internal_logger.info(f"[HeatmapGenerator:generate] Data for sheet '{target_sheet_name}' generated successfully ({heatmap_grid_dataframe.shape[0]}x{heatmap_grid_dataframe.shape[1]}).")
            else:
                self._internal_logger.warning(f"[HeatmapGenerator:generate] No valid data generated for sheet '{target_sheet_name}'. This sheet will not be included in the Excel file.")

        # Escribir todas las hojas válidas al archivo Excel
        if num_sheets_generated > 0:
            self._internal_logger.info(f"[HeatmapGenerator:generate] Writing {num_sheets_generated} heatmap data sheets to Excel file -> {output_excel_target_filepath}")
            try:
                excel_writer_obj = pd.ExcelWriter(output_excel_target_filepath, engine=excel_engine_to_use)
                for sheet_name_final_write, df_for_sheet_final_write in excel_sheets_to_write_map.items(): # 'sheet_name_final_write', 'df_for_sheet_final_write'
                    # Guardar CON índice y cabecera (representan los centros de bins o los nombres de variables)
                    df_for_sheet_final_write.to_excel(excel_writer_obj, sheet_name=sheet_name_final_write, index=True, header=True)
                excel_writer_obj.close() # Cierra y guarda el archivo
                self._internal_logger.info("[HeatmapGenerator:generate] Excel file with heatmap data saved successfully.")
            except Exception as e_write_excel_hm: # 'e_write_excel_hm'
                self._internal_logger.error(f"[HeatmapGenerator:generate] Error writing Excel file '{output_excel_target_filepath}': {e_write_excel_hm}", exc_info=True)
                if excel_writer_obj is not None and hasattr(excel_writer_obj, 'close') and not getattr(excel_writer_obj, '_closed', True):
                    try: excel_writer_obj.close()
                    except Exception as e_close_on_error: self._internal_logger.error(f"[HeatmapGenerator:generate] Error closing ExcelWriter after a write error: {e_close_on_error}") # 'e_close_on_error'
        else:
            self._internal_logger.warning("[HeatmapGenerator:generate] No valid heatmap data sheets were generated. Excel file will not be created.")