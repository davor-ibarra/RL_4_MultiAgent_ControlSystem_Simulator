import logging
import pandas as pd
import numpy as np
import os
import json
import gc
import traceback
from typing import List, Dict, Optional, Tuple, Any

# Intentar importar matplotlib y manejar error
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker # For MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Definir placeholders si no está disponible para evitar errores posteriores
    plt = None # type: ignore
    mcolors = None # type: ignore
    mticker = None # type: ignore


# Reutilizar funciones helper (adaptadas)
from utils.numpy_encoder import NumpyEncoder # Podría ser útil si cargamos json con numpy types

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)


# --- Helper Functions (Adaptadas de visualization.py antiguo y heatmap_generator.py) ---

def _find_latest_simulation_data(results_folder: str) -> Optional[str]:
    """Encuentra el archivo simulation_data JSON con el número de episodio más alto."""
    # (Misma lógica que en heatmap_generator)
    latest_file: Optional[str] = None; highest_ep: int = -1
    if not os.path.isdir(results_folder): logger.error(f"Carpeta resultados no existe: {results_folder}"); return None
    try:
        for filename in os.listdir(results_folder):
            if filename.startswith('simulation_data_') and filename.endswith('.json'):
                parts = filename[:-5].split('_'); # Remove .json and split by _
                if len(parts) >= 4 and parts[-2] == 'to':
                    try: end_ep = int(parts[-1])
                    except ValueError: continue
                    if end_ep > highest_ep: highest_ep = end_ep; latest_file = os.path.join(results_folder, filename)
    except FileNotFoundError: logger.error(f"Error: Carpeta no encontrada durante búsqueda: {results_folder}"); return None
    except Exception as e: logger.error(f"Error buscando último archivo datos: {e}"); return None
    if latest_file: logger.info(f"Último archivo datos encontrado para plots detallados: {os.path.basename(latest_file)}")
    else: logger.warning(f"No se encontraron archivos simulation_data_*_to_*.json en {results_folder}")
    return latest_file

def _extract_heatmap_data(detailed_data: List[Dict], x_var: str, y_var: str, filter_reasons: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae y concatena datos X, Y para heatmaps desde lista detallada, con filtro opcional."""
    # (Misma lógica que en heatmap_generator, pero usa logger de este módulo)
    x_all, y_all = [], []
    if not isinstance(detailed_data, list): logger.error("Formato inválido detailed_data: no es lista."); return np.array([]), np.array([])
    filtered_episodes = detailed_data
    if filter_reasons:
        if not isinstance(filter_reasons, list) or not all(isinstance(r, str) for r in filter_reasons): logger.warning(f"filter_termination_reason ({filter_reasons}) inválido. Ignorando filtro."); filter_reasons = None
        else:
            try:
                 original_count = len(filtered_episodes); filtered_episodes = [ ep for ep in filtered_episodes if isinstance(ep, dict) and get_last_or_value(ep, 'termination_reason', '') in filter_reasons ]; filtered_count = len(filtered_episodes);
                 logger.info( f"Filtro Heatmap por {filter_reasons}: {filtered_count}/{original_count} episodios mantenidos.")
            except Exception as e: logger.error(f"Error filtro episodios heatmap: {e}", exc_info=True); filtered_episodes = detailed_data; filter_reasons = None
    if not filtered_episodes: logger.warning(f"No quedan episodios tras filtro heatmap {filter_reasons}."); return np.array([]), np.array([])
    valid_points_count = 0
    for i, ep_data in enumerate(filtered_episodes):
        if not isinstance(ep_data, dict): continue
        x_raw = ep_data.get(x_var); y_raw = ep_data.get(y_var)
        if x_raw is None or y_raw is None or not isinstance(x_raw, list) or not isinstance(y_raw, list): continue
        try:
            min_len = min(len(x_raw), len(y_raw)); x_num = pd.to_numeric(x_raw[:min_len], errors='coerce'); y_num = pd.to_numeric(y_raw[:min_len], errors='coerce'); mask = np.isfinite(x_num) & np.isfinite(y_num); num_valid_in_ep = np.sum(mask)
            if num_valid_in_ep > 0: x_all.append(x_num[mask]); y_all.append(y_num[mask]); valid_points_count += num_valid_in_ep
        except Exception as e: logger.warning(f"Error procesando datos numéricos heatmap ep {get_last_or_value(ep_data,'episode',i)} ({x_var}, {y_var}): {e}"); continue
    if not x_all or not y_all: logger.warning(f"No se encontraron puntos válidos para heatmap '{y_var}' vs '{x_var}'{' filtro ' + str(filter_reasons) if filter_reasons else ''}."); return np.array([]), np.array([])
    try:
        x_combined = np.concatenate(x_all); y_combined = np.concatenate(y_all);
        logger.info( f"Datos extraídos heatmap '{y_var}' vs '{x_var}': {len(x_combined)} pts."); return x_combined, y_combined
    except Exception as e: logger.error(f"Error concatenando datos heatmap ({x_var}, {y_var}): {e}", exc_info=True); return np.array([]), np.array([])

def _apply_common_mpl_styles(ax, fig, plot_cfg):
    """Aplica estilos matplotlib comunes desde config (adaptado de visualization.py antiguo)."""
    if not MATPLOTLIB_AVAILABLE: return
    cfg = plot_cfg.get('config', {})
    # Titles/Labels
    ax.set_title(cfg.get('title', ''), fontsize=cfg.get('title_fontsize', 14))
    ax.set_xlabel(cfg.get('xlabel', ''), fontsize=cfg.get('xlabel_fontsize', 12))
    ax.set_ylabel(cfg.get('ylabel', ''), fontsize=cfg.get('ylabel_fontsize', 12))
    # Ticks
    xtick_rotation = cfg.get('xtick_rotation', 0)
    tick_fs = cfg.get('tick_fontsize', 10)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fs * 0.8)
    # Grid
    ax.grid(visible=cfg.get('grid_on', False), linestyle='--', alpha=0.6)
    # Limits
    xmin, xmax = cfg.get('xmin'), cfg.get('xmax')
    ymin, ymax = cfg.get('ymin'), cfg.get('ymax')
    if xmin is not None or xmax is not None: ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None: ax.set_ylim(bottom=ymin, top=ymax)
    # Tick Locator
    num_xticks = cfg.get('num_xticks'); num_yticks = cfg.get('num_yticks')
    if num_xticks is not None and num_xticks > 0:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=num_xticks, prune='both', integer=ax.get_xlabel().lower()=='episode [-]')) # Integer for episodes
    if num_yticks is not None and num_yticks > 0:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=num_yticks, prune='both'))
    # Rotation
    if xtick_rotation != 0: plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right", rotation_mode="anchor")
    else: plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    # Legend
    legend_pos = cfg.get('legend_pos', 'best')
    legend_fs = cfg.get('legend_fontsize', 'small')
    if cfg.get('show_legend', False) and ax.get_legend_handles_labels()[0]: # Check if there are handles to show
        legend_title = cfg.get('legend_title', None)
        if legend_pos == 'outside':
            ax.legend(title=legend_title, bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=legend_fs)
        else:
            ax.legend(title=legend_title, loc=legend_pos, fontsize=legend_fs)
    elif not cfg.get('show_legend', False) and ax.get_legend() is not None:
         ax.get_legend().remove()
    # Tight Layout
    # rect_right = 0.85 if legend_pos == 'outside' and cfg.get('show_legend', False) else 1.0
    # try: fig.tight_layout(rect=[0, 0, rect_right, 1])
    # except Exception: pass # Ignore tight_layout errors silently

def get_last_or_value(data_dict: Dict[str, List[Any]], key: str, default: Any = np.nan) -> Any:
    """Helper interno para PlotGenerator si necesita obtener valores finales de datos detallados."""
    value = data_dict.get(key)
    if isinstance(value, list): return value[-1] if value else default
    elif value is not None: return value
    else: return default

# --- Clase PlotGenerator ---

class PlotGenerator:
    """
    Servicio para generar gráficos basados en la configuración y los datos de simulación.
    """
    def __init__(self, logger_injected: logging.Logger):
        self.logger = logger_injected
        if not isinstance(self.logger, logging.Logger): self.logger = logging.getLogger(__name__)
        self.logger.info(f"PlotGenerator instance created. Matplotlib available: {MATPLOTLIB_AVAILABLE}")
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib no encontrado. La generación de gráficos fallará. Instalar: pip install matplotlib")
        # Configuración global de estilo si se desea
        # if MATPLOTLIB_AVAILABLE: plt.style.use('seaborn-v0_8-darkgrid')

    def generate(self, plot_configs: List[Dict], summary_df: pd.DataFrame, results_folder: str):
        """Genera y guarda las visualizaciones según la configuración."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib no disponible. Saltando generación de gráficos.")
            return
        if not os.path.isdir(results_folder):
            self.logger.error(f"Carpeta de resultados no encontrada: {results_folder}. No se pueden guardar gráficos.")
            return

        self.logger.info(f"Iniciando generación de hasta {len(plot_configs)} gráficos en: {results_folder}")
        detailed_data_cache: Optional[List[Dict]] = None # Cache para datos detallados

        for i, p_cfg in enumerate(plot_configs):
            if not isinstance(p_cfg, dict) or not p_cfg.get('enabled', True):
                continue # Saltar config inválida o deshabilitada

            plot_type = p_cfg.get('type')
            source = p_cfg.get('source')
            output_filename = p_cfg.get('output_filename', f"plot_{plot_type}_{i}.png")
            filepath = os.path.join(results_folder, output_filename)
            cfg_inner = p_cfg.get('config', {})
            filter_reasons = cfg_inner.get('filter_termination_reason')

            self.logger.info(f"--- Procesando Plot {i+1}/{len(plot_configs)}: Tipo='{plot_type}', Fuente='{source}', Salida='{output_filename}' ---")

            data_to_use: Any = None
            try:
                # --- Cargar/Preparar Datos ---
                if source == 'summary':
                    if summary_df.empty:
                        self.logger.warning(f"Fuente 'summary' seleccionada pero summary_df está vacío. Saltando plot '{output_filename}'.")
                        continue
                    data_to_use = summary_df.copy() # Usar copia
                    # Filtrar DataFrame de resumen si aplica
                    if filter_reasons and 'termination_reason' in data_to_use.columns:
                        original_count = len(data_to_use)
                        data_to_use = data_to_use[data_to_use['termination_reason'].isin(filter_reasons)]
                        self.logger.info(f"Filtro Summary por {filter_reasons}: {len(data_to_use)}/{original_count} episodios mantenidos.")
                        if data_to_use.empty:
                             self.logger.warning("Summary DF vacío tras filtro. Saltando plot.")
                             continue

                elif source == 'detailed':
                    if detailed_data_cache is None: # Cargar solo una vez si es necesario
                        detail_json_path = _find_latest_simulation_data(results_folder)
                        if not detail_json_path:
                            self.logger.error("Fuente 'detailed' seleccionada pero no se encontró archivo simulation_data_*.json. Saltando plots detallados.")
                            break # No continuar con plots detallados si no hay datos
                        try:
                            with open(detail_json_path, 'r', encoding='utf-8') as f:
                                detailed_data_cache = json.load(f) # Cargar todo (puede ser grande)
                            if not isinstance(detailed_data_cache, list):
                                 self.logger.error("Archivo detallado no contiene una lista JSON."); detailed_data_cache = None; break
                            self.logger.info(f"Datos detallados cargados desde {os.path.basename(detail_json_path)} ({len(detailed_data_cache)} episodios)")
                        except Exception as load_e:
                            self.logger.error(f"Error cargando datos detallados desde {detail_json_path}: {load_e}", exc_info=True)
                            detailed_data_cache = None; break # No continuar si falla la carga
                    if detailed_data_cache is None: continue # Saltar si la carga falló o no había archivo
                    data_to_use = detailed_data_cache # Pasar la lista completa

                else:
                    self.logger.warning(f"Fuente de datos inválida ('{source}') para plot '{output_filename}'. Saltando.")
                    continue

                # --- Llamar a Función de Ploteo Específica ---
                fig, ax = plt.subplots(figsize=(cfg_inner.get('figsize_w', 12), cfg_inner.get('figsize_h', 6)))

                plot_successful = False
                if plot_type == 'line':
                    plot_successful = self._plot_line(ax, fig, data_to_use, p_cfg)
                elif plot_type == 'bar':
                     plot_successful = self._plot_bar(ax, fig, data_to_use, p_cfg) # data_to_use es summary_df aquí
                elif plot_type == 'heatmap':
                     plot_successful = self._plot_heatmap(ax, fig, data_to_use, p_cfg) # data_to_use es lista detallada aquí
                else:
                    self.logger.warning(f"Tipo de plot desconocido '{plot_type}'. Saltando.")

                # --- Guardar y Limpiar ---
                if plot_successful:
                    _apply_common_mpl_styles(ax, fig, p_cfg) # Aplicar estilos comunes ANTES de guardar
                    try:
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        self.logger.info(f"Gráfico guardado: {output_filename}")
                    except Exception as save_e:
                        self.logger.error(f"Fallo al guardar gráfico '{output_filename}': {save_e}")
                # Cerrar figura siempre para liberar memoria, incluso si el ploteo falló
                plt.close(fig)
                gc.collect()

            except Exception as e:
                self.logger.error(f"Error inesperado generando plot '{output_filename}': {e}\n{traceback.format_exc()}")
                if 'fig' in locals() and fig: plt.close(fig) # Asegurar cierre si hay error
                gc.collect()

        self.logger.info("--- Generación de Gráficos Finalizada ---")

    # --- Funciones de Ploteo Internas ---

    def _plot_line(self, ax, fig, data_df: pd.DataFrame, plot_cfg: dict) -> bool:
        """Genera un gráfico de línea."""
        cfg = plot_cfg.get('config', {})
        x_var = plot_cfg.get('x_variable')
        y_var = plot_cfg.get('y_variable')
        if not x_var or not y_var or x_var not in data_df.columns or y_var not in data_df.columns:
            self.logger.warning(f"Variables X='{x_var}' o Y='{y_var}' no válidas o no encontradas en DataFrame resumen. Saltando línea.")
            return False

        # Preparar datos
        df = data_df[[x_var, y_var]].copy()
        df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
        df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
        df = df.dropna().sort_values(by=x_var)
        if df.empty: self.logger.warning(f"No hay datos válidos para línea '{y_var}' vs '{x_var}'. Saltando."); return False

        # Plotear
        ax.plot(df[x_var], df[y_var],
                color=cfg.get('line_color', '#1f77b4'),
                linewidth=cfg.get('line_width', 1.5),
                linestyle=cfg.get('line_style', '-'),
                marker=cfg.get('marker_style', ''),
                markersize=cfg.get('marker_size', 0),
                markerfacecolor=cfg.get('marker_color', cfg.get('line_color', '#ff7f0e')), # Usar color línea si no hay color marcador
                markeredgecolor=cfg.get('marker_color', cfg.get('line_color', '#ff7f0e')))

        # Añadir títulos auto si no están definidos
        if not cfg.get('xlabel'): cfg['xlabel'] = x_var.replace('_', ' ').title() + ' [-]' if x_var=='episode' else x_var.replace('_', ' ').title()
        if not cfg.get('ylabel'): cfg['ylabel'] = y_var.replace('_', ' ').title()
        return True


    def _plot_bar(self, ax, fig, data_df: pd.DataFrame, plot_cfg: dict) -> bool:
        """Genera un gráfico de barras (simple o agrupado)."""
        cfg = plot_cfg.get('config', {})
        variable = plot_cfg.get('variable')
        group_size = cfg.get('group_size')
        if not variable or variable not in data_df.columns:
            self.logger.warning(f"Variable '{variable}' no válida o no encontrada para gráfico de barras. Saltando.")
            return False

        plot_data = None
        plot_kind = 'bar'
        is_stacked = False

        if group_size and 'episode' in data_df.columns:
            df = data_df[['episode', variable]].copy()
            df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
            df = df.dropna(subset=['episode'])
            if df.empty: self.logger.warning("No hay episodios válidos para agrupar barras."); return False
            df['episode_group'] = (df['episode'] // group_size) * group_size
            plot_data = df.groupby(['episode_group', variable]).size().unstack(fill_value=0)
            if not cfg.get('xlabel'): cfg['xlabel'] = 'Episode Group Start [-]'
            if not cfg.get('ylabel'): cfg['ylabel'] = 'Count [-]'
            is_stacked = True
            if not cfg.get('legend_title'): cfg['legend_title'] = variable.replace('_', ' ').title()
            if cfg.get('show_legend') is None: cfg['show_legend'] = True # Mostrar leyenda por defecto si es agrupado
        else:
            plot_data = data_df[variable].value_counts().sort_index()
            if not cfg.get('xlabel'): cfg['xlabel'] = variable.replace('_', ' ').title()
            if not cfg.get('ylabel'): cfg['ylabel'] = 'Frequency [-]'
            if cfg.get('show_legend') is None: cfg['show_legend'] = False # Ocultar leyenda por defecto si es simple

        if plot_data is None or plot_data.empty:
            self.logger.warning(f"No hay datos para gráfico de barras '{variable}'. Saltando.")
            return False

        # Plotear
        cmap_name = cfg.get('cmap', 'viridis' if is_stacked else 'tab10') # Usar colormaps diferentes
        bar_width = cfg.get('bar_width', 0.8)
        try:
            plot_data.plot(kind=plot_kind, stacked=is_stacked, ax=ax, colormap=cmap_name, width=bar_width)
        except Exception as plot_e:
             self.logger.error(f"Error durante ploteo de barras para '{variable}': {plot_e}"); return False
        return True


    def _plot_heatmap(self, ax, fig, detailed_data: List[Dict], plot_cfg: dict) -> bool:
        """Genera un heatmap."""
        cfg = plot_cfg.get('config', {})
        x_var = plot_cfg.get('x_variable')
        y_var = plot_cfg.get('y_variable')
        if not x_var or not y_var: self.logger.warning("Variables X o Y no definidas para heatmap. Saltando."); return False

        # Extraer datos (usa helper)
        filter_reasons = cfg.get('filter_termination_reason')
        x_data, y_data = _extract_heatmap_data(detailed_data, x_var, y_var, filter_reasons)
        if x_data.size == 0 or y_data.size == 0: self.logger.warning(f"No hay datos válidos para heatmap '{y_var}' vs '{x_var}'. Saltando."); return False

        # Preparar parámetros hist2d
        bins = cfg.get('bins', 50)
        cmap = cfg.get('cmap', 'hot')
        log_scale = cfg.get('log_scale', False)
        cmin, cmax = cfg.get('cmin'), cfg.get('cmax')
        xmin_cfg, xmax_cfg = cfg.get('xmin'), cfg.get('xmax'); ymin_cfg, ymax_cfg = cfg.get('ymin'), cfg.get('ymax')
        hist_range = None
        if all(v is not None for v in [xmin_cfg, xmax_cfg, ymin_cfg, ymax_cfg]): hist_range = [[xmin_cfg, xmax_cfg], [ymin_cfg, ymax_cfg]]

        # Configurar normalización (log o lineal)
        norm = None
        if log_scale:
            # Asegurar vmin > 0 para LogNorm si se especifica cmin
            effective_cmin = cmin if cmin is not None and cmin > 1e-9 else None # Evitar 0 o negativos
            norm = mcolors.LogNorm(vmin=effective_cmin, vmax=cmax)
            hist_cmin_arg = None # Dejar que LogNorm maneje el mínimo
        else:
            norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
            hist_cmin_arg = cmin # Pasar cmin a hist2d si es lineal

        # Plotear hist2d
        try:
            counts, xedges, yedges, img = ax.hist2d(
                x_data, y_data, bins=bins, cmap=cmap, norm=norm, range=hist_range, cmin=hist_cmin_arg
            )
        except Exception as hist_e:
            self.logger.error(f"Error durante ploteo hist2d para '{y_var}' vs '{x_var}': {hist_e}"); return False

        # Añadir colorbar
        cbar = fig.colorbar(img, ax=ax)
        cbar_label = cfg.get('clabel', 'Frequency') + (' (Log Scale)' if log_scale else '')
        cbar.set_label(cbar_label, fontsize=cfg.get('clabel_fontsize', 10))
        cbar.ax.tick_params(labelsize=cfg.get('tick_fontsize', 10))

        # Añadir títulos auto si no están definidos
        if not cfg.get('xlabel'): cfg['xlabel'] = x_var.replace('_', ' ').title()
        if not cfg.get('ylabel'): cfg['ylabel'] = y_var.replace('_', ' ').title()
        return True