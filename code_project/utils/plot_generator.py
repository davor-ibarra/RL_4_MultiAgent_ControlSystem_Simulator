import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker # Para MaxNLocator
import os
import numpy as np
import pandas as pd
import logging
import traceback
import gc # Garbage collector para liberar memoria de plots
import json # Para cargar datos detallados si es necesario
from typing import Optional, List, Dict

# Importar NumpyEncoder si se necesita para cargar/guardar datos internamente (poco probable)
# from utils.numpy_encoder import NumpyEncoder

# Definir logger para este módulo
logger = logging.getLogger(__name__)

# Necesita openpyxl para leer/escribir Excel si carga datos de heatmap
try:
    import openpyxl
except ImportError:
    logging.warning("Módulo 'openpyxl' no encontrado. Funcionalidades relacionadas con Excel pueden fallar.")
    pass


class PlotGenerator:
    """
    Servicio responsable de generar diferentes tipos de gráficos (líneas, barras, heatmaps)
    basados en configuraciones y datos de simulación (resumen o detallados).
    """
    def __init__(self, logger_instance: logging.Logger):
        """
        Inicializa el PlotGenerator.

        Args:
            logger_instance: Instancia del logger inyectado.
        """
        self.logger = logger_instance
        self.logger.info("PlotGenerator instance created.")
        # Directorio para buscar archivos de datos si detailed_data no se proporciona
        self._results_folder: Optional[str] = None

    # --- Métodos Helper Internos (Privados) ---

    def _find_latest_detailed_data_file(self) -> Optional[str]:
        """Busca el último archivo simulation_data_...json en la carpeta de resultados."""
        if not self._results_folder or not os.path.isdir(self._results_folder):
             self.logger.error("_find_latest_detailed_data_file: Carpeta de resultados no establecida o inválida.")
             return None

        latest_file: Optional[str] = None
        highest_episode_num: int = -1
        self.logger.debug(f"Buscando último archivo detailed_data en: {self._results_folder}")
        try:
            for filename in os.listdir(self._results_folder):
                if filename.startswith('simulation_data_') and filename.endswith('.json'):
                    parts = filename[:-5].split('_')
                    if len(parts) >= 4 and parts[-2] == 'to':
                        try:
                            last_episode = int(parts[-1])
                            if last_episode > highest_episode_num:
                                highest_episode_num = last_episode
                                latest_file = os.path.join(self._results_folder, filename)
                        except (ValueError, IndexError):
                            pass # Ignorar archivos con formato inválido
        except Exception as e:
            self.logger.error(f"Error buscando último archivo detailed_data: {e}", exc_info=True)
            return None

        if latest_file:
             self.logger.info(f"Último archivo detailed_data encontrado: {os.path.basename(latest_file)}")
        else:
             self.logger.warning(f"No se encontraron archivos simulation_data_*.json en {self._results_folder}")
        return latest_file

    def _load_detailed_data(self) -> Optional[List[Dict]]:
        """Carga datos detallados desde el último archivo JSON encontrado."""
        data_filepath = self._find_latest_detailed_data_file()
        if not data_filepath:
             return None
        try:
            self.logger.info(f"Cargando datos detallados desde: {os.path.basename(data_filepath)}")
            with open(data_filepath, 'r', encoding='utf-8') as f:
                detailed_data = json.load(f)
            if not isinstance(detailed_data, list):
                self.logger.error(f"Archivo detailed_data ({os.path.basename(data_filepath)}) no contiene una lista JSON.")
                return None
            self.logger.info(f"Cargados {len(detailed_data)} registros de episodios.")
            return detailed_data
        except Exception as e:
            self.logger.error(f"Error cargando datos detallados desde {data_filepath}: {e}", exc_info=True)
            return None


    def _apply_common_mpl_styles(self, ax: plt.Axes, fig: plt.Figure, plot_cfg: Dict):
        """Aplica estilos comunes de Matplotlib desde la configuración del plot."""
        cfg = plot_cfg.get('config', {}) # Obtener dict de config interno
        if not isinstance(cfg, dict): cfg = {} # Usar dict vacío si config no existe o es inválido

        # --- Títulos y Etiquetas ---
        title_fontsize = cfg.get('title_fontsize', 14)
        label_fontsize = cfg.get('label_fontsize', 12) # Usar uno para ambos ejes si no se especifica individualmente
        xlabel_fontsize = cfg.get('xlabel_fontsize', label_fontsize)
        ylabel_fontsize = cfg.get('ylabel_fontsize', label_fontsize)

        ax.set_title(cfg.get('title', ''), fontsize=title_fontsize, pad=cfg.get('title_pad', 10))
        ax.set_xlabel(cfg.get('xlabel', ''), fontsize=xlabel_fontsize, labelpad=cfg.get('xlabel_pad', 8))
        ax.set_ylabel(cfg.get('ylabel', ''), fontsize=ylabel_fontsize, labelpad=cfg.get('ylabel_pad', 8))

        # --- Ticks (Marcas en los ejes) ---
        tick_fontsize = cfg.get('tick_fontsize', 10)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=cfg.get('tick_pad', 5))
        # Rotación de ticks del eje X
        xtick_rotation = cfg.get('xtick_rotation', 0)
        if xtick_rotation != 0:
            plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right", rotation_mode="anchor")
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center") # Default alineación horizontal

        # Número de ticks (MaxNLocator)
        num_xticks = cfg.get('num_xticks')
        num_yticks = cfg.get('num_yticks')
        # Aplicar MaxNLocator solo si se especifica un número positivo
        if isinstance(num_xticks, int) and num_xticks > 0:
            # Determinar si usar integer=True (e.g., para episodios)
            is_episode_axis = 'episode' in cfg.get('xlabel', '').lower()
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=num_xticks, prune='both', integer=is_episode_axis))
        if isinstance(num_yticks, int) and num_yticks > 0:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=num_yticks, prune='both'))

        # --- Límites de los Ejes ---
        # Permitir None para auto-escalado
        xmin, xmax = cfg.get('xmin'), cfg.get('xmax')
        ymin, ymax = cfg.get('ymin'), cfg.get('ymax')
        if xmin is not None or xmax is not None: ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None: ax.set_ylim(bottom=ymin, top=ymax)

        # --- Rejilla (Grid) ---
        if cfg.get('grid_on', False):
            ax.grid(visible=True, linestyle=cfg.get('grid_linestyle', '--'), alpha=cfg.get('grid_alpha', 0.6))
        else:
            ax.grid(visible=False)

        # --- Leyenda ---
        show_legend = cfg.get('show_legend', bool(ax.get_legend_handles_labels()[0])) # Mostrar si hay algo que mostrar
        if show_legend:
            legend_pos = cfg.get('legend_pos', 'best')
            legend_fs = cfg.get('legend_fontsize', 'small')
            legend_title = cfg.get('legend_title')
            ncol = cfg.get('legend_ncol', 1) # Número de columnas en la leyenda

            # Manejo de posición 'outside'
            if legend_pos == 'outside':
                # Colocar fuera, arriba a la derecha del plot area
                ax.legend(title=legend_title, fontsize=legend_fs, ncol=ncol,
                          bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
                # Ajustar layout para dar espacio (se hace al final)
            else:
                ax.legend(title=legend_title, fontsize=legend_fs, ncol=ncol, loc=legend_pos)
        elif ax.get_legend() is not None:
            # Eliminar leyenda si no se quiere mostrar pero se creó automáticamente
            ax.get_legend().remove()


    def _save_plot(self, fig: plt.Figure, filename_base: str):
        """Guarda la figura en la carpeta de resultados."""
        if not self._results_folder:
             self.logger.error("No se puede guardar plot: Carpeta de resultados no establecida.")
             plt.close(fig) # Cerrar figura de todas formas
             gc.collect()
             return

        try:
            # Crear nombre completo del archivo (asegurar extensión .png)
            if not filename_base.lower().endswith('.png'):
                 filename_base += '.png'
            filepath = os.path.join(self._results_folder, filename_base)

            # Ajustar layout antes de guardar (especialmente para leyenda 'outside')
            # El ajuste debe considerar si la leyenda está fuera
            has_outside_legend = False
            legend = ax.get_legend() # type: ignore # Asumir que ax está definido donde se llama _save_plot
            if legend and hasattr(legend, '_outside_loc'): # Matplotlib interno puede cambiar
                  # Heurística: si bbox está fuera de (0,0) a (1,1)
                  bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()) # type: ignore
                  if bbox.x1 > 1 or bbox.y1 > 1 or bbox.x0 < 0 or bbox.y0 < 0:
                       has_outside_legend = True
            # Alternativa si 'legend_pos' se pasó en config
            # plot_cfg = ... # Necesitaría acceso a plot_cfg aquí
            # has_outside_legend = plot_cfg.get('config',{}).get('legend_pos') == 'outside' and plot_cfg.get('config',{}).get('show_legend',False)


            # Usar tight_layout o fig.subplots_adjust
            if has_outside_legend:
                 # Ajustar margen derecho para dar espacio a la leyenda
                 fig.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar rect [left, bottom, right, top]
            else:
                 fig.tight_layout()


            # Guardar la figura
            fig.savefig(filepath, dpi=cfg.get('dpi', 300), bbox_inches='tight') # type: ignore # cfg no definido aquí, usar default
            self.logger.info(f"Plot guardado exitosamente -> {os.path.basename(filepath)}")

        except Exception as e:
            self.logger.error(f"Fallo al guardar plot '{filename_base}': {e}", exc_info=True)
        finally:
            # Cerrar la figura siempre para liberar memoria
            plt.close(fig)
            # Llamar explícitamente al recolector de basura
            gc.collect()


    # --- Métodos de Ploteo Específicos (Privados o Públicos Protegidos) ---

    def _plot_generic_line(self, data: pd.DataFrame, plot_cfg: dict, filename: str):
        """Genera un gráfico de línea genérico."""
        cfg = plot_cfg.get('config', {})
        x_var = plot_cfg.get('x_variable')
        y_var = plot_cfg.get('y_variable')
        filter_reasons = cfg.get('filter_termination_reason') # Lista o None

        # Validar datos de entrada y variables
        if data is None or data.empty:
            self.logger.warning(f"No hay datos (DataFrame vacío) para generar gráfico de línea '{filename}'.")
            return
        if not x_var or not y_var:
            self.logger.warning(f"Variables X ('{x_var}') o Y ('{y_var}') no especificadas para línea '{filename}'.")
            return
        if x_var not in data.columns or y_var not in data.columns:
            self.logger.warning(f"Variables X ('{x_var}') o Y ('{y_var}') no encontradas en el DataFrame para línea '{filename}'. Columnas: {data.columns.tolist()}")
            return

        self.logger.info(f"Generando gráfico de línea: '{filename}' (Y={y_var} vs X={x_var})")
        fig, ax = None, None # Inicializar para bloque finally
        try:
            fig, ax = plt.subplots(figsize=(cfg.get('figsize_w', 12), cfg.get('figsize_h', 6)))

            # --- Preparación de Datos ---
            df = data.copy()
            # Filtrar por razón de terminación si se especifica
            if filter_reasons and 'termination_reason' in df.columns:
                 original_count = len(df)
                 # Asegurar que filter_reasons es lista de strings
                 if isinstance(filter_reasons, list) and all(isinstance(r, str) for r in filter_reasons):
                      df = df[df['termination_reason'].isin(filter_reasons)]
                      self.logger.info(f"Línea '{filename}': Filtrado por {filter_reasons}, {len(df)}/{original_count} puntos mantenidos.")
                 else:
                      self.logger.warning(f"Línea '{filename}': filtro 'filter_termination_reason' inválido ({filter_reasons}). Ignorando filtro.")

            # Convertir a numérico y eliminar NaNs para las variables del plot
            df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
            df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
            df = df.dropna(subset=[x_var, y_var]).sort_values(by=x_var)

            if df.empty:
                self.logger.warning(f"No quedan puntos de datos válidos para gráfico de línea '{filename}' tras filtrar/limpiar.")
                if fig: plt.close(fig); gc.collect()
                return

            x_data = df[x_var]
            y_data = df[y_var]

            # --- Ploteo ---
            ax.plot(x_data, y_data,
                    color=cfg.get('line_color', '#1f77b4'), # Default azul
                    linewidth=cfg.get('line_width', 1.5),
                    linestyle=cfg.get('line_style', '-'),
                    marker=cfg.get('marker_style', ''), # No marker por defecto
                    markersize=cfg.get('marker_size', 0),
                    label=cfg.get('y_label', y_var) # Usar ylabel para la etiqueta de la línea
                   )

            # --- Estilos ---
            # Auto-generar etiquetas si no están en config
            cfg['xlabel'] = cfg.get('xlabel', x_var.replace('_', ' ').title())
            cfg['ylabel'] = cfg.get('ylabel', y_var.replace('_', ' ').title())
            # Aplicar estilos comunes (título, etiquetas, grid, límites, leyenda, ticks)
            self._apply_common_mpl_styles(ax, fig, plot_cfg)

            # Guardar plot
            self._save_plot(fig, filename)

        except Exception as e:
            self.logger.error(f"Error generando gráfico de línea '{filename}': {e}\n{traceback.format_exc()}")
            if fig: plt.close(fig); gc.collect() # Asegurar cierre en caso de error


    def _plot_generic_bar(self, data: pd.DataFrame, plot_cfg: dict, filename: str):
        """Genera un gráfico de barras genérico (frecuencia o agrupado)."""
        cfg = plot_cfg.get('config', {})
        variable = plot_cfg.get('variable') # Variable a contar/agrupar
        group_by = cfg.get('group_by') # Variable para agrupar (e.g., 'episode')
        group_size = cfg.get('group_size') # Tamaño de grupo para agrupar variable numérica

        # Validar datos y variable principal
        if data is None or data.empty:
            self.logger.warning(f"No hay datos (DataFrame vacío) para generar gráfico de barras '{filename}'.")
            return
        if not variable or variable not in data.columns:
            self.logger.warning(f"Variable '{variable}' no especificada o no encontrada en DataFrame para barras '{filename}'. Columnas: {data.columns.tolist()}")
            return

        self.logger.info(f"Generando gráfico de barras: '{filename}' (Variable={variable})")
        fig, ax = None, None
        try:
            fig, ax = plt.subplots(figsize=(cfg.get('figsize_w', 10), cfg.get('figsize_h', 5)))

            # --- Preparación de Datos ---
            df = data.copy()
            plot_data = None
            is_stacked = False

            if group_by:
                # Agrupar por una variable categórica y contar 'variable'
                if group_by not in df.columns:
                     self.logger.warning(f"Variable de agrupación '{group_by}' no encontrada. Haciendo conteo simple de '{variable}'.")
                     plot_data = df[variable].value_counts().sort_index()
                     cfg['xlabel'] = cfg.get('xlabel', variable.replace('_', ' ').title())
                     cfg['ylabel'] = cfg.get('ylabel', 'Frequency')
                else:
                     # Crear tabla de contingencia (requiere que ambas sean categóricas o discretas)
                     try:
                          plot_data = pd.crosstab(df[group_by], df[variable])
                          is_stacked = True # Usar stacked bar para crosstab
                          cfg['xlabel'] = cfg.get('xlabel', group_by.replace('_', ' ').title())
                          cfg['ylabel'] = cfg.get('ylabel', 'Count')
                          # Título de leyenda basado en la variable contada
                          cfg['legend_title'] = cfg.get('legend_title', variable.replace('_', ' ').title())
                     except Exception as e_cross:
                          self.logger.warning(f"No se pudo crear crosstab para '{variable}' vs '{group_by}'. Haciendo conteo simple. Error: {e_cross}")
                          plot_data = df[variable].value_counts().sort_index()
                          cfg['xlabel'] = cfg.get('xlabel', variable.replace('_', ' ').title())
                          cfg['ylabel'] = cfg.get('ylabel', 'Frequency')

            elif group_size and 'episode' in df.columns:
                # Agrupar episodios numéricos y contar 'variable'
                try:
                    df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
                    df = df.dropna(subset=['episode'])
                    if df.empty: raise ValueError("No hay episodios numéricos válidos para agrupar.")
                    # Crear grupos de episodios
                    df['episode_group'] = (df['episode'] // group_size) * group_size
                    # Contar ocurrencias de 'variable' dentro de cada grupo de episodios
                    plot_data = df.groupby('episode_group')[variable].value_counts().unstack(fill_value=0)
                    is_stacked = True # Usar stacked bar
                    cfg['xlabel'] = cfg.get('xlabel', f'Episode Group (Size {group_size})')
                    cfg['ylabel'] = cfg.get('ylabel', 'Count')
                    cfg['legend_title'] = cfg.get('legend_title', variable.replace('_', ' ').title())
                except Exception as e_group:
                     self.logger.warning(f"Error agrupando episodios para barras '{filename}'. Haciendo conteo simple. Error: {e_group}")
                     plot_data = df[variable].value_counts().sort_index()
                     cfg['xlabel'] = cfg.get('xlabel', variable.replace('_', ' ').title())
                     cfg['ylabel'] = cfg.get('ylabel', 'Frequency')

            else:
                # Conteo simple de la 'variable'
                plot_data = df[variable].value_counts().sort_index()
                cfg['xlabel'] = cfg.get('xlabel', variable.replace('_', ' ').title())
                cfg['ylabel'] = cfg.get('ylabel', 'Frequency')

            # Verificar si hay datos para plotear tras el procesamiento
            if plot_data is None or plot_data.empty:
                self.logger.warning(f"No hay datos para plotear en gráfico de barras '{filename}' tras procesar.")
                if fig: plt.close(fig); gc.collect()
                return

            # --- Ploteo ---
            cmap = cfg.get('cmap', 'viridis') # Colormap
            bar_width = cfg.get('bar_width', 0.8)
            color_arg = plt.get_cmap(cmap) if is_stacked else plt.get_cmap(cmap)(np.linspace(0.1, 0.9, len(plot_data)))

            plot_data.plot(kind='bar', stacked=is_stacked, ax=ax,
                           colormap=cmap if is_stacked else None,
                           color=None if is_stacked else color_arg,
                           width=bar_width,
                           edgecolor=cfg.get('bar_edgecolor', 'black'),
                           linewidth=cfg.get('bar_linewidth', 0.5))

            # --- Estilos ---
            cfg['show_legend'] = cfg.get('show_legend', is_stacked) # Mostrar leyenda por defecto si es stacked
            self._apply_common_mpl_styles(ax, fig, plot_cfg)
            # Rotar etiquetas X si son muchas o largas
            if len(plot_data.index) > 10 and not isinstance(plot_data.index, pd.RangeIndex):
                 plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


            # Guardar plot
            self._save_plot(fig, filename)

        except Exception as e:
            self.logger.error(f"Error generando gráfico de barras '{filename}': {e}\n{traceback.format_exc()}")
            if fig: plt.close(fig); gc.collect()


    def _extract_heatmap_data(self, detailed_data: list, x_var: str, y_var: str, filter_reasons: list = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrae datos X, Y para heatmaps desde datos detallados.
        Reutiliza la lógica de HeatmapGenerator._extract_heatmap_data si es necesario,
        o la mantiene aquí si PlotGenerator necesita autonomía.
        (Copiada de HeatmapGenerator._extract_heatmap_data en Paso 3 por completitud)
        """
        # --- Inicio Copia de HeatmapGenerator._extract_heatmap_data ---
        x_all, y_all = [], []
        if not isinstance(detailed_data, list): return np.array([]), np.array([])
        episodes = detailed_data
        if filter_reasons:
            if not isinstance(filter_reasons, list) or not all(isinstance(r, str) for r in filter_reasons):
                 self.logger.warning(f"_extract_heatmap_data: filter_reasons inválido ({filter_reasons}). Ignorando.")
                 filter_reasons = None
            else:
                try:
                    original_count = len(episodes); episodes = [ep for ep in episodes if isinstance(ep, dict) and ep.get('termination_reason') in filter_reasons]; filtered_count = len(episodes)
                    self.logger.info(f"_extract_heatmap_data: Filtro {filter_reasons}: {filtered_count}/{original_count} episodios.")
                except Exception as e: self.logger.error(f"Error filtrando episodios para heatmap: {e}", exc_info=True); episodes = detailed_data; filter_reasons = None
        if not episodes: return np.array([]), np.array([])
        valid_points_count = 0
        for i, ep_data in enumerate(episodes):
            if not isinstance(ep_data, dict): continue
            x_raw = ep_data.get(x_var); y_raw = ep_data.get(y_var)
            if x_raw is None or y_raw is None: continue
            if not isinstance(x_raw, (list, np.ndarray)) or not isinstance(y_raw, (list, np.ndarray)): continue
            try:
                min_len = min(len(x_raw), len(y_raw)); x_num = pd.to_numeric(x_raw[:min_len], errors='coerce'); y_num = pd.to_numeric(y_raw[:min_len], errors='coerce'); mask = np.isfinite(x_num) & np.isfinite(y_num); num_valid_in_ep = np.sum(mask)
                if num_valid_in_ep > 0: x_all.append(x_num[mask]); y_all.append(y_num[mask]); valid_points_count += num_valid_in_ep
            except Exception as e: self.logger.warning(f"Error procesando datos episodio {ep_data.get('episode', i)} para heatmap ({x_var}, {y_var}): {e}"); continue
        if not x_all or not y_all: self.logger.warning(f"No hay puntos válidos para heatmap '{y_var}' vs '{x_var}' {' con filtro ' + str(filter_reasons) if filter_reasons else ''}."); return np.array([]), np.array([])
        try: x_combined = np.concatenate(x_all); y_combined = np.concatenate(y_all); self.logger.info(f"Datos extraídos heatmap '{y_var}' vs '{x_var}': {len(x_combined)} puntos."); return x_combined, y_combined
        except Exception as e: self.logger.error(f"Error concatenando datos heatmap ({x_var}, {y_var}): {e}", exc_info=True); return np.array([]), np.array([])
        # --- Fin Copia ---


    def _plot_generic_heatmap(self, detailed_data: list, plot_cfg: dict, filename: str):
        """Genera un gráfico de heatmap (histograma 2D)."""
        cfg = plot_cfg.get('config', {})
        x_var = plot_cfg.get('x_variable')
        y_var = plot_cfg.get('y_variable')
        filter_reasons = cfg.get('filter_termination_reason') # Lista o None

        # Validar datos y variables
        if not detailed_data:
            self.logger.warning(f"No hay datos detallados para generar heatmap '{filename}'.")
            return
        if not x_var or not y_var:
            self.logger.warning(f"Variables X ('{x_var}') o Y ('{y_var}') no especificadas para heatmap '{filename}'.")
            return

        self.logger.info(f"Generando heatmap: '{filename}' (Y={y_var} vs X={x_var})")
        fig, ax = None, None
        try:
            # --- Preparación de Datos ---
            # Extraer datos (puede requerir cargar desde archivo si detailed_data está vacío)
            if not detailed_data: # Lista vacía
                 loaded_data = self._load_detailed_data()
                 if loaded_data is None:
                      self.logger.error(f"No se pudieron cargar datos detallados para heatmap '{filename}'.")
                      return
                 detailed_data = loaded_data # Usar datos cargados

            x_data, y_data = self._extract_heatmap_data(detailed_data, x_var, y_var, filter_reasons)

            if x_data.size == 0 or y_data.size == 0:
                self.logger.warning(f"No se encontraron puntos de datos válidos para heatmap '{filename}' tras extraer/filtrar.")
                return

            # --- Ploteo (Histograma 2D) ---
            fig, ax = plt.subplots(figsize=(cfg.get('figsize_w', 10), cfg.get('figsize_h', 8))) # Un poco más alto para colorbar

            bins = cfg.get('bins', 50)
            if not isinstance(bins, int) or bins <=0: bins=50
            cmap_name = cfg.get('cmap', 'inferno') # 'hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
            log_scale = cfg.get('log_scale', False) # Usar escala logarítmica para counts?
            # Límites para la barra de color (counts)
            cmin, cmax = cfg.get('cmin'), cfg.get('cmax')

            # Rango para el histograma (límites de ejes X e Y)
            xmin, xmax = cfg.get('xmin'), cfg.get('xmax')
            ymin, ymax = cfg.get('ymin'), cfg.get('ymax')
            hist_range = None
            if all(isinstance(v, (int, float)) and np.isfinite(v) for v in [xmin, xmax, ymin, ymax]):
                 if xmin < xmax and ymin < ymax: hist_range = [[xmin, xmax], [ymin, ymax]]
                 else: self.logger.warning(f"Límites de rango inválidos para heatmap '{filename}'. Ignorando.")
            elif any(v is not None for v in [xmin, xmax, ymin, ymax]):
                 self.logger.warning(f"Límites de rango incompletos/inválidos para heatmap '{filename}'. Ignorando.")

            # Configurar normalización de color (lineal o log)
            norm = None
            if log_scale:
                 # LogNorm necesita vmin > 0 si se especifica. Si cmin es 0 o None, que determine automáticamente.
                 log_vmin = cmin if cmin is not None and cmin > 0 else None
                 norm = mcolors.LogNorm(vmin=log_vmin, vmax=cmax)
            else:
                 norm = mcolors.Normalize(vmin=cmin, vmax=cmax) # Funciona bien con None

            # cmin para hist2d: umbral mínimo de counts para mostrar un bin
            # Funciona mejor con escala lineal. Con Log, mejor controlar vmin en LogNorm.
            hist_cmin = cmin if not log_scale and cmin is not None and cmin > 0 else None


            # Generar el histograma 2D
            counts, xedges, yedges, img = ax.hist2d(
                x_data, y_data,
                bins=bins,
                range=hist_range,
                cmap=cmap_name,
                norm=norm,
                cmin=hist_cmin # Umbral mínimo de counts
            )

            # --- Barra de Color ---
            cbar = fig.colorbar(img, ax=ax, pad=0.02) # Añadir un poco de padding
            cbar_label = cfg.get('clabel', 'Frequency')
            if log_scale: cbar_label += ' (Log Scale)'
            cbar.set_label(cbar_label, fontsize=cfg.get('clabel_fontsize', 10))
            cbar.ax.tick_params(labelsize=cfg.get('tick_fontsize', 10))

            # --- Estilos ---
            cfg['xlabel'] = cfg.get('xlabel', x_var.replace('_', ' ').title())
            cfg['ylabel'] = cfg.get('ylabel', y_var.replace('_', ' ').title())
            # Aplicar estilos comunes DESPUÉS de hist2d y colorbar
            self._apply_common_mpl_styles(ax, fig, plot_cfg)
            # Asegurar que grid_on=False por defecto para heatmaps si no se especifica
            if 'grid_on' not in cfg: ax.grid(visible=False)


            # Guardar plot
            self._save_plot(fig, filename)

        except Exception as e:
            self.logger.error(f"Error generando heatmap '{filename}': {e}\n{traceback.format_exc()}")
            if fig: plt.close(fig); gc.collect()


    # --- Método Público Principal (Dispatcher) ---

    def generate(self, plot_configs: list, summary_df: pd.DataFrame, detailed_data: list, results_folder: str):
        """
        Genera todos los gráficos definidos en plot_configs.

        Args:
            plot_configs (list): Lista de diccionarios de configuración de plots
                                 (de vis_config['plots']).
            summary_df (pd.DataFrame): DataFrame con los datos de resumen por episodio.
            detailed_data (list): Lista con datos detallados por episodio (puede estar vacía).
            results_folder (str): Carpeta donde se guardarán los gráficos.
        """
        if not results_folder or not os.path.isdir(results_folder):
             self.logger.error("PlotGenerator.generate: Carpeta de resultados inválida o no proporcionada. No se pueden generar gráficos.")
             return
        self._results_folder = results_folder # Guardar para uso interno (_save_plot, _load_detailed)

        if not plot_configs:
            self.logger.info("No hay configuraciones de plots para generar.")
            return

        # Mapeo de tipos de plot a métodos de generación
        plot_functions = {
            'bar': self._plot_generic_bar,
            'line': self._plot_generic_line,
            'heatmap': self._plot_generic_heatmap
            # Añadir más tipos aquí (e.g., scatter, histogram)
        }

        self.logger.info(f"Iniciando generación de {len(plot_configs)} plots configurados...")

        for i, plot_cfg in enumerate(plot_configs):
            if not isinstance(plot_cfg, dict):
                self.logger.warning(f"Saltando config de plot #{i+1}: formato inválido (no es diccionario).")
                continue

            plot_type = plot_cfg.get('type')
            is_enabled = plot_cfg.get('enabled', True)
            output_filename = plot_cfg.get('output_filename', f"plot_{plot_type}_{i+1}.png")

            if not is_enabled:
                self.logger.info(f"Saltando plot '{output_filename}' (Tipo: {plot_type}) - Deshabilitado.")
                continue

            if plot_type not in plot_functions:
                self.logger.warning(f"Saltando plot '{output_filename}': Tipo de plot desconocido '{plot_type}'. "
                                    f"Tipos disponibles: {list(plot_functions.keys())}")
                continue

            plot_func = plot_functions[plot_type]
            source = plot_cfg.get('source') # 'summary' o 'detailed'

            self.logger.info(f"--- Generando Plot {i+1}/{len(plot_configs)}: '{output_filename}' (Tipo: {plot_type}, Fuente: {source}) ---")

            try:
                # Seleccionar datos y llamar a la función de ploteo
                if source == 'summary':
                     if summary_df is None or summary_df.empty:
                          self.logger.warning(f"Fuente 'summary' seleccionada para '{output_filename}', pero summary_df está vacío. Saltando plot.")
                          continue
                     # Llamar a la función con el DataFrame de resumen
                     plot_func(data=summary_df, plot_cfg=plot_cfg, filename=output_filename)

                elif source == 'detailed':
                     # Llamar a la función con la lista de datos detallados
                     # La función interna (_plot_generic_heatmap o similar) manejará si necesita cargar datos si la lista está vacía.
                     plot_func(detailed_data=detailed_data, plot_cfg=plot_cfg, filename=output_filename)

                else:
                    self.logger.warning(f"Saltando plot '{output_filename}': Fuente de datos ('source') inválida o faltante: '{source}'. Usar 'summary' o 'detailed'.")
                    continue

            except Exception as e:
                # Loguear error pero continuar con los siguientes plots
                self.logger.error(f"Fallo al generar plot '{output_filename}': {e}", exc_info=True)
                # Asegurar que la figura se cierre si hubo error durante la creación/guardado
                # (plt.close se llama en _save_plot y en los bloques except de _plot_generic_*)

        self.logger.info("Generación de todos los plots configurados finalizada.")