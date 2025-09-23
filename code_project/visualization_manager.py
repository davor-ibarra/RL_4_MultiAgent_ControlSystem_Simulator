# utils/plotting/visualization_manager.py
import logging
import os
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import pandas as pd

# 2.1: Importaciones para visualizaciones
from interfaces.plot_generator import PlotGenerator
from utils.data.heatmap_generator import HeatmapGenerator

# 2.2: TYPE_CHECKING para Container x si se llega a usar (actualmente no se usa directamente)
if TYPE_CHECKING:
    from di_container import Container

# 2.3: Logger a nivel de módulo (aunque la instancia inyectada prevalece)
logger = logging.getLogger(__name__)

class VisualizationManager:
    """
    Orquesta la generación de visualizaciones basadas en la configuración.
    Coordina la creación de datos de heatmap y luego la generación de plots individuales.
    """
    def __init__(self, logger_instance: logging.Logger, plot_generator: PlotGenerator,
                 heatmap_generator: HeatmapGenerator, vis_config_data: Optional[Dict[str, Any]],
                 results_folder: str):
        # 2.4: Usar logger inyectado
        self.logger = logger_instance
        self.plot_generator = plot_generator
        self.heatmap_generator = heatmap_generator
        # Asegurar que vis_config sea un dict, aunque esté vacío
        self.vis_config = vis_config_data if isinstance(vis_config_data, dict) else {}
        self.results_folder = results_folder

        self.logger.info("VisualizationManager instance created.")

        # 2.5: Validaciones básicas de dependencias
        if not isinstance(logger_instance, logging.Logger):
            logger.error("VisualizationManager: Logger inválido proporcionado.")
            raise TypeError("Logger inválido.")
        if plot_generator is None:
            self.logger.error("VisualizationManager: PlotGenerator no proporcionado.")
            raise ValueError("PlotGenerator es requerido.")
        if heatmap_generator is None:
            self.logger.error("VisualizationManager: HeatmapGenerator no proporcionado.")
            raise ValueError("HeatmapGenerator es requerido.")
        if not os.path.isdir(results_folder):
            self.logger.error(f"VisualizationManager: Carpeta de resultados no válida: {results_folder}")

    def _generate_heatmap_data_if_needed(self):
        """Genera datos para heatmaps si están configurados."""
        # 2.6: Obtener configuraciones de plots del dict de config
        plot_configs = self.vis_config.get("plots", [])
        if not isinstance(plot_configs, list):
            self.logger.warning("La clave 'plots' en la configuración de visualización no es una lista. No se generarán plots.")
            return

        # 2.7: Filtrar solo las configuraciones de tipo 'heatmap' que estén habilitadas
        #      y añadirles el índice interno para referencia en HeatmapGenerator.
        heatmap_plot_configs_with_index: List[Dict] = []
        for i, p_cfg in enumerate(plot_configs):
            # Comprobar que sea un dict, tenga tipo 'heatmap' y esté habilitado (o enabled no exista -> True)
            if isinstance(p_cfg, dict) and \
                p_cfg.get("type") == "heatmap" and \
                p_cfg.get("enabled", True):
                # Añadir el índice original de la lista a la configuración
                p_cfg_copy = p_cfg.copy()
                p_cfg_copy['_internal_plot_index'] = i
                heatmap_plot_configs_with_index.append(p_cfg_copy)


        if not heatmap_plot_configs_with_index:
            self.logger.info("No hay plots de tipo 'heatmap' habilitados en la configuración. No se generará archivo de datos heatmap.")
            return

        self.logger.info(f"Iniciando pre-generación de datos para {len(heatmap_plot_configs_with_index)} heatmaps...")
        # 2.8: Definir nombre de archivo Excel de salida
        output_excel_filepath = os.path.join(self.results_folder, "data_heatmaps.xlsx")

        try:
            # 2.9: Llamar a HeatmapGenerator.generate pasando la lista filtrada y con índice
            self.heatmap_generator.generate(
                results_folder=self.results_folder,
                heatmap_configs=heatmap_plot_configs_with_index, # Pasa la lista procesada
                output_excel_filepath=output_excel_filepath
            )
            # El logging de éxito/fracaso está dentro de HeatmapGenerator.generate
            self.logger.info("Llamada a pre-generación de datos para heatmaps completada.")
        except Exception as e:
            # Captura errores inesperados en la llamada a generate
            self.logger.error(f"Error inesperado durante la llamada a HeatmapGenerator.generate: {e}", exc_info=True)

    def run(self):
        """Ejecuta la generación de todas las visualizaciones configuradas."""
        # 2.10: Obtener configuraciones de plots
        plot_configs_raw = self.vis_config.get("plots", [])
        if not isinstance(plot_configs_raw, list) or not plot_configs_raw:
            self.logger.warning("No hay configuraciones de plots válidas en 'vis_config'. No se generarán visualizaciones.")
            return

        self.logger.info(f"Iniciando generación de hasta {len(plot_configs_raw)} plots configurados...")

        # 2.11: Generar datos de heatmap PRIMERO
        self._generate_heatmap_data_if_needed()

        # 2.12: Iterar sobre TODAS las configuraciones de plots para generar los gráficos individuales
        num_generated = 0
        num_skipped_disabled = 0
        num_skipped_invalid = 0
        for i, plot_cfg_original in enumerate(plot_configs_raw):
            # Validar formato y si está habilitado
            if not isinstance(plot_cfg_original, dict):
                self.logger.warning(f"Configuración de plot #{i+1} ignorada: no es un diccionario.")
                num_skipped_invalid += 1
                continue
            if not plot_cfg_original.get("enabled", True):
                plot_name_debug = plot_cfg_original.get("name", f"idx_{i}")
                self.logger.info(f"Plot '{plot_name_debug}' (índice {i}) deshabilitado. Saltando.")
                num_skipped_disabled +=1
                continue # Saltar si está explícitamente deshabilitado

            # Añadir índice interno para referencia en PlotGenerator si es necesario
            # (MatplotlibPlotGenerator ya lo recibe en Paso 1 y 2)
            plot_cfg_copy = plot_cfg_original.copy()
            plot_cfg_copy['_internal_plot_index'] = i

            plot_name = plot_cfg_copy.get("name", f"plot_{plot_cfg_copy.get('type', 'unknown')}_{i+1}")
            self.logger.info(f"--- Generando Plot #{i+1}: '{plot_name}' (Tipo: {plot_cfg_copy.get('type', 'N/A')}) ---")

            try:
                # 2.13: Llamar a la interfaz PlotGenerator
                self.plot_generator.generate_plot(
                    plot_config=plot_cfg_copy,
                    results_folder=self.results_folder
                )
                num_generated += 1
            except NotImplementedError as nie:
                self.logger.error(f"Error generando plot '{plot_name}': Tipo de plot no implementado por PlotGenerator: {nie}")
            except FileNotFoundError as fnfe:
                self.logger.error(f"Error generando plot '{plot_name}': Archivo de datos no encontrado: {fnfe}")
            except ValueError as ve:
                self.logger.error(f"Error generando plot '{plot_name}': Datos o configuración inválidos: {ve}")
            except Exception as e:
                self.logger.error(f"Error inesperado generando plot '{plot_name}': {e}", exc_info=True)

        self.logger.info(f"Generación de visualizaciones finalizada.")
        self.logger.info(f"  - Intentados/Generados: {num_generated}")
        self.logger.info(f"  - Saltados (Deshabilitados): {num_skipped_disabled}")
        self.logger.info(f"  - Saltados (Inválidos): {num_skipped_invalid}")