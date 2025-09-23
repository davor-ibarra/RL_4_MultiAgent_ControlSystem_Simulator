# visualization_runner.py
import logging
import pandas as pd
import os
from typing import List, Dict, Optional

# Importar el contenedor para resolver dependencias
from di_container import Container
# --- IMPORT RENOMBRADO ---
from utils.visualization_generator import VisualizationGenerator # Antes PlotGenerator
# -------------------------

logger = logging.getLogger(__name__)

def run_visualizations(vis_config: Optional[Dict],
                       summary_data: List[Dict],
                       all_episodes_data: List[Dict], # No usado directamente aquí
                       results_folder: str,
                       container: Container):
    """
    Orquesta la generación de gráficos usando VisualizationGenerator resuelto desde DI.

    Args:
        vis_config: Configuración de visualización cargada (puede ser None).
        summary_data: Lista de diccionarios de resumen por episodio.
        all_episodes_data: Lista de datos detallados (ignorado aquí, VisGen carga si necesita).
        results_folder: Carpeta donde se guardarán las figuras.
        container: Instancia del contenedor DI para resolver VisualizationGenerator.
    """
    logger.info("--- Iniciando Generación de Visualizaciones ---")

    if not vis_config or not isinstance(vis_config, dict):
        logger.info("Config visualización no válida/ausente. Omitiendo visualizaciones.")
        return
    plot_configs = vis_config.get('plots')
    if not plot_configs or not isinstance(plot_configs, list):
        logger.warning("Config visualización sin sección 'plots' válida. Omitiendo.")
        return
    if not summary_data:
        logger.warning("No hay datos de resumen disponibles para visualización (summary_df estará vacío).")
        # VisGen podría aún generar plots basados en datos detallados si los carga

    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    # --- Resolver y Ejecutar VisualizationGenerator ---
    try:
        logger.info("Resolviendo VisualizationGenerator desde contenedor...")
        # --- RESOLVER CLASE RENOMBRADA ---
        vis_generator: VisualizationGenerator = container.resolve(VisualizationGenerator)
        # ---------------------------------
        logger.info(f"VisualizationGenerator resuelto: {type(vis_generator).__name__}")

        logger.info("Ejecutando VisualizationGenerator.generate...")
        vis_generator.generate(
            plot_configs=plot_configs,
            summary_df=summary_df,
            results_folder=results_folder,
            # No pasamos all_episodes_data, VisGen lo carga si necesita
        )
        logger.info("--- Generación de Visualizaciones Completada ---")

    except ValueError as e: logger.error(f"Error resolviendo VisualizationGenerator: {e}", exc_info=True)
    except AttributeError as e: logger.error(f"Error: VisualizationGenerator resuelto sin método 'generate': {e}", exc_info=True)
    except FileNotFoundError as e: logger.error(f"Error de archivo durante visualización (¿datos detallados no encontrados?): {e}", exc_info=True)
    except ImportError as e: logger.error(f"Error importación matplotlib/libs req. por VisualizationGenerator: {e}", exc_info=True)
    except Exception as e: logger.error(f"Error inesperado durante generación de visualizaciones: {e}", exc_info=True)