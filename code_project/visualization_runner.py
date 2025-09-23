import logging
import pandas as pd
import os # No estrictamente necesario aquí si PlotGenerator maneja rutas
from typing import List, Dict, Optional

# Importar el contenedor para resolver dependencias y PlotGenerator como placeholder
from di_container import Container, PlotGenerator # Usar PlotGenerator real cuando esté definido

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

def run_visualizations(vis_config: Optional[Dict],
                       summary_data: List[Dict],
                       all_episodes_data: List[Dict], # Datos detallados (usualmente lista vacía si se guardó por batch)
                       results_folder: str,
                       container: Container): # Recibir contenedor para resolver PlotGenerator
    """
    Orquesta la generación de gráficas basadas en la configuración de visualización
    y los datos de la simulación. Resuelve PlotGenerator desde el contenedor.

    Args:
        vis_config: Configuración de visualización cargada (puede ser None).
        summary_data: Lista de diccionarios de resumen por episodio.
        all_episodes_data: Lista de datos detallados (puede estar vacía si se guardó en batches).
                           PlotGenerator podría necesitar cargar datos desde archivos si esta lista está vacía.
        results_folder: Carpeta donde se guardarán las figuras generadas.
        container: Instancia del contenedor DI para resolver PlotGenerator.
    """
    logger.info("--- Iniciando Generación de Visualizaciones ---")

    # --- Validación Inicial ---
    if not vis_config:
        logger.info("Configuración de visualización no proporcionada o no cargada. Omitiendo visualizaciones.")
        return

    if not vis_config.get('enabled', False):
        logger.info("Visualización deshabilitada en la configuración. Omitiendo.")
        return

    plot_configs = vis_config.get('plots')
    if not plot_configs or not isinstance(plot_configs, list):
        logger.warning("Configuración de visualización sin sección 'plots' o no es una lista. Omitiendo visualizaciones.")
        return

    # Comprobar si hay datos para graficar (al menos resumen)
    if not summary_data and not all_episodes_data:
        logger.warning("No hay datos de resumen ni detallados disponibles para generar visualizaciones.")
        # PlotGenerator podría intentar cargar desde archivos, manejar eso internamente.
        # return # Podríamos retornar aquí o dejar que PlotGenerator maneje la carga

    # Convertir resumen a DataFrame si existe
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    # --- Resolver y Ejecutar PlotGenerator ---
    try:
        logger.info("Resolviendo PlotGenerator desde el contenedor...")
        # Asume que PlotGenerator está registrado en el contenedor (hecho en Paso 1)
        plot_generator: PlotGenerator = container.resolve(PlotGenerator)
        logger.info("PlotGenerator resuelto exitosamente.")

        # Llamar al método principal del generador de gráficos
        # Pasar datos, configs y carpeta de salida
        logger.info("Ejecutando PlotGenerator.generate...")
        plot_generator.generate(
            plot_configs=plot_configs,
            summary_df=summary_df,
            # Pasar datos detallados (puede ser lista vacía)
            # PlotGenerator decidirá si usarla o cargar desde archivo
            detailed_data=all_episodes_data,
            results_folder=results_folder
            # Pasar config completa si PlotGenerator la necesita? O solo vis_config?
            # vis_config=vis_config # Pasar si necesita más que solo 'plots'
        )
        logger.info("--- Generación de Visualizaciones Completada ---")

    except ValueError as e:
         # Capturar errores específicos de resolución del contenedor
         logger.error(f"Error resolviendo PlotGenerator: {e}", exc_info=True)
    except AttributeError as e:
         # Si PlotGenerator no tiene el método 'generate'
         logger.error(f"Error: PlotGenerator resuelto no tiene el método 'generate'. {e}", exc_info=True)
    except Exception as e:
        # Capturar cualquier otro error durante la generación de gráficos
        logger.error(f"Error inesperado durante la generación de visualizaciones: {e}", exc_info=True)