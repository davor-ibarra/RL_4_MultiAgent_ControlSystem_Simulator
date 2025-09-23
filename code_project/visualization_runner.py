import logging
import pandas as pd
import os # Para manejo de rutas opcional
from typing import List, Dict, Optional

# Importar el contenedor para resolver dependencias
from di_container import Container
# Importar PlotGenerator (real o placeholder)
from di_container import PlotGenerator # Asume que está definido en di_container

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

def run_visualizations(vis_config: Optional[Dict],
                       summary_data: List[Dict],
                       all_episodes_data: List[Dict], # Datos detallados (lista vacía si se guarda por batch)
                       results_folder: str,
                       container: Container): # Recibir contenedor
    """
    Orquesta la generación de gráficas basadas en la configuración y los datos.
    Resuelve PlotGenerator desde el contenedor DI.

    Args:
        vis_config: Configuración de visualización cargada (puede ser None).
        summary_data: Lista de diccionarios de resumen por episodio.
        all_episodes_data: Lista de datos detallados (puede estar vacía).
        results_folder: Carpeta donde se guardarán las figuras.
        container: Instancia del contenedor DI para resolver PlotGenerator.
    """
    logger.info("--- Iniciando Generación de Visualizaciones ---")

    # --- Validación Inicial ---
    if not vis_config:
        logger.info("Config visualización no proporcionada. Omitiendo visualizaciones.")
        return
    if not isinstance(vis_config, dict): # Doble check
         logger.warning("vis_config no es un diccionario válido. Omitiendo visualizaciones.")
         return
    # 'enabled' flag ya fue chequeado en config_loader, no es necesario aquí de nuevo
    # if not vis_config.get('enabled', False): ...

    plot_configs = vis_config.get('plots')
    if not plot_configs or not isinstance(plot_configs, list):
        logger.warning("Config visualización sin sección 'plots' o no es lista. Omitiendo.")
        return

    # Comprobar si hay datos para graficar (al menos resumen)
    if not summary_data and not all_episodes_data:
        logger.warning("No hay datos resumen ni detallados disponibles para visualización.")
        # PlotGenerator podría intentar cargar desde archivos si se implementa así
        # return # O dejar que PlotGenerator maneje la falta de datos

    # Convertir resumen a DataFrame si existe
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    # --- Resolver y Ejecutar PlotGenerator ---
    try:
        logger.info("Resolviendo PlotGenerator desde contenedor...")
        # Resolver la dependencia usando el contenedor inyectado
        plot_generator: PlotGenerator = container.resolve(PlotGenerator)
        logger.info(f"PlotGenerator resuelto: {type(plot_generator).__name__}")

        # Llamar al método principal del generador de gráficos
        logger.info("Ejecutando PlotGenerator.generate...")
        # PlotGenerator debe definir qué argumentos necesita exactamente
        plot_generator.generate(
            plot_configs=plot_configs, # Pasar solo la lista de plots
            summary_df=summary_df,
            # Pasar la ruta a la carpeta, PlotGenerator carga datos detallados si los necesita
            results_folder=results_folder,
            # Opcional: pasar vis_config completo si necesita más info
            # vis_config=vis_config
            # Opcional: pasar lista detallada (probablemente mejor no si es grande)
            # detailed_data_list=all_episodes_data
        )
        logger.info("--- Generación de Visualizaciones Completada ---")

    except ValueError as e: # Error específico de resolución DI
         logger.error(f"Error resolviendo PlotGenerator: {e}", exc_info=True)
    except AttributeError as e: # Si PlotGenerator no tiene el método esperado
         logger.error(f"Error: PlotGenerator resuelto no tiene método esperado (e.g., 'generate'): {e}", exc_info=True)
    except FileNotFoundError as e: # Si PlotGenerator intenta cargar datos y no los encuentra
         logger.error(f"Error de archivo durante visualización (¿datos detallados no encontrados?): {e}", exc_info=True)
    except ImportError as e: # Si PlotGenerator requiere una librería no instalada (e.g., matplotlib)
         logger.error(f"Error de importación durante visualización. Falta librería: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error inesperado durante generación de visualizaciones: {e}", exc_info=True)