import time
import logging
import os # Necesario para os.path.join
from typing import Optional, Any, Dict # Añadir tipos

from config_loader import load_and_validate_config
from logging_configurator import configure_file_logger
from di_container import build_container, Container # Importar Container

# Importar servicios y manejadores que se resolverán
from result_handler import ResultHandler
from simulation_manager import SimulationManager
from visualization_runner import run_visualizations
from interfaces.rl_agent import RLAgent # Para type hint

# Configuración básica de logging (consola) - Se configurará a fichero después si aplica
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
# Obtener logger raíz una vez
root_logger = logging.getLogger()

def main():
    """
    Función principal que orquesta la simulación completa usando inyección de dependencias.
    """
    # --- [1] Timer de ejecución ---
    start_time = time.time()

    # --- [2] Carga de configuración ---
    config, vis_config, logging_config = load_and_validate_config('config.yaml')
    if config is None:
        root_logger.critical("Fallo al cargar la configuración. Abortando.")
        return

    # --- [3] Preparación carpeta de resultados ---
    try:
        # Usar una instancia temporal de ResultHandler solo para esto si no se quiere DI aún,
        # o asumir que setup_results_folder puede ser estático. Hagámoslo estático por ahora.
        results_folder = ResultHandler.setup_results_folder(config['environment']['results_folder'])
    except Exception as e:
        root_logger.error(f"No se pudo crear carpeta resultados: {e}", exc_info=True)
        return

    # --- [4] Configuración de logging a fichero ---
    # Pasar results_folder explícitamente
    configure_file_logger(logging_config, results_folder)
    root_logger.info(f"Resultados se guardarán en: {results_folder}")
    root_logger.info("Logging a fichero configurado (si está habilitado).")

    # --- [5] Construcción del contenedor DI ---
    try:
        container: Container = build_container(config)
        # Registrar carpeta de resultados como dependencia singleton
        container.register(str, lambda c: results_folder, singleton=True)
        root_logger.info("Contenedor de Inyección de Dependencias construido.")
    except Exception as e:
        root_logger.critical(f"Error crítico construyendo el contenedor DI: {e}", exc_info=True)
        return

    # Obtener logger configurado del contenedor para usar de aquí en adelante
    logger = container.resolve(logging.Logger)

    # --- [6] Guardado de metadata inicial ---
    try:
        # Resolver ResultHandler desde el contenedor
        result_handler = container.resolve(ResultHandler)
        metadata = {
            'environment_details': {
                'code_version': '4.0.0', # Actualizar versión
                'run_timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
                'results_folder': results_folder,
                # Acceder a config via contenedor para consistencia
                'reward_mode': container.resolve(dict)['environment']['reward_setup'].get('learning_strategy')
            },
            'config_parameters': container.resolve(dict), # Usar config del contenedor
            'visualization_config': vis_config
        }
        # Llamar al método de instancia con la carpeta
        result_handler.save_metadata(metadata, results_folder)
        logger.info("Metadata inicial guardada.")
    except Exception as e:
        logger.error(f"Error guardando metadata: {e}", exc_info=True)
        # Considerar si continuar o abortar aquí

    # --- [7] Inicialización de componentes y ejecución de simulación ---
    all_data, summary = [], []
    agent_instance: Optional[RLAgent] = None # Para el guardado final
    try:
        # Resolver SimulationManager desde el contenedor
        # Ya no se necesita WorldInitializer
        logger.info("Resolviendo SimulationManager...")
        sim_manager = container.resolve(SimulationManager)
        logger.info("SimulationManager resuelto. Iniciando simulación...")

        # SimulationManager ahora es responsable de obtener sus propias dependencias (Environment, Agent, etc.)
        # El método run solo necesita el config y la carpeta de resultados (que puede obtener del contenedor también)
        # Modificamos la llamada: sim_manager.run() obtiene config y results_folder del container
        all_data, summary = sim_manager.run() # Se simplifica la llamada
        logger.info("Simulación finalizada.")

        # Obtener la instancia del agente para el guardado final (si es necesario)
        # Esto asume que el SimulationManager puede exponerlo o que lo resolvemos directamente
        # Por simplicidad, resolvemos Agent directamente si es necesario para finalize
        if config.get('simulation', {}).get('save_agent_state', False):
            agent_instance = container.resolve(RLAgent) # Resolver agente al final

    except ValueError as e:
         # Error específico de DI (e.g., dependencia no registrada)
         logger.critical(f"Error de Inyección de Dependencias: {e}", exc_info=True)
         # No continuar si DI falla aquí
         return
    except Exception as e:
        logger.error(f"Error crítico durante la ejecución de la simulación: {e}", exc_info=True)
        # all_data y summary ya están inicializados como listas vacías

    # --- [8] Finalización de resultados ---
    try:
        # Usar la misma instancia de ResultHandler resuelta antes
        logger.info("Finalizando y guardando resultados...")
        result_handler.finalize(
            config=container.resolve(dict), # Usar config del contenedor
            summary_data=summary,
            all_episodes_data=all_data,
            agent=agent_instance, # Pasar la instancia del agente resuelta
            results_folder=results_folder # Pasar la carpeta
        )
        logger.info("Resultados finalizados y guardados.")
    except Exception as e:
        logger.error(f"Error en finalización de resultados: {e}", exc_info=True)

    # --- [9] Generación de visualizaciones ---
    if vis_config and vis_config.get('enabled', False):
        logger.info("Iniciando generación de visualizaciones...")
        try:
            # run_visualizations necesita resolver PlotGenerator internamente o recibirlo
            # Pasamos el contenedor para que resuelva lo necesario
            run_visualizations(
                vis_config=vis_config,
                summary_data=summary,
                all_episodes_data=all_data,
                results_folder=results_folder,
                container=container # Pasar contenedor para resolver PlotGenerator
            )
            logger.info("Visualizaciones generadas.")
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {e}", exc_info=True)
    else:
        logger.info("Visualización deshabilitada en la configuración o archivo de vis no cargado.")

    # --- [10] Reporte de duración ---
    duration = time.time() - start_time
    logger.info(f"--- Ejecución Completa Finalizada en {duration:.2f}s ---")


if __name__ == "__main__":
    main()