# main.py
import time
import logging
import os
import gc
import sys
from typing import Optional, Any, Dict, List, Tuple

from utils.config.config_loader import load_and_validate_config
from utils.config.logging_configurator import configure_file_logger
from di_container import build_container, Container
from utils.data.result_handler import ResultHandler
from visualization_manager import VisualizationManager # Llamada explícita
from interfaces.rl_agent import RLAgent # Interfaz para type hint

# --- Configuración Inicial de Logging (Consola) ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler) # Limpiar handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)-4s - %(name)-8s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)
root_logger.info("Basic Console Logging Configured. Base: DEBUG, Console: INFO")

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
root_logger.debug("Logging level for Matplotlib & PIL set to WARNING.")


def main():
    start_time = time.time()
    logger = logging.getLogger(__name__) # Logger específico de main
    logger.info("--- ============================ ---")
    logger.info("--- Starting Main Execution ---")
    logger.info("--- ============================ ---")

    # Inicializar variables
    config: Optional[Dict[str, Any]] = None
    vis_config: Optional[Dict[str, Any]] = None
    logging_config_from_file: Dict[str, Any] = {} # Renombrar para claridad
    results_folder: Optional[str] = None
    container_instance: Optional[Container] = None # Renombrar para claridad
    # Usar resolved_logger para el logger del contenedor, logger para el logger de main.py
    resolved_logger: Optional[logging.Logger] = None
    result_handler_instance: Optional[ResultHandler] = None
    agent_instance_final: Optional[RLAgent] = None
    all_episodes_data: List[Dict] = [] # Normalmente vacío si se guarda por batch
    summary_data: List[Dict] = []

    try:
        # --- 1. Cargar Configuración ---
        logger.info("[MAIN] Step 1: Loading and validating configuration...")
        config, vis_config, logging_config_from_file = load_and_validate_config('config.yaml')
        if config is None:
            root_logger.critical("[MAIN] CRITICAL FAILURE: Main config loading failed. Aborting.")
            sys.exit(1)
        logger.info("[MAIN] Main configuration loaded and validated.")
        if vis_config: logger.info("[MAIN] Visualization configuration loaded.")
        else: logger.info("[MAIN] Visualization disabled or config not found/invalid.")
        #logger.debug(f"Logging configuration extracted: {logging_config_from_file}")

        # --- 2. Configurar Entorno de Resultados y Logging a Fichero ---
        logger.info("[MAIN] Step 2: Preparing results folder and file logging...")
        # Leer 'results_folder' de la nueva ubicación
        results_folder_base_name = config.get('environment', {}).get('results_folder', 'results_history_default')
        results_folder = ResultHandler.setup_results_folder(results_folder_base_name)
        logger.info(f"[MAIN] Results folder prepared: {results_folder}")

        # Configurar logging a fichero usando la config extraída y la carpeta creada
        configure_file_logger(logging_config_from_file, results_folder)
        # A partir de aquí, logs (según nivel) irán también al fichero

        # --- 3. Construir Contenedor DI ---
        logger.info("[MAIN] Step 3: Building DI container...")
        container_instance = build_container(config, vis_config)
        # Registrar 'results_folder' como una instancia de string resoluble
        container_instance.register(str, lambda c: results_folder, singleton=True) # type: ignore[misc]
        logger.info("[MAIN] DI container built and 'results_folder' registered.")

        # --- 4. Resolver Dependencias Iniciales y Guardar Metadata ---
        logger.info("[MAIN] Step 4: Resolving initial dependencies and saving metadata...")
        resolved_logger = container_instance.resolve(logging.Logger) # Logger del contenedor
        result_handler_instance = container_instance.resolve(ResultHandler)

        if resolved_logger is None or result_handler_instance is None: # resolved_logger no debería ser None
            raise RuntimeError("[MAIN] Critical failure resolving Logger or ResultHandler from DI.")
        resolved_logger.info("[MAIN] --- Execution Phase Started (Using Container Logger) ---")

        metadata = {
            'execution_details': {
                'framework_version': '6.0.0_Refactor_to_fast_components', # Versión actualizada
                'run_timestamp': time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                'results_folder': results_folder,
                'python_version': sys.version.split()[0],
                'command_line': sys.argv
            },
            'config_used': {
                'main_config': config,
                'visualization_config': vis_config if vis_config else "Disabled/NotLoaded",
                'logging_config': logging_config_from_file
            }
        }
        result_handler_instance.save_metadata(metadata, results_folder)
        resolved_logger.info("[MAIN] Initial metadata saved.")

        # --- 5. Ejecución de Simulación ---
        resolved_logger.info("[MAIN] Step 5: Executing simulation...")
        sim_manager_instance = container_instance.resolve('simulation_manager.SimulationManager')
        if sim_manager_instance is None:
            raise RuntimeError("[MAIN] Critical failure resolving SimulationManager.")

        all_episodes_data, summary_data = sim_manager_instance.run()
        resolved_logger.info(f"[MAIN] Simulation completed. {len(summary_data)} episodes processed.")

        # Obtener instancia final del agente si se guarda
        # Leer 'save_agent_state' de la nueva ubicación
        should_save_agent_state = config.get('environment', {}).get('save_agent_state', False)
        if should_save_agent_state:
            resolved_logger.debug("[MAIN] Resolving final RLAgent instance for saving...")
            agent_instance_final = container_instance.resolve(RLAgent) # Usar interfaz como token
            if agent_instance_final is None:
                resolved_logger.warning("[MAIN] Agent state saving enabled, but final RLAgent instance could not be resolved.")
            else:
                resolved_logger.debug(f"[MAIN] Final RLAgent instance ({type(agent_instance_final).__name__}) resolved.")

    except (RuntimeError, ValueError, ImportError, TypeError, KeyError) as e:
        log_func = resolved_logger.critical if resolved_logger else root_logger.critical
        log_func(f"[MAIN] CRITICAL error during initialization or execution: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        log_func = resolved_logger.critical if resolved_logger else root_logger.critical
        log_func(f"[MAIN] UNEXPECTED and unhandled error: {e}", exc_info=True)
        sys.exit(1)

    # --- 6. Finalización y Guardado de Resultados ---
    # Este bloque se ejecuta solo si la simulación (Paso 5) fue exitosa
    if result_handler_instance and results_folder and config and resolved_logger: # Asegurar que config y resolved_logger existan
        resolved_logger.info("[MAIN] Step 6: Finalizing and saving results...")
        try:
            result_handler_instance.finalize(
                config=config,
                vis_config=vis_config, # Pasar vis_config aunque no lo use directamente ResultHandler
                summary_data=summary_data,
                all_episodes_data=all_episodes_data,
                agent=agent_instance_final,
                results_folder=results_folder
            )
            resolved_logger.info("[MAIN] Results finalized and saved by ResultHandler.")
        except Exception as e:
            resolved_logger.error(f"[MAIN] Error during results finalization: {e}", exc_info=True)
            # No necesariamente salir, pero sí loguear el error.

    # --- 7. Generación de Visualizaciones ---
    # Este bloque se ejecuta solo si la simulación (Paso 5) fue exitosa
    # Leer 'visualization.enabled' de la nueva ubicación
    visualization_enabled = config.get('visualization', {}).get('enabled', False) if config else False
    vis_config_data_for_log = config.get('visualization') if config else None # Para log

    if visualization_enabled and container_instance and results_folder and resolved_logger:
        resolved_logger.info("[MAIN] Step 7: Generating visualizations...")
        try:
             vis_manager = container_instance.resolve(VisualizationManager)
             if vis_manager:
                  vis_manager.run()
                  resolved_logger.info("[MAIN] Visualization generation via VisualizationManager.run() completed.")
             else:
                  resolved_logger.error("[MAIN] Could not resolve VisualizationManager from DI. Visualizations skipped.")
        except Exception as e:
             resolved_logger.error(f"[MAIN] Error during visualization generation: {e}", exc_info=True)
    elif results_folder and resolved_logger : # Checkear resolved_logger para evitar error si falló antes
        resolved_logger.info(f"[MAIN] Step 7: Visualization disabled (enabled={visualization_enabled}, config present={vis_config_data_for_log is not None}). Skipping.")
    elif logger: # Fallback al logger de main si resolved_logger no está
        logger.info(f"[MAIN] Step 7: Visualization disabled or prerequisites missing. Skipping. (Enabled={visualization_enabled})")


    # --- Finalización ---
    duration = time.time() - start_time
    final_logger_to_use = resolved_logger if resolved_logger else logger
    final_logger_to_use.info("--- =============================== ---")
    final_logger_to_use.info(f"--- Main Execution Finished in {duration:.2f}s ---")
    if results_folder:
        final_logger_to_use.info(f"Results located at: {results_folder}")
    else:
        final_logger_to_use.warning("[MAIN] Results folder path not available (likely due to early error).")
    final_logger_to_use.info("--- =============================== ---")
    # Limpieza explícita de objetos grandes o potencialmente problemáticos
    try:
        del config, vis_config, container_instance, result_handler_instance, agent_instance_final
        del all_episodes_data, summary_data
        if 'sim_manager_instance' in locals(): del sim_manager_instance
        if 'vis_manager' in locals(): del vis_manager
        gc.collect()
        final_logger_to_use.info("--- Main objects cleanup attempted ---")
    except Exception as e_cleanup:
        final_logger_to_use.warning(f"--- Error during main objects cleanup: {e_cleanup} ---")

    logging.shutdown()

if __name__ == "__main__":
    main()