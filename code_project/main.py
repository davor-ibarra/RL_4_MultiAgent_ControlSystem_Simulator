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
# Nombres de variables aquí son locales y descriptivos, se mantienen.
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Base level for root logger
for handler_instance in root_logger.handlers[:]: root_logger.removeHandler(handler_instance) # 'handler_instance'
console_stream_handler = logging.StreamHandler(sys.stdout) # 'console_stream_handler'
console_stream_handler.setLevel(logging.INFO) # Console output level
console_log_formatter = logging.Formatter('%(asctime)s - %(levelname)-4s - %(name)-8s - %(message)s', # 'console_log_formatter'
                                      datefmt='%Y-%m-%d %H:%M:%S')
console_stream_handler.setFormatter(console_log_formatter)
root_logger.addHandler(console_stream_handler)
root_logger.info("Basic Console Logging Configured. Root Level: DEBUG, Console Handler Level: INFO")

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
root_logger.debug("Logging level for Matplotlib & PIL set to WARNING for console.")


def main():
    execution_start_time = time.time() # 'execution_start_time'
    main_module_logger = logging.getLogger(__name__) # 'main_module_logger', Logger específico de main
    main_module_logger.info("--- ============================ ---")
    main_module_logger.info("--- Starting Main Execution ---")
    main_module_logger.info("--- ============================ ---")

    # Inicializar variables
    main_configuration: Optional[Dict[str, Any]] = None
    visualization_configuration: Optional[Dict[str, Any]] = None
    file_logging_configuration: Dict[str, Any] = {}
    # Nueva variable para las directivas de datos procesadas
    processed_data_directives_config: Optional[Dict[str, Any]] = None
    output_directory_path: Optional[str] = None
    di_container: Optional[Container] = None
    
    # Usar container_resolved_logger para el logger del contenedor, main_module_logger para el logger de main.py
    container_resolved_logger: Optional[logging.Logger] = None
    result_handler_service: Optional[ResultHandler] = None
    final_agent_instance: Optional[RLAgent] = None
    simulation_summary_data: List[Dict] = []

    try:
        # --- 1. Cargar Configuración ---
        main_module_logger.info("[MAIN] Step 1: Loading and validating configuration...")
        main_configuration, visualization_configuration, file_logging_configuration, processed_data_directives_config = load_and_validate_config('super_config.yaml')
        if main_configuration is None:
            root_logger.critical("[MAIN] CRITICAL FAILURE: Main config loading failed. Aborting.")
            sys.exit(1)
        main_module_logger.info("[MAIN] Main configuration loaded and validated.")
        if visualization_configuration: main_module_logger.info("[MAIN] Visualization configuration loaded.")
        else: main_module_logger.info("[MAIN] Visualization disabled or config not found/invalid.")
        #main_module_logger.debug(f"Logging configuration extracted: {file_logging_configuration}")
        if processed_data_directives_config:
            main_module_logger.info("[MAIN] Data handling directives loaded and validated.")
            # main_module_logger.debug(f"[MAIN] Data directives: {processed_data_directives_config}")
        else: # Esto solo ocurriría si load_and_validate_config tiene una lógica para devolver None aquí pero main_config OK (no debería)
            main_module_logger.warning("[MAIN] Data handling directives not loaded (or config issue). Using defaults if applicable by components.")

        # --- 2. Configurar Entorno de Resultados y Logging a Fichero ---
        main_module_logger.info("[MAIN] Step 2: Preparing output directory and file logging...")
        # Leer 'output_root' (nuevo nombre) de la config
        output_root_base_name = main_configuration.get('data_handling', {}).get('output_root', 'results_default_output') # 'output_root_base_name', 'output_root'
        output_directory_path = ResultHandler.setup_results_folder(output_root_base_name) # 'output_directory_path'
        main_module_logger.info(f"[MAIN] Output directory prepared: {output_directory_path}")

        # Configurar logging a fichero usando la config extraída y la carpeta creada
        configure_file_logger(file_logging_configuration, output_directory_path)
        # A partir de aquí, logs (según nivel) irán también al fichero

        # --- 3. Construir Contenedor DI ---
        main_module_logger.info("[MAIN] Step 3: Building DI container...")
        di_container = build_container(
            main_config=main_configuration, # Renombrar para claridad
            vis_config=visualization_configuration, # Renombrar
            processed_data_directives=processed_data_directives_config, # Renombrar
            output_dir=output_directory_path # <<< NUEVO ARGUMENTO
        )
        main_module_logger.info(f"[MAIN] DI container built and 'output_directory_path' ({output_directory_path}) registered as resolvable string.")

        # --- 4. Resolver Dependencias Iniciales y Guardar Metadata ---
        main_module_logger.info("[MAIN] Step 4: Resolving initial dependencies and saving metadata...")
        container_resolved_logger = di_container.resolve(logging.Logger)
        result_handler_service = di_container.resolve(ResultHandler)

        if container_resolved_logger is None or result_handler_service is None:
            raise RuntimeError("[MAIN] Critical failure resolving Logger or ResultHandler from DI.")
        container_resolved_logger.info("[MAIN] --- Execution Phase Started (Using Container Logger) ---")

        run_metadata = { # 'run_metadata'
            'execution_details': {
                'framework_version': '8.0.0_Refactor_Scaled_Final', # Versión actualizada
                'run_timestamp': time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                'output_directory': output_directory_path, # 'output_directory'
                'python_version': sys.version.split()[0],
                'command_line_arguments': sys.argv # 'command_line_arguments'
            },
            'config_used': {
                'main_config': main_configuration,
                'visualization_config': visualization_configuration if visualization_configuration else "Disabled/NotLoaded",
                'logging_config': file_logging_configuration,
                'data_handling_directives': processed_data_directives_config if processed_data_directives_config else "NotLoaded/Defaults"
            }
        }
        result_handler_service.save_metadata(run_metadata, output_directory_path)
        container_resolved_logger.info("[MAIN] Initial run metadata saved.")

        # --- 5. Ejecución de Simulación ---
        container_resolved_logger.info("[MAIN] Step 5: Executing simulation...")
        # El token 'simulation_manager.SimulationManager' es un string, no una clase.
        simulation_manager_instance = di_container.resolve('simulation_manager.SimulationManager')
        if simulation_manager_instance is None:
            raise RuntimeError("[MAIN] Critical failure resolving SimulationManager.")

        # SimulationManager.run() ahora devuelve ([], summary_data)
        # all_episodes_detailed_data_main será una lista vacía.
        all_episodes_detailed_data_main, simulation_summary_data = simulation_manager_instance.run()
        container_resolved_logger.info(f"[MAIN] Simulation completed. {len(simulation_summary_data)} episodes processed.")

        # Obtener instancia final del agente si se guarda (usar 'data_handling' para 'save_agent_state')
        should_save_final_agent_state_main = main_configuration.get('data_handling', {}).get('save_agent_state', False)
        if should_save_final_agent_state_main:
            container_resolved_logger.debug("[MAIN] Resolving final RLAgent instance for saving...")
            final_agent_instance = di_container.resolve(RLAgent) # Usar interfaz como token
            if final_agent_instance is None:
                container_resolved_logger.warning("[MAIN] Agent state saving enabled, but final RLAgent instance could not be resolved.")
            else:
                container_resolved_logger.debug(f"[MAIN] Final RLAgent instance ({type(final_agent_instance).__name__}) resolved.")

    except (RuntimeError, ValueError, ImportError, TypeError, KeyError) as e_runtime_main:
        active_logger = container_resolved_logger if container_resolved_logger else root_logger
        active_logger.critical(f"[MAIN] CRITICAL error during initialization or execution: {e_runtime_main}", exc_info=True)
        sys.exit(1)
    except Exception as e_unhandled_main:
        active_logger_unhandled = container_resolved_logger if container_resolved_logger else root_logger
        active_logger_unhandled.critical(f"[MAIN] UNEXPECTED and unhandled error: {e_unhandled_main}", exc_info=True)
        sys.exit(1)

    # --- 6. Finalización y Guardado de Resultados ---
    # Este bloque se ejecuta solo si la simulación (Paso 5) fue exitosa
    if result_handler_service and output_directory_path and main_configuration and container_resolved_logger and processed_data_directives_config:
        container_resolved_logger.info("[MAIN] Step 6: Finalizing and saving results...")
        try:
            result_handler_service.finalize(
                main_config=main_configuration,
                summary_data=simulation_summary_data,
                agent=final_agent_instance,
                output_dir_finalize=output_directory_path
            )
            container_resolved_logger.info("[MAIN] Results finalized and saved by ResultHandler.")
        except Exception as e_finalize_main:
            container_resolved_logger.error(f"[MAIN] Error during results finalization: {e_finalize_main}", exc_info=True)
            # No necesariamente salir, pero sí loguear el error.

    # --- 7. Generación de Visualizaciones ---
    # Este bloque se ejecuta solo si la simulación (Paso 5) fue exitosa
    # Leer 'visualization.enabled' de la nueva ubicación
    visualization_feature_enabled_main = main_configuration.get('visualization', {}).get('enabled', False) if main_configuration else False
    visualization_config_for_log_main = main_configuration.get('visualization') if main_configuration else None

    if visualization_feature_enabled_main and di_container and output_directory_path and container_resolved_logger:
        container_resolved_logger.info("[MAIN] Step 7: Generating visualizations...")
        try:
             visualization_manager_service = di_container.resolve(VisualizationManager) # 'visualization_manager_service'
             if visualization_manager_service:
                  visualization_manager_service.run()
                  container_resolved_logger.info("[MAIN] Visualization generation via VisualizationManager.run() completed.")
             else:
                  container_resolved_logger.error("[MAIN] Could not resolve VisualizationManager from DI. Visualizations skipped.")
        except Exception as e_visualize: # 'e_visualize'
             container_resolved_logger.error(f"[MAIN] Error during visualization generation: {e_visualize}", exc_info=True)
    elif output_directory_path and container_resolved_logger :
        container_resolved_logger.info(f"[MAIN] Step 7: Visualization disabled (enabled={visualization_feature_enabled_main}, config present={visualization_config_for_log_main is not None}). Skipping.")
    elif main_module_logger:
        main_module_logger.info(f"[MAIN] Step 7: Visualization disabled or prerequisites missing. Skipping. (Enabled={visualization_feature_enabled_main})")


    # --- Finalización ---
    total_execution_duration_sec = time.time() - execution_start_time # 'total_execution_duration_sec'
    final_logger = container_resolved_logger if container_resolved_logger else main_module_logger # 'final_logger'
    final_logger.info("--- =============================== ---")
    final_logger.info(f"--- Main Execution Finished in {total_execution_duration_sec:.2f}s ---")
    if output_directory_path:
        final_logger.info(f"Output data located at: {output_directory_path}")
    else:
        final_logger.warning("[MAIN] Output directory path not available (likely due to early error).")
    final_logger.info("--- =============================== ---")
    
    # Limpieza explícita de objetos grandes o potencialmente problemáticos
    try:
        del main_configuration, visualization_configuration, di_container, result_handler_service, final_agent_instance
        del all_episodes_detailed_data_main, simulation_summary_data
        # Usar locals() para chequear existencia de variables de forma segura antes de 'del'
        if 'simulation_manager_instance' in locals(): del simulation_manager_instance
        if 'visualization_manager_service' in locals(): del visualization_manager_service # Nombre actualizado
        gc.collect()
        final_logger.info("--- Main objects cleanup attempted ---")
    except Exception as e_cleanup_main: # 'e_cleanup_main'
        final_logger.warning(f"--- Error during main objects cleanup: {e_cleanup_main} ---")

    logging.shutdown()

if __name__ == "__main__":
    main()