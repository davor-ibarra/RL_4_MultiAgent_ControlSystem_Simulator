# main.py
import time
import logging
import os
import sys
from typing import Optional, Any, Dict, List, Tuple

# Importaciones del proyecto
from config_loader import load_and_validate_config
from logging_configurator import configure_file_logger
from di_container import build_container, Container
from result_handler import ResultHandler
# SimulationManager se resuelve desde el container
from simulation_manager import SimulationManager
# --- IMPORT RENOMBRADO ---
from visualization_runner import run_visualizations # Usa VisualizationGenerator internamente
# -------------------------
from interfaces.rl_agent import RLAgent # Interfaz para type hint

# --- Configuración Inicial de Logging (Consola) ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)-4s - %(name)-4s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)
root_logger.info("Logging Básico (Consola) Configurado. Nivel Base: DEBUG, Nivel Consola: INFO")

def main():
    """Función principal de ejecución de la simulación."""
    start_time = time.time()
    logger = logging.getLogger(__name__) # Logger específico de main
    logger.info("--- ============================ ---")
    logger.info("--- Iniciando Ejecución Principal ---")
    logger.info("--- ============================ ---")

    # --- 1. Cargar Configuración ---
    logger.info("Cargando y validando configuración...")
    config: Optional[Dict[str, Any]] = None; vis_config: Optional[Dict[str, Any]] = None; logging_config: Dict[str, Any] = {}
    try:
        config, vis_config, logging_config = load_and_validate_config('config.yaml')
        if config is None: logger.critical("Fallo crítico al cargar config principal. Abortando."); sys.exit(1)
        logger.info("Configuración principal cargada y validada.")
        if vis_config: logger.info("Configuración de visualización cargada.")
        else: logger.info("Visualización deshabilitada o config no encontrada/inválida.")
        logger.debug(f"Configuración de logging extraída: {logging_config}")
    except Exception as e: logger.critical(f"Error carga/validación config: {e}", exc_info=True); sys.exit(1)

    # --- 2. Configurar Entorno de Resultados y Logging a Fichero ---
    results_folder: Optional[str] = None
    try:
        results_folder_base_name = config.get('environment', {}).get('results_folder', 'results_history')
        results_folder = ResultHandler.setup_results_folder(results_folder_base_name)
        logger.info(f"Carpeta de resultados preparada: {results_folder}")
        # Configurar logging a fichero
        configure_file_logger(logging_config, results_folder)
    except Exception as e: logger.critical(f"Error setup resultados/logging: {e}", exc_info=True); sys.exit(1)

    # --- 3. Construir Contenedor DI y Registrar results_folder ---
    container: Optional[Container] = None
    try:
        logger.info("Construyendo contenedor DI...")
        container = build_container(config)
        if results_folder is None: raise RuntimeError("results_folder no creado.") # Safety check
        # Registrar la carpeta de resultados para que otros la resuelvan
        container.register(str, lambda c: results_folder, singleton=True) # type: ignore
        logger.info("Contenedor DI construido y 'results_folder' registrado.")
    except Exception as e: logger.critical(f"Error crítico construyendo contenedor DI: {e}", exc_info=True); sys.exit(1)

    # --- 4. Resolver Dependencias Iniciales y Guardar Metadata ---
    resolved_logger: Optional[logging.Logger] = None; result_handler_instance: Optional[ResultHandler] = None
    try:
        if container is None: raise RuntimeError("Contenedor DI no creado.")
        resolved_logger = container.resolve(logging.Logger) # type: ignore
        if resolved_logger is None: resolved_logger = logger # Fallback
        resolved_logger.info("--- Fase de Ejecución Iniciada (Usando Logger del Contenedor) ---")
        result_handler_instance = container.resolve(ResultHandler) # type: ignore
        if result_handler_instance is None: raise ValueError("Fallo resolviendo ResultHandler.")
        resolved_logger.debug("ResultHandler resuelto.")

        resolved_logger.info("Guardando metadata inicial...")
        metadata = {
             'execution_details': { 'framework_version': '5.2.0_DI', 'run_timestamp': time.strftime("%Y-%m-%dT%H:%M:%S%z"), 'results_folder': results_folder, 'python_version': sys.version, 'command_line': sys.argv },
             'config_used': { 'main_config': config, 'visualization_config': vis_config, 'logging_config': logging_config } }
        result_handler_instance.save_metadata(metadata, results_folder) # type: ignore
        resolved_logger.info("Metadata inicial guardada.")
    except ValueError as e: resolved_logger.critical(f"Error DI resolviendo componentes iniciales: {e}", exc_info=True); sys.exit(1)
    except Exception as e: resolved_logger.critical(f"Error inesperado inicialización post-contenedor: {e}", exc_info=True); sys.exit(1)

    # --- 5. Ejecución de Simulación ---
    all_episodes_data: List[Dict] = []; summary_data: List[Dict] = []; agent_instance_final: Optional[RLAgent] = None
    sim_manager_instance: Optional[SimulationManager] = None

    try:
        resolved_logger.info("Iniciando la ejecución de la simulación...")
        sim_manager_instance = container.resolve(SimulationManager)
        if sim_manager_instance is None: raise ValueError("Fallo crítico al resolver SimulationManager.")
        resolved_logger.debug("SimulationManager resuelto. Llamando a sim_manager_instance.run()...")

        all_episodes_data, summary_data = sim_manager_instance.run() # ¡Aquí se ejecuta la simulación!

        resolved_logger.debug("sim_manager_instance.run() completado.")
        resolved_logger.info(f"Simulación completada. {len(summary_data)} episodios procesados.")

        # Obtener instancia final del agente si se necesita guardar
        should_save_agent_state = config.get('simulation', {}).get('save_agent_state', False)
        if should_save_agent_state:
             resolved_logger.debug("Resolviendo RLAgent (final) desde contenedor...")
             agent_instance_final = container.resolve(RLAgent) # type: ignore
             if agent_instance_final is None: resolved_logger.warning("No se pudo resolver RLAgent final.")
             else: resolved_logger.debug("RLAgent (final) resuelto.")

    except ValueError as e: resolved_logger.critical(f"Error DI durante simulación: {e}", exc_info=True); sys.exit(1)
    except Exception as e: resolved_logger.critical(f"Error crítico durante simulación: {e}", exc_info=True); sys.exit(1)

    # --- 6. Finalización y Guardado de Resultados ---
    try:
        resolved_logger.info("Finalizando y guardando resultados...")
        if result_handler_instance is None or results_folder is None: raise RuntimeError("ResultHandler o results_folder son None antes de finalize.")
        resolved_logger.debug("Llamando a result_handler.finalize...")
        result_handler_instance.finalize(
            config=config, summary_data=summary_data,
            all_episodes_data=all_episodes_data, # Usualmente vacío si se guarda por batch
            agent=agent_instance_final, results_folder=results_folder )
        resolved_logger.debug("result_handler.finalize completado.")
        resolved_logger.info("Resultados finalizados y guardados.")
    except Exception as e: resolved_logger.error(f"Error durante finalización de resultados: {e}", exc_info=True)

    # --- 7. Generación de Visualizaciones ---
    visualization_enabled = vis_config is not None
    if visualization_enabled:
        resolved_logger.info("Iniciando generación de visualizaciones...")
        try:
             if container is None or results_folder is None: raise RuntimeError("Container o results_folder son None antes de visualización.")
             # Llamar a visualization_runner (que usará VisualizationGenerator)
             run_visualizations(
                 vis_config=vis_config, summary_data=summary_data,
                 all_episodes_data=all_episodes_data, results_folder=results_folder,
                 container=container )
             resolved_logger.info("Generación de visualizaciones completada.")
        except Exception as e: resolved_logger.error(f"Error durante generación de visualizaciones: {e}", exc_info=True)
    else:
        resolved_logger.info("Visualización deshabilitada. Omitiendo generación de gráficos.")

    # --- Finalización ---
    duration = time.time() - start_time
    resolved_logger.info("--- ============================ ---")
    resolved_logger.info(f"--- Ejecución Principal Finalizada en {duration:.2f}s ---")
    resolved_logger.info("--- ============================ ---")
    logging.shutdown() # Asegurar flush y cierre de handlers

if __name__ == "__main__":
    main()