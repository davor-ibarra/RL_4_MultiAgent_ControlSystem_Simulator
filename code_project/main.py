# main.py
import time
import logging
import os
import sys
from typing import Optional, Any, Dict, List, Tuple

# Importaciones del proyecto
# 1.1: Importaciones reorganizadas y algunas eliminadas si se resuelven vía DI
from utils.config.config_loader import load_and_validate_config
from utils.config.logging_configurator import configure_file_logger
from di_container import build_container, Container
from utils.data.result_handler import ResultHandler
# SimulationManager se resuelve desde el container
# from simulation_manager import SimulationManager # Se resuelve vía DI
# --- IMPORT RENOMBRADO ---
from visualization_manager import VisualizationManager # Se mantiene la llamada explícita
# -------------------------
from interfaces.rl_agent import RLAgent # Interfaz para type hint

# --- Configuración Inicial de Logging (Consola) ---
# 1.2: Se mantiene la configuración básica de consola inicial
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Nivel base permisivo
# Limpiar handlers existentes para evitar duplicados si se re-ejecuta
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
console_handler = logging.StreamHandler(sys.stdout)
# Nivel INFO para la consola, DEBUG irá al archivo si está configurado
console_handler.setLevel(logging.INFO)
# Formato más limpio y estándar
console_formatter = logging.Formatter('%(asctime)s - %(levelname)-4s - %(name)-8s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)
# Usar el logger raíz para el mensaje inicial, antes de obtener el logger específico
root_logger.info("Logging Básico (Consola) Configurado. Nivel Base: DEBUG, Nivel Consola: INFO")

# Ajustar nivel de logging para librerías verbosas (Matplotlib, PIL)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
root_logger.debug("Nivel de logging para Matplotlib y PIL ajustado a WARNING.")

def main():
    """Función principal de ejecución de la simulación."""
    start_time = time.time()
    # 1.3: Obtener logger específico para main
    logger = logging.getLogger(__name__) # Logger específico de main
    logger.info("--- ============================ ---")
    logger.info("--- Iniciando Ejecución Principal ---")
    logger.info("--- ============================ ---")

    config: Optional[Dict[str, Any]] = None
    vis_config: Optional[Dict[str, Any]] = None
    logging_config: Dict[str, Any] = {}
    results_folder: Optional[str] = None
    container: Optional[Container] = None
    resolved_logger: Optional[logging.Logger] = None
    result_handler_instance: Optional[ResultHandler] = None
    agent_instance_final: Optional[RLAgent] = None
    all_episodes_data: List[Dict] = []
    summary_data: List[Dict] = []

    try:
        # --- 1. Cargar Configuración ---
        logger.info("Paso 1: Cargando y validando configuración...")
        config, vis_config, logging_config = load_and_validate_config('config.yaml')
        # 1.4: Fail-Fast si la config principal falla
        if config is None:
            # Usar root_logger porque el específico podría no estar configurado aún
            root_logger.critical("Fallo crítico al cargar config principal. Abortando.")
            sys.exit(1)
        logger.info("Configuración principal cargada y validada.")
        if vis_config: logger.info("Configuración de visualización cargada.")
        else: logger.info("Visualización deshabilitada o config no encontrada/inválida.")
        #logger.debug(f"Configuración de logging extraída: {logging_config}")

        # --- 2. Configurar Entorno de Resultados y Logging a Fichero ---
        # 2.1: Crear carpeta de resultados ANTES de configurar el logger de fichero
        logger.info("Paso 2: Preparando carpeta de resultados...")
        results_folder_base_name = config.get('environment', {}).get('results_folder', 'results_history')
        # 2.2: Usar el método estático de ResultHandler
        results_folder = ResultHandler.setup_results_folder(results_folder_base_name)
        logger.info(f"Carpeta de resultados preparada: {results_folder}")

        # 2.3: Configurar logging a fichero usando la carpeta creada
        logger.info("Configurando logging a fichero...")
        configure_file_logger(logging_config, results_folder)
        # A partir de aquí, los logs (según nivel) irán también al fichero

        # --- 3. Construir Contenedor DI y Registrar results_folder ---
        logger.info("Paso 3: Construyendo contenedor DI...")
        container = build_container(config, vis_config)
        # 3.1: Registrar results_folder como una instancia de string resoluble
        container.register(str, lambda c: results_folder, singleton=True) # type: ignore[misc]
        logger.info("Contenedor DI construido y 'results_folder' registrado.")

        # --- 4. Resolver Dependencias Iniciales y Guardar Metadata ---
        logger.info("Paso 4: Resolviendo dependencias iniciales y guardando metadata...")
        # 4.1: Resolver logger y result_handler desde el contenedor
        resolved_logger = container.resolve(logging.Logger)
        result_handler_instance = container.resolve(ResultHandler)
        if resolved_logger is None or result_handler_instance is None:
            raise RuntimeError("Fallo crítico al resolver Logger o ResultHandler desde DI.")
        resolved_logger.info("--- Fase de Ejecución Iniciada (Usando Logger del Contenedor) ---")
        #resolved_logger.debug("Logger y ResultHandler resueltos.")

        # 4.2: Guardar metadata usando result_handler y results_folder
        metadata = {
            'execution_details': {
                'framework_version': '5.3.0_DI', # Versión actualizada
                'run_timestamp': time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                'results_folder': results_folder,
                'python_version': sys.version.split()[0], # Versión más corta
                'command_line': sys.argv
            },
            'config_used': {
                'main_config': config,
                'visualization_config': vis_config if vis_config else "Disabled", # Más claro si está desactivado
                'logging_config': logging_config
            }
        }
        result_handler_instance.save_metadata(metadata, results_folder)
        resolved_logger.info("Metadata inicial guardada.")

        # --- 5. Ejecución de Simulación ---
        resolved_logger.info("Paso 5: Ejecutando la simulación...")
        # 5.1: Resolver SimulationManager
        sim_manager_instance = container.resolve('simulation_manager.SimulationManager') # Usar token explícito si hay ambigüedad
        if sim_manager_instance is None:
            raise RuntimeError("Fallo crítico al resolver SimulationManager.")
        #resolved_logger.debug("SimulationManager resuelto. Iniciando sim_manager_instance.run()...")

        # 5.2: Ejecutar la simulación
        all_episodes_data, summary_data = sim_manager_instance.run()

        resolved_logger.info(f"Simulación completada. {len(summary_data)} episodios procesados.")

        # 5.3: Obtener instancia final del agente (si se guarda) post-simulación
        should_save_agent_state = config.get('simulation', {}).get('save_agent_state', False)
        if should_save_agent_state:
            resolved_logger.debug("Resolviendo instancia final de RLAgent...")
            # Usar la interfaz como token
            agent_instance_final = container.resolve(RLAgent)
            if agent_instance_final is None:
                resolved_logger.warning("Guardado de estado del agente habilitado, pero no se pudo resolver RLAgent final.")
            else:
                resolved_logger.debug(f"Instancia final de {type(agent_instance_final).__name__} resuelta.")


    # 1.5: Simplificar bloque except principal a errores críticos de inicialización/ejecución
    except (RuntimeError, ValueError, ImportError, TypeError, KeyError) as e:
        # Usar root_logger si resolved_logger no está disponible
        log_func = resolved_logger.critical if resolved_logger else root_logger.critical
        log_func(f"Error CRÍTICO durante la inicialización o ejecución: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e: # Captura cualquier otra excepción inesperada
        log_func = resolved_logger.critical if resolved_logger else root_logger.critical
        log_func(f"Error INESPERADO y no controlado: {e}", exc_info=True)
        sys.exit(1)

    # --- 6. Finalización y Guardado de Resultados ---
    # Este bloque se ejecuta solo si la simulación (Paso 5) fue exitosa
    if result_handler_instance and results_folder:
        resolved_logger.info("Paso 6: Finalizando y guardando resultados...")
        try:
            # 6.1: Llamada explícita a finalize
            result_handler_instance.finalize(
                config=config, # Pasar config completa por si finalize lo necesita
                vis_config=vis_config,
                summary_data=summary_data,
                all_episodes_data=all_episodes_data, # Suele estar vacío
                agent=agent_instance_final,
                results_folder=results_folder
            )
            resolved_logger.info("Resultados finalizados y guardados.")
        except Exception as e:
            resolved_logger.error(f"Error durante la finalización de resultados: {e}", exc_info=True)
            # No necesariamente salir, pero sí loguear el error.

    # --- 7. Generación de Visualizaciones ---
    # Este bloque se ejecuta solo si la simulación (Paso 5) fue exitosa
    # 7.1: Comprobar si la visualización está habilitada
    visualization_enabled = config.get('visualization', {}).get('enabled', False)
    vis_config_dict = config.get('visualization', {}) # Obtener dict para logs

    if visualization_enabled and container and results_folder:
        resolved_logger.info("Paso 7: Generando visualizaciones...")
        try:
             # Resolver y ejecutar el VisualizationManager
             vis_manager = container.resolve(VisualizationManager)
             if vis_manager:
                  vis_manager.run() # El manager se encarga de todo
                  resolved_logger.info("Generación de visualizaciones completada (llamada a VisualizationManager.run).")
             else:
                  resolved_logger.error("No se pudo resolver VisualizationManager desde DI.")

        except Exception as e:
             resolved_logger.error(f"Error durante la generación de visualizaciones: {e}", exc_info=True)
    elif results_folder:
        resolved_logger.info(f"Paso 7: Visualización deshabilitada en config (enabled={visualization_enabled}, config found={vis_config_dict is not None}). Omitiendo generación de gráficos.")

    # --- Finalización ---
    duration = time.time() - start_time
    final_logger = resolved_logger if resolved_logger else logger # Usar el logger disponible
    final_logger.info("--- ============================ ---")
    final_logger.info(f"--- Ejecución Principal Finalizada en {duration:.2f}s ---")
    final_logger.info(f"Resultados en: {results_folder}")
    final_logger.info("--- ============================ ---")
    logging.shutdown() # Asegurar flush y cierre de handlers

if __name__ == "__main__":
    main()