import time
import logging
import os
import sys # Para sys.exit
from typing import Optional, Any, Dict, List, Tuple # Añadir tipos necesarios

# Importaciones del proyecto
from config_loader import load_and_validate_config
from logging_configurator import configure_file_logger
from di_container import build_container, Container # Importar contenedor y builder
from result_handler import ResultHandler
from simulation_manager import SimulationManager
# Importar SimulationManager aquí NO es necesario, se resuelve desde el container
from visualization_runner import run_visualizations
from interfaces.rl_agent import RLAgent # Interfaz para type hint

# --- Configuración Inicial de Logging (Consola) ---
# 1. Obtener el logger raíz
root_logger = logging.getLogger()
# 2. Establecer el nivel del logger RAÍZ al MÁS BAJO deseado (DEBUG)
#    Esto permite que los mensajes DEBUG y superiores pasen el filtro del logger.
root_logger.setLevel(logging.DEBUG)
# 3. ELIMINAR handlers existentes del logger raíz para evitar duplicados
#    Esto es crucial si el script se ejecuta varias veces o si otras libs añaden handlers.
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
# 4. Crear y configurar el HANDLER específico para la CONSOLA
console_handler = logging.StreamHandler(sys.stdout)
# Establecer el nivel MÍNIMO que se mostrará en la consola (INFO)
console_handler.setLevel(logging.INFO)
# Definir un formato para la consola
# Ajustar formato si es necesario (ej: reducir ancho de name)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)-4s - %(name)-4s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
# 5. Añadir SÓLO el handler de consola configurado al logger raíz
root_logger.addHandler(console_handler)
# 6. Mensaje de confirmación inicial (se mostrará en consola porque es INFO)
root_logger.info("Logging Básico (Consola) Configurado. Nivel Base: DEBUG, Nivel Consola: INFO")


def main():
    """Función principal de ejecución de la simulación."""
    start_time = time.time()
    # Usar root_logger o un logger específico del módulo main
    logger = logging.getLogger(__name__)
    logger.info("--- ============================ ---")
    logger.info("--- Iniciando Ejecución Principal ---")
    logger.info("--- ============================ ---")

    # --- 1. Cargar Configuración ---
    logger.info("Cargando y validando configuración...")
    config: Optional[Dict[str, Any]] = None
    vis_config: Optional[Dict[str, Any]] = None
    logging_config: Dict[str, Any] = {}
    try:
        config, vis_config, logging_config = load_and_validate_config('config.yaml')
        if config is None:
            # load_and_validate_config ya logueó el error crítico
            logger.critical("Fallo crítico al cargar la configuración principal. Abortando.")
            sys.exit(1) # Salir con código de error
        logger.info("Configuración principal cargada.")
        if vis_config:
             logger.info("Configuración de visualización cargada.")
        else:
             logger.info("Visualización deshabilitada o configuración no encontrada/inválida.")
        logger.debug(f"Configuración de logging extraída: {logging_config}")

    except Exception as e:
        logger.critical(f"Error inesperado durante la carga de configuración: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Configurar Entorno de Resultados y Logging a Fichero ---
    results_folder: Optional[str] = None
    try:
        # Crear carpeta de resultados usando el método estático
        # Acceder a la clave de forma segura
        results_folder_base_name = config.get('environment', {}).get('results_folder', 'results_history')
        results_folder = ResultHandler.setup_results_folder(results_folder_base_name)
        logger.info(f"Carpeta de resultados preparada: {results_folder}")

        # Configurar logging a fichero usando la carpeta creada y la config de logging
        # Ahora el nivel del FileHandler dependerá de logging_config['levels'].
        # Recibirá mensajes desde DEBUG porque el root logger está en DEBUG.
        configure_file_logger(logging_config, results_folder)

    except (OSError, ValueError, KeyError) as e:
        logger.critical(f"Error configurando carpeta de resultados o logging a fichero: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error inesperado configurando resultados/logging: {e}", exc_info=True)
        sys.exit(1)


    # --- 3. Construir Contenedor DI ---
    container: Optional[Container] = None
    try:
        logger.info("Construyendo contenedor de Inyección de Dependencias (DI)...")
        container = build_container(config)
        # Registrar la carpeta de resultados en el contenedor para que otros la puedan resolver
        # Asegurarse de que results_folder no sea None aquí (ya habría salido antes)
        if results_folder is None:
             raise RuntimeError("Error crítico: results_folder no se pudo crear pero la ejecución continuó.") # Safety check
        container.register(str, lambda c: results_folder, singleton=True) # type: ignore
        logger.info("Contenedor DI construido y 'results_folder' registrado.")
        # root_logger.debug(f"Tokens registrados en el contenedor: {container.get_registered_tokens()}")
    except Exception as e:
        logger.critical(f"Error crítico construyendo el contenedor DI: {e}", exc_info=True)
        sys.exit(1)


    # --- 4. Resolver Dependencias Iniciales y Guardar Metadata ---
    resolved_logger: Optional[logging.Logger] = None
    result_handler_instance: Optional[ResultHandler] = None
    try:
        if container is None: raise RuntimeError("Contenedor DI no creado.")
        resolved_logger = container.resolve(logging.Logger) # type: ignore
        if resolved_logger is None: resolved_logger = logger # Fallback
        resolved_logger.info("--- Fase de Ejecución Iniciada (Usando Logger del Contenedor) ---")
        result_handler_instance = container.resolve(ResultHandler) # type: ignore
        if result_handler_instance is None: raise ValueError("Fallo resolviendo ResultHandler.")
        resolved_logger.debug("ResultHandler resuelto.")

        # Guardar metadata inicial
        resolved_logger.info("Guardando metadata inicial...")
        metadata = {
             'execution_details': {
                 'framework_version': '5.1.0_DI', # Actualizar versión indicando DI
                 'run_timestamp': time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 'results_folder': results_folder,
                 'python_version': sys.version,
                 'command_line': sys.argv,
             },
             'config_used': { # Separar config para claridad
                  'main_config': config,
                  'visualization_config': vis_config, # Puede ser None
                  'logging_config': logging_config
             }
             # Podríamos añadir información del sistema operativo, librerías clave, etc.
        }
        result_handler_instance.save_metadata(metadata, results_folder) # type: ignore # results_folder is not None
        resolved_logger.info("Metadata inicial guardada.")

    except ValueError as e: # Error específico de resolución DI
         resolved_logger.critical(f"Error de Inyección de Dependencias resolviendo componentes iniciales: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
         # Usar root_logger por si resolved_logger falló
         resolved_logger.critical(f"Error inesperado durante la inicialización post-contenedor: {e}", exc_info=True)
         sys.exit(1)


    # --- 5. Inicialización y Ejecución de Simulación ---
    all_episodes_data: List[Dict] = []
    summary_data: List[Dict] = []
    agent_instance_final: Optional[RLAgent] = None
    sim_manager_instance: Any = None # No importar tipo concreto aquí

    try:
        resolved_logger.info("Iniciando la ejecución de la simulación...")
        resolved_logger.debug("Resolviendo SimulationManager desde contenedor...")
        # SimulationManager es transient (no singleton), se crea una nueva instancia
        # Usamos Any para el tipo aquí para evitar importación directa
        sim_manager_instance = container.resolve(SimulationManager) # Resolver por string falla, usar tipo concreto se ajusta a DI
        if sim_manager_instance is None:
             raise ValueError("Fallo crítico al resolver SimulationManager.")
        resolved_logger.debug("SimulationManager resuelto.")

        resolved_logger.debug("Llamando a sim_manager_instance.run()...")
        # El método run() internamente resolverá Environment, Agent, etc. usando el contenedor
        # Asegurar que sim_manager_instance tiene el método 'run' (aunque usemos Any)
        if not hasattr(sim_manager_instance, 'run'):
             raise AttributeError("La instancia resuelta de SimulationManager no tiene el método 'run'.")

        all_episodes_data, summary_data = sim_manager_instance.run()
        resolved_logger.debug("sim_manager_instance.run() completado.")
        resolved_logger.info(f"Simulación completada. {len(summary_data)} episodios procesados.")

        # Obtener instancia final del agente si se necesita guardar
        # Acceder a config de forma segura
        should_save_agent_state = config.get('simulation', {}).get('save_agent_state', False)
        if should_save_agent_state:
             resolved_logger.debug("Resolviendo RLAgent (final) desde contenedor...")
             # Agent es singleton, obtenemos la instancia existente usada en la simulación
             agent_instance_final = container.resolve(RLAgent) # type: ignore
             if agent_instance_final is None:
                  resolved_logger.warning("No se pudo resolver RLAgent final a pesar de save_agent_state=True.")
             else:
                  resolved_logger.debug("RLAgent (final) resuelto.")

    except ValueError as e:
         resolved_logger.critical(f"Error DI durante simulación: {e}", exc_info=True); sys.exit(1) # Simplificado
    except Exception as e:
        resolved_logger.critical(f"Error crítico durante simulación: {e}", exc_info=True); sys.exit(1) # Simplificado


    # --- 6. Finalización y Guardado de Resultados ---
    try:
        resolved_logger.info("Finalizando y guardando resultados...")
        # Asegurarse de que las instancias no sean None
        if result_handler_instance is None or results_folder is None:
             raise RuntimeError("Error crítico: ResultHandler o results_folder son None antes de finalize.")
        resolved_logger.debug("Llamando a result_handler.finalize...")
        result_handler_instance.finalize(
            config=config, # Pasar config original
            summary_data=summary_data,
            all_episodes_data=all_episodes_data, # Puede estar vacío si se guardó por batch
            agent=agent_instance_final, # Pasar agente final si se resolvió
            results_folder=results_folder
        )
        resolved_logger.debug("result_handler.finalize completado.")
        resolved_logger.info("Resultados finalizados y guardados.")
    except Exception as e:
        resolved_logger.error(f"Error durante la finalización de resultados: {e}", exc_info=True)
        # Continuar para intentar visualización si es posible


    # --- 7. Generación de Visualizaciones ---
    # Usar vis_config cargado al inicio
    visualization_enabled_in_config = vis_config is not None # Es None si está deshabilitado o falló la carga
    if visualization_enabled_in_config:
        resolved_logger.info("Iniciando generación de visualizaciones...")
        try:
             # Asegurar que los argumentos no sean None
            if container is None or results_folder is None:
                raise RuntimeError("Error crítico: Container o results_folder son None antes de visualización.")

            # Pasar el contenedor para que resuelva PlotGenerator internamente
            run_visualizations(
                vis_config=vis_config, # Pasar la config de visualización cargada
                summary_data=summary_data,
                all_episodes_data=all_episodes_data,
                results_folder=results_folder,
                container=container
            )
            resolved_logger.info("Generación de visualizaciones completada.")
        except Exception as e:
            resolved_logger.error(f"Error durante la generación de visualizaciones: {e}", exc_info=True)
    else:
        resolved_logger.info("Visualización deshabilitada. Omitiendo generación de gráficos.")


    # --- Finalización ---
    duration = time.time() - start_time
    resolved_logger.info("--- ============================ ---")
    resolved_logger.info(f"--- Ejecución Principal Finalizada en {duration:.2f}s ---")
    resolved_logger.info("--- ============================ ---")
    # Flush and close file handlers explicitly? Usually handled by Python exit.
    # logging.shutdown() # Ensures handlers are flushed/closed


if __name__ == "__main__":
    main()