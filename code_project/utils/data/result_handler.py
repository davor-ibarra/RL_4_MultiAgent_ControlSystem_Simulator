import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

# (2.1) Mantener import de NumpyEncoder
from utils.data.numpy_encoder import NumpyEncoder
# Importar RLAgent sólo para type hints
if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
# (2.2) ELIMINAR import de HeatmapGenerator
# from utils.data.heatmap_generator import HeatmapGenerator

# (2.3) Usar logger específico del módulo (obtenido vía DI en __init__)
# logger = logging.getLogger(__name__) # Quitar logger a nivel de módulo

class ResultHandler:
    """
    Servicio centralizado para gestionar el guardado de artefactos de resultados
    (metadatos, datos de episodios, estado del agente, resumen).
    Recibe dependencias (logger) vía DI.
    La carpeta de resultados se pasa como argumento a los métodos de guardado.
    NO es responsable de generar datos para visualización (e.g., heatmaps).
    """
    # (2.4) ELIMINAR heatmap_generator de __init__
    def __init__(self, logger: logging.Logger):
        """
        Inicializa ResultHandler con dependencias inyectadas.

        Args:
            logger: Instancia del logger configurado.
        """
        self._logger = logger
        if not isinstance(logger, logging.Logger):
            logging.getLogger().critical("ResultHandler: Se requiere una instancia de logging.Logger válida.")
            raise TypeError("Se requiere una instancia de logging.Logger válida.")
        # (2.5) ELIMINAR atributo heatmap_generator
        # self._heatmap_generator = heatmap_generator
        self._last_agent_state_json_path: Optional[str] = None
        self._logger.info("ResultHandler instance created.")

    @staticmethod
    def setup_results_folder(base_results_folder: str = 'results_history') -> str:
        """
        Crea la carpeta de resultados única para esta ejecución basada en timestamp.
        Este método es estático y se llama ANTES de instanciar ResultHandler.

        Args:
            base_results_folder (str): Nombre de la carpeta base (relativa al script principal).

        Returns:
            str: La ruta absoluta a la carpeta de resultados creada.

        Raises:
            OSError: Si falla la creación de la carpeta.
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M') # (2.6) Añadir segundos para mayor unicidad
        results_folder = "FOLDER_PATH_ERROR" # Default en caso de error
        try:
             # (2.7) Determinar ruta base de forma más robusta
             try:
                 # Intentar basarse en el script que se está ejecutando (main.py)
                 script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
             except Exception:
                 # Fallback al directorio de trabajo actual
                 script_dir = os.getcwd()
                 logging.warning(f"[ResultHandler:Setup] No se pudo determinar el directorio del script, usando CWD: {script_dir}")

             # Construir ruta absoluta a la carpeta base y la carpeta de ejecución
             base_folder_path = os.path.join(script_dir, base_results_folder)
             results_folder = os.path.join(base_folder_path, timestamp)
             os.makedirs(results_folder, exist_ok=True)
             logging.info(f"[ResultHandler:Setup] Results folder created/verified: {results_folder}")
             return results_folder
        except OSError as e:
             logging.error(f"[ResultHandler:Setup] Failed to create results folder {results_folder}: {e}")
             raise # Relanzar para detener ejecución (Fail-Fast)
        except Exception as e_setup:
            logging.error(f"[ResultHandler:Setup] Unexpected error setting up results folder: {e_setup}", exc_info=True)
            raise RuntimeError(f"Unexpected error setting up results folder: {e_setup}") from e_setup


    # --- Métodos de Guardado (requieren 'results_folder') ---

    def save_episode_batch(self, batch_data: List[Dict], results_folder: str, last_episode: int):
        """Guarda un lote de datos detallados de episodios en un archivo JSON."""
        if not os.path.isdir(results_folder):
            self._logger.error(f"La carpeta de resultados '{results_folder}' no existe al guardar batch. Omitiendo.")
            return
        if not batch_data:
            self._logger.warning("Intento de guardar lote de episodios vacío. Omitiendo.")
            return

        try:
            num_episodes_in_batch = len(batch_data)
            # Determinar primer episodio con cuidado (puede ser 0)
            first_episode_in_batch = max(0, last_episode - num_episodes_in_batch + 1)

            filename_range = f"{first_episode_in_batch}_to_{last_episode}"
            filename = f"simulation_data_ep_{filename_range}.json"
            path = os.path.join(results_folder, filename)
            self._logger.info(f"Guardando lote episodios {filename_range} -> {filename}")

            with open(path, 'w', encoding='utf-8') as f:
                # Usar NumpyEncoder importado
                json.dump(batch_data, f, cls=NumpyEncoder, indent=2)
            self._logger.info(f"Lote episodios guardado exitosamente.")

        except TypeError as e:
            self._logger.error(f"Error serializando datos del lote a JSON: {e}. Comprobar tipos.", exc_info=True)
        except OSError as e:
            self._logger.error(f"Error OS guardando lote episodios en {path}: {e}")
        except Exception as e:
            self._logger.error(f"Error inesperado guardando lote episodios en {path}: {e}", exc_info=True)

    def save_metadata(self, metadata: Dict[str, Any], results_folder: str):
        """Guarda metadatos de la simulación en un archivo JSON."""
        if not os.path.isdir(results_folder):
            self._logger.error(f"La carpeta de resultados '{results_folder}' no existe al guardar metadata. Omitiendo.")
            return
        path = os.path.join(results_folder, 'metadata.json')
        self._logger.info(f"Guardando metadata -> {os.path.basename(path)}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, cls=NumpyEncoder, indent=4)
            self._logger.info(f"Metadata guardada exitosamente.")
        except TypeError as e:
            self._logger.error(f"Error serializando metadata a JSON: {e}. Comprobar tipos.", exc_info=True)
        except OSError as e:
            self._logger.error(f"Error OS guardando metadata en {path}: {e}")
        except Exception as e:
            self._logger.error(f"Error inesperado guardando metadata en {path}: {e}", exc_info=True)

    def save_agent_state(self, agent: Optional['RLAgent'], episode: int, results_folder: str):
        """Guarda el estado interno serializable del agente en un archivo JSON."""
        if not os.path.isdir(results_folder):
            self._logger.error(f"Carpeta resultados '{results_folder}' no existe al guardar estado agente. Omitiendo.")
            return
        if agent is None:
            self._logger.warning("Intento de guardar estado agente, pero instancia es None.")
            return
        if not hasattr(agent, 'get_agent_state_for_saving') or not callable(getattr(agent, 'get_agent_state_for_saving')):
            self._logger.error(f"Agente ({type(agent).__name__}) no implementa 'get_agent_state_for_saving()'. No se guarda estado.")
            return

        self._logger.info(f"Guardando estado del agente en episodio {episode}...")
        try:
            agent_state_dict = agent.get_agent_state_for_saving()
            if not agent_state_dict: # Si devuelve dict vacío, advertir
                 self._logger.warning(f"get_agent_state_for_saving() devolvió un diccionario vacío para ep {episode}.")
                 # Decidir si guardar el archivo vacío o no. Optamos por no guardarlo.
                 return

            filename = f"agent_state_ep_{episode}.json"
            path = os.path.join(results_folder, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(agent_state_dict, f, cls=NumpyEncoder, indent=2)
            self._logger.info(f"Estado del agente guardado -> {os.path.basename(path)}")
            self._last_agent_state_json_path = path
        except AttributeError as e:
            self._logger.error(f"Error de atributo obteniendo estado del agente: {e}", exc_info=True)
        except TypeError as e:
            self._logger.error(f"Error serializando estado del agente a JSON: {e}. Comprobar datos.", exc_info=True)
        except OSError as e:
            self._logger.error(f"Error OS guardando estado del agente en {path}: {e}")
        except Exception as e:
            self._logger.error(f"Error inesperado guardando estado del agente en {path}: {e}", exc_info=True)


    def convert_json_agent_state_to_excel(self, json_path: str, excel_path: str):
        """Convierte un archivo JSON de estado de agente a formato Excel (requiere openpyxl)."""
        self._logger.info(f"Intentando convertir estado agente JSON a Excel...")
        self._logger.info(f"  - JSON Origen: {os.path.basename(json_path)}")
        self._logger.info(f"  - Excel Destino: {os.path.basename(excel_path)}")
        try:
            import openpyxl # type: ignore
        except ImportError:
            self._logger.error("Biblioteca 'openpyxl' no encontrada. No se puede convertir a Excel. Instalar: pip install openpyxl")
            return

        writer = None
        saved_sheets = 0
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                agent_state_data = json.load(f)

            if not isinstance(agent_state_data, dict):
                self._logger.error(f"Formato JSON inesperado en {os.path.basename(json_path)}. Se esperaba un diccionario.")
                return

            # (2.8) Preparar datos para escribir, pero abrir writer solo si hay datos
            sheets_to_write = {}
            for key_base, data_dict in agent_state_data.items():
                if isinstance(data_dict, dict):
                    for gain, table_data in data_dict.items():
                        sheet_name = f"{key_base}_{gain}"[:31]
                        if isinstance(table_data, list) and table_data:
                            try:
                                df = pd.DataFrame(table_data)
                                # Verificar si el DataFrame no está vacío después de la conversión
                                if not df.empty:
                                     sheets_to_write[sheet_name] = df
                                else:
                                     self._logger.debug(f"DataFrame vacío para tabla '{sheet_name}', no se guardará en Excel.")
                            except Exception as e_df:
                                self._logger.warning(f"No se pudo convertir tabla '{sheet_name}' a DataFrame: {e_df}")
                        elif isinstance(table_data, list):
                            self._logger.debug(f"Tabla '{sheet_name}' está vacía, no se guarda en Excel.")
                        else:
                            self._logger.warning(f"Datos para '{sheet_name}' no son una lista, no se guarda en Excel.")
                else:
                     self._logger.warning(f"Clave de alto nivel '{key_base}' en JSON no es diccionario.")

            # Escribir solo si hay hojas preparadas
            if sheets_to_write:
                writer = pd.ExcelWriter(excel_path, engine='openpyxl')
                for sheet_name, df in sheets_to_write.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    saved_sheets += 1
                writer.close() # Llama a save()
                self._logger.info(f"Estado del agente convertido a Excel ({saved_sheets} hojas) -> {os.path.basename(excel_path)}")
            else:
                self._logger.warning("No se encontraron datos válidos en el estado del agente JSON para convertir a Excel. No se creó el archivo.")

        except FileNotFoundError:
            self._logger.error(f"Archivo JSON agente no encontrado: {json_path}")
        except json.JSONDecodeError as e:
            self._logger.error(f"Error decodificando JSON '{os.path.basename(json_path)}': {e}")
        except OSError as e:
            self._logger.error(f"Error OS guardando Excel en {excel_path}: {e}")
        except IndexError as e_excel: # Capturar específicamente el error de openpyxl
            self._logger.error(f"Error de openpyxl al guardar Excel (posiblemente archivo vacío o dañado internamente): {e_excel}")
        except Exception as e:
            self._logger.error(f"Error inesperado convirtiendo JSON a Excel: {e}", exc_info=True)
        finally:
             # Asegurar cierre del writer si se abrió y falló antes de close()
             if writer is not None and saved_sheets > 0 and hasattr(writer, 'close') and not getattr(writer, 'closed', True):
                 try: writer.close()
                 except: pass


    def save_summary_table(self, summary_list: List[Dict], results_folder: str):
        """Guarda la tabla de resumen de episodios en un archivo Excel."""
        if not os.path.isdir(results_folder):
            self._logger.error(f"Carpeta resultados '{results_folder}' no existe al guardar resumen. Omitiendo.")
            return
        path = os.path.join(results_folder, 'episodes_summary.xlsx')
        if not summary_list:
            self._logger.warning("Lista de resúmenes vacía. No se guardará archivo de resumen.")
            return

        self._logger.info(f"Guardando tabla de resumen -> {os.path.basename(path)}")
        try:
            import openpyxl # type: ignore
        except ImportError:
            self._logger.error("Biblioteca 'openpyxl' no encontrada. No se puede guardar resumen en Excel.")
            return

        try:
            df = pd.DataFrame(summary_list)
            if 'episode' in df.columns:
                # Convertir a Int64 que soporta <NA>
                df['episode'] = pd.to_numeric(df['episode'], errors='coerce').astype('Int64')

            # Reordenar columnas
            cols = df.columns.tolist()
            preferred_order = [
                'episode', 'episode_time', 'total_reward', 'performance', 'termination_reason',
                'episode_duration_s','avg_stability_score', 'w_stab_kp_cf', 'w_stab_ki_cf', 'w_stab_kd_cf', 
                'total_agent_decisions', 'final_epsilon', 'final_learning_rate', 'final_kp', 'final_ki', 'final_kd',
            ]
            # Incluir todas las demás columnas después de las preferidas
            ordered_cols = [c for c in preferred_order if c in cols] + sorted([c for c in cols if c not in preferred_order])
            df_ordered = df[ordered_cols]

            df_ordered.to_excel(path, index=False, engine='openpyxl')
            self._logger.info(f"Tabla de resumen guardada exitosamente.")
        except OSError as e:
            self._logger.error(f"Error OS guardando archivo de resumen en {path}: {e}")
        except Exception as e:
            self._logger.error(f"Error inesperado guardando tabla de resumen en {path}: {e}", exc_info=True)

    def finalize(self,
                 config: Dict[str, Any],
                 vis_config: Optional[Dict[str, Any]], # Recibe vis_config pero no lo usa directamente
                 summary_data: List[Dict],
                 all_episodes_data: List[Dict],
                 agent: Optional['RLAgent'],
                 results_folder: str):
        """
        Realiza las tareas finales de guardado al final de la simulación.
        NO genera datos de heatmap.
        """
        # 3.7: Lógica de finalize sin heatmap_generator.
        if not os.path.isdir(results_folder):
            self._logger.error(f"Carpeta resultados '{results_folder}' no existe en finalize. Abortando finalización.")
            return
        self._logger.info(f"--- Iniciando Proceso de Finalización de ResultHandler en: {results_folder} ---")

        sim_cfg = config.get('simulation', {})
        env_cfg = config.get('environment', {})

        # 1. Guardar estado final del agente
        save_final_state = sim_cfg.get('save_agent_state', False)
        if save_final_state and agent is not None:
            max_episodes = env_cfg.get('max_episodes', 0)
            # El último episodio es max_episodes - 1 (indexado desde 0)
            final_episode_index = max(0, max_episodes - 1)
            self.save_agent_state(agent, final_episode_index, results_folder)
        elif save_final_state:
            self._logger.warning("Guardado final agente habilitado, pero agente es None.")
        else:
            self._logger.info("Guardado final del estado del agente deshabilitado o no aplicable.")

        # 2. Convertir último estado guardado a Excel
        if self._last_agent_state_json_path and os.path.exists(self._last_agent_state_json_path):
            excel_path = os.path.join(results_folder, 'agent_state_tables_final.xlsx')
            self.convert_json_agent_state_to_excel(self._last_agent_state_json_path, excel_path)
        elif save_final_state:
            self._logger.warning("No se encontró archivo JSON del estado final del agente para convertir a Excel (puede ser normal si no se guardó).")

        # 3. Guardar tabla de resumen
        if summary_data: # Solo guardar si hay datos
            self.save_summary_table(summary_data, results_folder)
        else:
            self._logger.info("No hay datos de resumen para guardar.")

        # 4. Guardar datos detallados si no se guardaron por batch (poco común)
        if all_episodes_data:
            self._logger.warning("'all_episodes_data' contiene datos al finalizar. Intentando guardar como un último batch...")
            max_ep = env_cfg.get('max_episodes', 0)
            final_ep_idx = max(0, max_ep - 1) if max_ep > 0 else 0
            self.save_episode_batch(all_episodes_data, results_folder, final_ep_idx)

        self._logger.info("--- Proceso de Finalización de ResultHandler Completado ---")