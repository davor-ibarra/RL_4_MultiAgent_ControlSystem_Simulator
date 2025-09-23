import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from utils.numpy_encoder import NumpyEncoder
# Importar RLAgent sólo para type hints
if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent
# Importar HeatmapGenerator normalmente ya que se usa en __init__
from heatmap_generator import HeatmapGenerator

class ResultHandler:
    """
    Servicio centralizado para gestionar el guardado de todos los artefactos
    de resultados de la simulación. Ahora recibe la carpeta de resultados explícitamente.
    """

    def __init__(self, logger: logging.Logger, heatmap_generator: HeatmapGenerator):
        """
        Inicializa el ResultHandler con sus dependencias (logger y heatmap_generator).
        Ya no mantiene el estado de results_folder internamente.

        Args:
            logger: Instancia del logger configurado.
            heatmap_generator: Instancia del generador de datos de heatmap.
        """
        self._logger = logger
        # --- VALIDACIÓN INTERNA ELIMINADA ---
        # La responsabilidad de pasar el tipo correcto es del contenedor DI.
        # Simplemente asignamos las dependencias inyectadas.
        self._heatmap_generator = heatmap_generator
        self._last_agent_state_json_path: Optional[str] = None # Mantenemos esto para la conversión a Excel
        self._logger.info("ResultHandler instance created.")

    @staticmethod
    def setup_results_folder(base_results_folder: str = 'results_history') -> str:
        """
        Crea la carpeta de resultados única para esta ejecución basada en timestamp.

        Args:
            base_results_folder (str): Nombre de la carpeta base (relativa al script).

        Returns:
            str: La ruta absoluta a la carpeta de resultados creada.

        Raises:
            OSError: Si falla la creación de la carpeta.
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        # Obtener ruta absoluta basada en la ubicación de *este* archivo (result_handler.py)
        # Asume que la carpeta base está relativa al directorio del proyecto/script.
        try:
             # ASUME QUE result_handler.py está en el directorio raíz del proyecto
             project_root = os.path.dirname(os.path.abspath(__file__))
             # Si estuviera en una subcarpeta 'services', usaría:
             # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Sube un nivel
             base_folder_path = os.path.join(project_root, base_results_folder)
             results_folder = os.path.join(base_folder_path, timestamp)
             os.makedirs(results_folder, exist_ok=True)
             # Usar logging estándar aquí, no el logger de instancia
             logging.info(f"[ResultHandler] Results folder created/verified: {results_folder}")
             return results_folder
        except OSError as e:
             logging.error(f"[ResultHandler] Failed to create results folder {results_folder}: {e}")
             raise # Relanzar para detener la ejecución si no se puede crear


    # --- Métodos de Guardado (ahora requieren 'results_folder') ---

    def save_episode_batch(self, batch_data: List[Dict], results_folder: str, last_episode: int):
        """Guarda un lote de datos detallados de episodios en un archivo JSON."""
        if not os.path.isdir(results_folder): self._logger.error(f"[ResultHandler] La carpeta de resultados '{results_folder}' no existe al guardar batch. Omitiendo."); return
        if not batch_data: self._logger.warning("[ResultHandler] Se intentó guardar un lote de episodios vacío. Omitiendo."); return
        try:
            # --- Calcular el primer episodio del lote ---
            num_episodes_in_batch = len(batch_data)
            # Asumiendo que los episodios son consecutivos y terminan en last_episode
            first_episode_in_batch = last_episode - num_episodes_in_batch + 1

            # Validar que el cálculo sea razonable (el primer episodio debe ser >= 0)
            if first_episode_in_batch < 0:
                 self._logger.warning(f"[ResultHandler] Cálculo del primer episodio dio negativo ({first_episode_in_batch}). Usando 0 como inicio. last_episode={last_episode}, batch_size={num_episodes_in_batch}")
                 first_episode_in_batch = 0 # Ajustar a 0 si el cálculo da negativo

            # Usar los números calculados para el nombre del archivo
            filename_range = f"{first_episode_in_batch}_to_{last_episode}"

            filename = f"simulation_data_{filename_range}.json"
            path = os.path.join(results_folder, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, cls=NumpyEncoder, indent=2)
            self._logger.info(f"[ResultHandler] Lote de episodios {filename_range} guardado -> {os.path.basename(path)}")
        except IndexError: self._logger.error("[ResultHandler] No se puede determinar el rango del archivo: batch_data está vacío o malformado.")
        except TypeError as e: self._logger.error(f"[ResultHandler] Error serializando datos del lote a JSON: {e}. Comprobar tipos de datos.", exc_info=True)
        except OSError as e: self._logger.error(f"[ResultHandler] Error de OS guardando lote de episodios en {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler] Error inesperado guardando lote de episodios en {path}: {e}", exc_info=True)


    def save_metadata(self, metadata: Dict[str, Any], results_folder: str):
        """Guarda metadatos de la simulación en un archivo JSON."""
        # (Código sin cambios)
        if not os.path.isdir(results_folder): self._logger.error(f"[ResultHandler] La carpeta de resultados '{results_folder}' no existe al guardar metadata. Omitiendo."); return
        path = os.path.join(results_folder, 'metadata.json')
        self._logger.info(f"[ResultHandler] Guardando metadata -> {os.path.basename(path)}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, cls=NumpyEncoder, indent=4)
            self._logger.info(f"[ResultHandler] Metadata guardada exitosamente.")
        except TypeError as e: self._logger.error(f"[ResultHandler] Error serializando metadata a JSON: {e}. Comprobar tipos en metadata_dict.", exc_info=True)
        except OSError as e: self._logger.error(f"[ResultHandler] Error de OS guardando metadata en {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler] Error inesperado guardando metadata en {path}: {e}", exc_info=True)


    def save_agent_state(self, agent: Optional['RLAgent'], episode: int, results_folder: str):
        """Guarda el estado interno serializable del agente en un archivo JSON."""
        # (Código sin cambios)
        if not os.path.isdir(results_folder): self._logger.error(f"[ResultHandler] La carpeta de resultados '{results_folder}' no existe al guardar estado del agente. Omitiendo."); return
        if agent is None: self._logger.warning("[ResultHandler] Se intentó guardar el estado del agente, pero la instancia es None."); return
        if not hasattr(agent, 'get_agent_state_for_saving'):
            self._logger.error(f"[ResultHandler] El agente ({type(agent).__name__}) no implementa get_agent_state_for_saving(). No se puede guardar el estado.")
            return

        self._logger.info(f"[ResultHandler] Guardando estado del agente en episodio {episode}...")
        try:
            agent_state_dict = agent.get_agent_state_for_saving()
            filename = f"agent_state_episode_{episode}.json"
            path = os.path.join(results_folder, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(agent_state_dict, f, cls=NumpyEncoder, indent=2)
            self._logger.info(f"[ResultHandler] Estado del agente guardado -> {os.path.basename(path)}")
            self._last_agent_state_json_path = path
        except AttributeError as e: self._logger.error(f"[ResultHandler] Error llamando a get_agent_state_for_saving en el agente: {e}")
        except TypeError as e: self._logger.error(f"[ResultHandler] Error serializando estado del agente a JSON: {e}. Comprobar datos devueltos.", exc_info=True)
        except OSError as e: self._logger.error(f"[ResultHandler] Error de OS guardando estado del agente en {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler] Error inesperado guardando estado del agente en {path}: {e}", exc_info=True)


    def convert_json_agent_state_to_excel(self, json_path: str, excel_path: str):
        """Convierte un archivo JSON de estado de agente a formato Excel (requiere openpyxl)."""
        # (Código sin cambios)
        self._logger.info(f"[ResultHandler] Intentando convertir estado del agente JSON a Excel...")
        self._logger.info(f"  - JSON Origen: {os.path.basename(json_path)}")
        self._logger.info(f"  - Excel Destino: {os.path.basename(excel_path)}")
        try: import openpyxl # type: ignore
        except ImportError: self._logger.error("[ResultHandler] Biblioteca 'openpyxl' no encontrada. No se puede convertir a Excel. Instalar: pip install openpyxl"); return
        try:
            with open(json_path, 'r', encoding='utf-8') as f: agent_state_data = json.load(f)
            if not isinstance(agent_state_data, dict): self._logger.error("[ResultHandler] Formato JSON inesperado."); return
            expected_keys = ['q_tables', 'visit_counts', 'baseline_tables']; missing = set(expected_keys) - set(agent_state_data.keys())
            if missing: self._logger.warning(f"[ResultHandler] JSON agente faltan claves: {missing}")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for key_base, data_dict in [('q', agent_state_data.get('q_tables')), ('visits', agent_state_data.get('visit_counts')), ('baseline', agent_state_data.get('baseline_tables'))]:
                     if isinstance(data_dict, dict):
                         for gain, table_data in data_dict.items():
                             if table_data and isinstance(table_data, list):
                                 try: pd.DataFrame(table_data).to_excel(writer, sheet_name=f"{key_base}_{gain}", index=False)
                                 except Exception as e_df: self._logger.warning(f"No se pudo convertir tabla '{key_base}_{gain}' a Excel: {e_df}")
                             # else: self._logger.debug(f"Tabla '{key_base}_{gain}' vacía o no es lista.")
                     # else: self._logger.warning(f"Clave '{key_base}_tables' no es dict.")
            self._logger.info(f"[ResultHandler] Estado del agente convertido a Excel -> {os.path.basename(excel_path)}")
        except FileNotFoundError: self._logger.error(f"[ResultHandler] Archivo JSON agente no encontrado: {json_path}")
        except json.JSONDecodeError as e: self._logger.error(f"[ResultHandler] Error decodificando JSON: {json_path}: {e}")
        except OSError as e: self._logger.error(f"[ResultHandler] Error OS guardando Excel: {excel_path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler] Error inesperado convirtiendo JSON a Excel: {e}", exc_info=True)


    def save_summary_table(self, summary_list: List[Dict], results_folder: str):
        """Guarda la tabla de resumen de episodios en un archivo Excel."""
        if not os.path.isdir(results_folder): self._logger.error(f"[ResultHandler] La carpeta de resultados '{results_folder}' no existe al guardar resumen. Omitiendo."); return
        path = os.path.join(results_folder, 'summary.xlsx')
        if not summary_list: self._logger.warning("[ResultHandler] La lista de resúmenes está vacía. No se guardará el archivo de resumen."); return

        self._logger.info(f"[ResultHandler] Guardando tabla de resumen -> {os.path.basename(path)}")
        try: import openpyxl # type: ignore
        except ImportError: self._logger.error("[ResultHandler] Biblioteca 'openpyxl' no encontrada. No se puede guardar resumen en Excel. Instalar: pip install openpyxl"); return

        try:
            df = pd.DataFrame(summary_list)
            if 'episode' in df.columns:
                df['episode'] = pd.to_numeric(df['episode'], errors='coerce').astype('Int64')

            # --- MODIFICACIÓN: Reordenar Columnas ---
            cols = df.columns.tolist()
            # Lista de columnas preferidas (incluyendo las solicitadas)
            preferred_order = [
                # Generales
                'episode', 'termination_reason', 'total_reward', 'performance',
                'episode_duration_s', 'avg_stability_score', 'total_agent_decisions',
                # Agente (finales)
                'final_epsilon', 'final_learning_rate', 'final_kp', 'final_ki', 'final_kd',
                # Otras métricas clave podrían ir aquí...
                # 'reward_mean', 'reward_std', ... # Métricas agregadas van después
            ]
            # Crear lista ordenada: preferidas existentes + resto
            ordered_cols = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
            df = df[ordered_cols] # Reindexar DataFrame
            # --- FIN MODIFICACIÓN ---

            df.to_excel(path, index=False, engine='openpyxl')
            self._logger.info(f"[ResultHandler] Tabla de resumen guardada exitosamente.")
        except OSError as e: self._logger.error(f"[ResultHandler] Error de OS guardando archivo de resumen en {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler] Error inesperado guardando tabla de resumen en {path}: {e}", exc_info=True)


    def generate_heatmap_data(self, vis_config: Optional[Dict], results_folder: str):
        """
        Genera los datos numéricos para los heatmaps definidos en vis_config.
        Busca automáticamente el último archivo de datos detallados si es necesario.
        """
        # (Código sin cambios)
        if not os.path.isdir(results_folder): self._logger.error(f"[ResultHandler] La carpeta de resultados '{results_folder}' no existe al generar datos heatmap. Omitiendo."); return
        if self._heatmap_generator is None: self._logger.error("[ResultHandler] HeatmapGenerator no está disponible. No se pueden generar datos."); return
        if not vis_config or not isinstance(vis_config, dict) or not vis_config.get('plots'):
            self._logger.info("[ResultHandler] No hay config visualización válida o 'plots'. Omitiendo datos heatmap.")
            return
        heatmap_configs = [p for p in vis_config.get("plots", []) if isinstance(p, dict) and p.get("type") == "heatmap" and p.get("enabled", True)]
        if not heatmap_configs: self._logger.info("[ResultHandler] No hay heatmaps habilitados."); return
        detailed_data_filepath = self._heatmap_generator.find_latest_simulation_data(results_folder)
        if not detailed_data_filepath: self._logger.error(f"[ResultHandler] No se encontró simulation_data_*.json en '{results_folder}' para heatmaps."); return
        output_excel_filepath = os.path.join(results_folder, "data_heatmaps.xlsx")
        self._logger.info(f"[ResultHandler] Generando datos heatmap desde '{os.path.basename(detailed_data_filepath)}' -> {os.path.basename(output_excel_filepath)}")
        try:
            self._heatmap_generator.generate(detailed_data_filepath=detailed_data_filepath, heatmap_configs=heatmap_configs, output_excel_filepath=output_excel_filepath)
            self._logger.info("[ResultHandler] Datos de heatmap generados exitosamente.")
        except Exception as e: self._logger.error(f"[ResultHandler] Error generando datos de heatmap: {e}", exc_info=True)


    def finalize(self,
                 config: Dict[str, Any],
                 summary_data: List[Dict],
                 all_episodes_data: List[Dict], # Nota: suele estar vacío si se guarda por batch
                 agent: Optional['RLAgent'],
                 results_folder: str):
        """
        Realiza las tareas finales de guardado de resultados al final de la simulación.
        """
        # (Código sin cambios)
        if not os.path.isdir(results_folder): self._logger.error(f"[ResultHandler] La carpeta de resultados '{results_folder}' no existe en finalize. Abortando finalización."); return
        self._logger.info(f"[ResultHandler] Iniciando proceso de finalización en: {results_folder}")
        sim_cfg = config.get('simulation', {}); env_cfg = config.get('environment', {}); vis_cfg = config.get('visualization')
        save_final_state = sim_cfg.get('save_agent_state', False)
        if save_final_state and agent is not None:
             max_episodes = env_cfg.get('max_episodes', 0); final_episode_index = max(0, max_episodes - 1)
             self.save_agent_state(agent, final_episode_index, results_folder)
        elif save_final_state and agent is None: self._logger.warning("[ResultHandler] Guardado final habilitado, pero agente es None.")
        else: self._logger.info("[ResultHandler] Guardado final agente deshabilitado o no aplicable.")
        if self._last_agent_state_json_path and os.path.exists(self._last_agent_state_json_path):
            excel_path = os.path.join(results_folder, 'agent_state_tables.xlsx')
            self.convert_json_agent_state_to_excel(self._last_agent_state_json_path, excel_path)
        elif save_final_state: self._logger.warning("[ResultHandler] No se encontró JSON agente para convertir a Excel.")
        self.save_summary_table(summary_data, results_folder)
        self.generate_heatmap_data(vis_cfg, results_folder)
        if all_episodes_data:
             self._logger.warning("[ResultHandler] 'all_episodes_data' contiene datos al finalizar. Intentando guardar...")
             max_ep = env_cfg.get('max_episodes', 0); self.save_episode_batch(all_episodes_data, results_folder, max(0, max_ep - 1))
        self._logger.info("[ResultHandler] Proceso de finalización completado.")