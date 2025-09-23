# utils/data/result_handler.py
import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from utils.data.numpy_encoder import NumpyEncoder # Importar el encoder personalizado

if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent # Para type hint

class ResultHandler:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._last_saved_agent_state_json_path: Optional[str] = None # Para saber qué convertir a Excel
        self._logger.info("[ResultHandler] Instance created.")

    @staticmethod
    def setup_results_folder(base_output_folder_name: str = 'results_history') -> str:
        """Crea una carpeta de resultados única con timestamp."""
        current_ts = datetime.now().strftime('%Y%m%d-%H%M')
        # Determinar directorio base del script de forma robusta
        try: script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception: script_dir = os.getcwd()
        
        results_root = os.path.join(script_dir, base_output_folder_name)
        specific_run_output_dir = os.path.join(results_root, current_ts)
        os.makedirs(specific_run_output_dir, exist_ok=True)
        logging.info(f"[ResultHandler:Setup] Output directory: {specific_run_output_dir}")
        return specific_run_output_dir

    def save_episode_batch(self,
                           episode_data_list_batch: List[Dict],
                           output_dir_path_batch: str,
                           last_episode_idx_in_batch: int):
        """Guarda un lote de datos detallados de episodios a JSON."""
        if not episode_data_list_batch:
            # self._logger.debug("[ResultHandler:SaveBatch] Empty episode data batch. Skipping save.")
            return
        
        num_eps_in_batch = len(episode_data_list_batch)
        first_ep_idx = max(0, last_episode_idx_in_batch - num_eps_in_batch + 1)
        filename_rng_str = f"{first_ep_idx}_to_{last_episode_idx_in_batch}"
        batch_filename = f"simulation_data_ep_{filename_rng_str}.json"
        full_filepath = os.path.join(output_dir_path_batch, batch_filename)
        
        # self._logger.info(f"[ResultHandler:SaveBatch] Saving batch for episodes {filename_rng_str} to {batch_filename}")
        with open(full_filepath, 'w', encoding='utf-8') as f_json:
            json.dump(episode_data_list_batch, f_json, cls=NumpyEncoder, indent=2) # Usar NumpyEncoder
        # self._logger.debug(f"[ResultHandler:SaveBatch] Batch of {num_eps_in_batch} episodes saved to {full_filepath}.")

    def save_metadata(self, metadata_to_save: Dict[str, Any], output_dir_meta: str):
        """Guarda metadata de la simulación a JSON."""
        meta_filepath = os.path.join(output_dir_meta, 'metadata_simulation_run.json')
        # self._logger.info(f"[ResultHandler:SaveMetadata] Saving execution metadata to {os.path.basename(meta_filepath)}")
        with open(meta_filepath, 'w', encoding='utf-8') as f_meta:
            json.dump(metadata_to_save, f_meta, cls=NumpyEncoder, indent=4) # Usar NumpyEncoder
        self._logger.info(f"[ResultHandler:SaveMetadata] Metadata saved to {meta_filepath}.")

    def save_agent_state(self,
                         agent_instance_to_save: Optional['RLAgent'],
                         episode_num_for_filename: int,
                         output_dir_agent: str):
        """Guarda el estado del agente a JSON."""
        if agent_instance_to_save is None:
            # self._logger.warning("[ResultHandler:SaveAgent] Agent instance is None. Skipping agent state save.")
            return
        
        # self._logger.info(f"[ResultHandler:SaveAgent] Attempting to save agent state for episode {episode_num_for_filename}...")
        agent_state_dict = agent_instance_to_save.get_agent_state_for_saving()
        if not agent_state_dict: # Si el método devuelve vacío, no guardar
            # self._logger.warning(f"[ResultHandler:SaveAgent] get_agent_state_for_saving() returned empty for ep {episode_num_for_filename}. Not saving.")
            return
            
        agent_filename = f"agent_state_ep_{episode_num_for_filename}.json"
        agent_filepath = os.path.join(output_dir_agent, agent_filename)
        
        with open(agent_filepath, 'w', encoding='utf-8') as f_agent_json:
            json.dump(agent_state_dict, f_agent_json, cls=NumpyEncoder, indent=2) # Usar NumpyEncoder
        self._logger.info(f"[ResultHandler:SaveAgent] Agent state for ep {episode_num_for_filename} saved to {agent_filename}")
        self._last_saved_agent_state_json_path = agent_filepath # Guardar ruta para posible conversión a Excel

    def convert_json_agent_state_to_excel(self,
                                          agent_json_path: str,
                                          output_excel_path: str):
        """Convierte un archivo JSON de estado de agente a un archivo Excel con múltiples hojas."""
        self._logger.info(f"[ResultHandler:ConvertJSON] Converting agent state: {os.path.basename(agent_json_path)} -> {os.path.basename(output_excel_path)}")
        try: 
            import openpyxl # Intentar importar aquí para que el error sea específico si falta
        except ImportError: 
            self._logger.error("[ResultHandler:ConvertJSON] 'openpyxl' library not found. Cannot convert agent state to Excel. Install: pip install openpyxl"); return

        try:
            with open(agent_json_path, 'r', encoding='utf-8') as f_json_agent_in:
                agent_state_data = json.load(f_json_agent_in)
            
            if not isinstance(agent_state_data, dict):
                self._logger.error(f"Agent state JSON ({os.path.basename(agent_json_path)}) is not a root dictionary. Cannot convert."); return
            
            # Usar ExcelWriter para múltiples hojas
            with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                num_sheets_written = 0
                # Iterar sobre las categorías principales (ej. "q_tables", "visit_counts", "baseline_tables")
                for category_key, category_data_dict in agent_state_data.items():
                    if isinstance(category_data_dict, dict):
                        # Iterar sobre las ganancias (sub-agentes) dentro de cada categoría
                        for gain_name_key, table_as_list_of_dicts in category_data_dict.items():
                            sheet_name = f"{category_key[:15]}_{gain_name_key[:15]}"[:31] # Nombre de hoja corto
                            if isinstance(table_as_list_of_dicts, list) and table_as_list_of_dicts:
                                df_sheet = pd.DataFrame(table_as_list_of_dicts)
                                if not df_sheet.empty:
                                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                                    num_sheets_written +=1
            if num_sheets_written > 0:
                 self._logger.info(f"[ResultHandler:ConvertJSON] Agent state ({num_sheets_written} sheets) converted to Excel: {os.path.basename(output_excel_path)}")
            else:
                 self._logger.warning(f"[ResultHandler:ConvertJSON] No data found in agent state JSON to convert to Excel sheets for {os.path.basename(agent_json_path)}.")

        except FileNotFoundError: self._logger.error(f"Agent state JSON file not found: {agent_json_path}")
        except json.JSONDecodeError: self._logger.error(f"JSON decode error in agent state file: {agent_json_path}")
        # pd.ExcelWriter o df.to_excel pueden lanzar varios errores (OSError, etc.)
        except Exception as e_excel: self._logger.error(f"Error converting agent state to Excel ({output_excel_path}): {e_excel}", exc_info=True)


    def save_summary_table(self, 
                           summary_data_list: List[Dict], 
                           output_dir_summary_table: str,
                           # summary_directives ya no es necesario si se asume que summarize_episode ya filtró
                           # O, si queremos reordenar/filtrar columnas aquí basado en config, se necesitaría.
                           # Por simpleza, asumimos que summary_data_list ya tiene las columnas deseadas.
                           # Si se necesita filtrado/reordenado aquí, pasar summary_directives.
                          ):
        """Guarda la lista de resúmenes de episodios a un archivo Excel."""
        if not summary_data_list: 
            # self._logger.debug("[ResultHandler:SaveSummary] Empty summary data list. Not saving summary file.")
            return
        
        summary_filename = 'episodes_summary_data.xlsx'
        summary_filepath = os.path.join(output_dir_summary_table, summary_filename)
        self._logger.info(f"[ResultHandler:SaveSummary] Saving episodes summary table to {summary_filename}")
        try: 
            import openpyxl 
        except ImportError: 
            self._logger.error("[ResultHandler:SaveSummary] 'openpyxl' needed for Excel summary. Install: pip install openpyxl"); return
        
        summary_df_to_save = pd.DataFrame(summary_data_list)
        if summary_df_to_save.empty:
            self._logger.warning("[ResultHandler:SaveSummary] Summary DataFrame is empty. Not saving file.")
            return

        if 'episode' in summary_df_to_save.columns: # Asegurar que 'episode' sea entero si existe
            summary_df_to_save['episode'] = pd.to_numeric(summary_df_to_save['episode'], errors='coerce').astype('Int64')
            # Opcional: mover 'episode' al principio si no lo está
            if 'episode' in summary_df_to_save.columns and summary_df_to_save.columns[0] != 'episode':
                cols = ['episode'] + [col for col in summary_df_to_save.columns if col != 'episode']
                summary_df_to_save = summary_df_to_save[cols]
        
        summary_df_to_save.to_excel(summary_filepath, index=False, engine='openpyxl')
        self._logger.info(f"[ResultHandler:SaveSummary] Episodes summary table saved to {summary_filepath} ({len(summary_df_to_save)} rows).")

    def finalize(self,
                 main_config: Dict[str, Any], 
                 # vis_config: Optional[Dict[str, Any]], # No usado directamente por finalize para guardado
                 processed_data_directives: Dict[str, Any],
                 summary_data: List[Dict], 
                 # all_episodes_detailed_data: List[Dict], # Ya no se pasa esto, se guarda en lotes
                 agent: Optional['RLAgent'], 
                 output_dir_finalize: str
                ):
        """Realiza operaciones de guardado finales (estado del agente, resumen)."""
        self._logger.info(f"[ResultHandler:Finalize] --- Starting ResultHandler Finalization in: {output_dir_finalize} ---")

        # 1. Guardar estado final del agente (si está habilitado en main_config.data_handling)
        save_final_agent_cfg = main_config.get('data_handling', {}).get('save_agent_state', False)
        if save_final_agent_cfg and agent is not None:
            max_eps_cfg_final = main_config.get('environment', {}).get('simulation', {}).get('max_episodes', 0)
            # Guardar con el índice del último episodio ejecutado
            final_ep_idx_to_save = max(0, max_eps_cfg_final - 1) if max_eps_cfg_final > 0 else 0
            self.save_agent_state(agent, final_ep_idx_to_save, output_dir_finalize)
            
            # Convertir el último estado guardado a Excel (si se guardó)
            if self._last_saved_agent_state_json_path and os.path.exists(self._last_saved_agent_state_json_path):
                agent_excel_path = os.path.join(output_dir_finalize, 'agent_state_final_tables.xlsx')
                self.convert_json_agent_state_to_excel(self._last_saved_agent_state_json_path, agent_excel_path)
        elif save_final_agent_cfg:
            self._logger.warning("[ResultHandler:Finalize] Agent state saving enabled, but no agent instance provided.")

        # 2. Guardar tabla de resumen de episodios (si está habilitado en directivas)
        if processed_data_directives.get('summary_enabled', False):
            if summary_data: 
                self.save_summary_table(summary_data, output_dir_finalize)
            else: 
                self._logger.info("[ResultHandler:Finalize] Summary data list is empty. Summary table not saved.")
        else:
            self._logger.info("[ResultHandler:Finalize] Summary table saving is disabled by data handling directives.")

        # Los datos detallados ya se guardaron en lotes por SimulationManager
        self._logger.info("[ResultHandler:Finalize] --- ResultHandler Finalization Complete ---")