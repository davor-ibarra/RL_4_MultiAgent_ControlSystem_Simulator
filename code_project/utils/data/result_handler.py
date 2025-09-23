# utils/data/result_handler.py
import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from utils.data.numpy_encoder import NumpyEncoder
if TYPE_CHECKING:
    from interfaces.rl_agent import RLAgent

class ResultHandler:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        if not isinstance(logger, logging.Logger):
            logging.getLogger().critical("[ResultHandler:__init__] CRITICAL: Valid logging.Logger instance required.")
            raise TypeError("Valid logging.Logger instance required.")
        self._last_agent_state_json_path: Optional[str] = None
        self._logger.info("[ResultHandler] Instance created.")

    @staticmethod
    def setup_results_folder(base_results_folder: str = 'results_history') -> str:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        results_folder = "FOLDER_PATH_ERROR"
        try:
             try: script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
             except Exception: script_dir = os.getcwd(); logging.warning(f"[ResultHandler:Setup] Could not determine script dir, using CWD: {script_dir}")
             base_folder_path = os.path.join(script_dir, base_results_folder)
             results_folder = os.path.join(base_folder_path, timestamp)
             os.makedirs(results_folder, exist_ok=True)
             logging.info(f"[ResultHandler:Setup] Results folder created/verified: {results_folder}")
             return results_folder
        except OSError as e:
             logging.error(f"[ResultHandler:Setup] Failed to create results folder {results_folder}: {e}")
             raise
        except Exception as e_setup:
            logging.error(f"[ResultHandler:Setup] Unexpected error setting up results folder: {e_setup}", exc_info=True)
            raise RuntimeError(f"Unexpected error setting up results folder: {e_setup}") from e_setup

    def save_episode_batch(self, batch_data: List[Dict], results_folder: str, last_episode: int):
        if not os.path.isdir(results_folder):
            self._logger.error(f"[ResultHandler:SaveBatch] Results folder '{results_folder}' not found. Skipping."); return
        if not batch_data: self._logger.warning("[ResultHandler:SaveBatch] Empty episode batch. Skipping."); return
        try:
            num_episodes_in_batch = len(batch_data)
            first_episode_in_batch = max(0, last_episode - num_episodes_in_batch + 1)
            filename_range = f"{first_episode_in_batch}_to_{last_episode}"
            filename = f"simulation_data_ep_{filename_range}.json"
            path = os.path.join(results_folder, filename)
            self._logger.info(f"[ResultHandler:SaveBatch] Saving batch episodes {filename_range} to {filename}")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, cls=NumpyEncoder, indent=2)
            self._logger.info(f"[ResultHandler:SaveBatch] Batch episodes saved successfully.")
        except TypeError as e: self._logger.error(f"[ResultHandler:SaveBatch] JSON serialization error: {e}. Check data types.", exc_info=True)
        except OSError as e: self._logger.error(f"[ResultHandler:SaveBatch] OS error saving batch to {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler:SaveBatch] Unexpected error saving batch to {path}: {e}", exc_info=True)

    def save_metadata(self, metadata: Dict[str, Any], results_folder: str):
        if not os.path.isdir(results_folder):
            self._logger.error(f"[ResultHandler:SaveMetadata] Results folder '{results_folder}' not found. Skipping."); return
        path = os.path.join(results_folder, 'metadata.json')
        self._logger.info(f"[ResultHandler:SaveMetadata] Saving metadata to {os.path.basename(path)}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, cls=NumpyEncoder, indent=4)
            self._logger.info(f"[ResultHandler:SaveMetadata] Metadata saved successfully.")
        except TypeError as e: self._logger.error(f"[ResultHandler:SaveMetadata] JSON serialization error: {e}. Check data types.", exc_info=True)
        except OSError as e: self._logger.error(f"[ResultHandler:SaveMetadata] OS error saving metadata to {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler:SaveMetadata] Unexpected error saving metadata to {path}: {e}", exc_info=True)

    def save_agent_state(self, agent: Optional['RLAgent'], episode: int, results_folder: str):
        if not os.path.isdir(results_folder):
            self._logger.error(f"[ResultHandler:SaveAgent] Results folder '{results_folder}' not found. Skipping."); return
        if agent is None: self._logger.warning("[ResultHandler:SaveAgent] Agent instance is None. Skipping save."); return
        if not hasattr(agent, 'get_agent_state_for_saving') or not callable(getattr(agent, 'get_agent_state_for_saving')):
            self._logger.error(f"[ResultHandler:SaveAgent] Agent ({type(agent).__name__}) missing 'get_agent_state_for_saving()'. Skipping."); return

        self._logger.info(f"[ResultHandler:SaveAgent] Saving agent state for episode {episode}...")
        try:
            agent_state_dict = agent.get_agent_state_for_saving()
            if not agent_state_dict:
                 self._logger.warning(f"[ResultHandler:SaveAgent] get_agent_state_for_saving() returned empty dict for ep {episode}. Not saving."); return
            filename = f"agent_state_ep_{episode}.json"
            path = os.path.join(results_folder, filename)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(agent_state_dict, f, cls=NumpyEncoder, indent=2)
            self._logger.info(f"[ResultHandler:SaveAgent] Agent state saved to {os.path.basename(path)}")
            self._last_agent_state_json_path = path
        except AttributeError as e: self._logger.error(f"[ResultHandler:SaveAgent] Attribute error getting agent state: {e}", exc_info=True)
        except TypeError as e: self._logger.error(f"[ResultHandler:SaveAgent] JSON serialization error for agent state: {e}. Check data.", exc_info=True)
        except OSError as e: self._logger.error(f"[ResultHandler:SaveAgent] OS error saving agent state to {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler:SaveAgent] Unexpected error saving agent state to {path}: {e}", exc_info=True)

    def convert_json_agent_state_to_excel(self, json_path: str, excel_path: str):
        self._logger.info(f"[ResultHandler:ConvertJSON] Converting agent state JSON to Excel: {os.path.basename(json_path)} -> {os.path.basename(excel_path)}")
        try: import openpyxl # type: ignore
        except ImportError: self._logger.error("[ResultHandler:ConvertJSON] 'openpyxl' not found. Cannot convert to Excel. Install: pip install openpyxl"); return

        writer = None; saved_sheets = 0
        try:
            with open(json_path, 'r', encoding='utf-8') as f: agent_state_data = json.load(f)
            if not isinstance(agent_state_data, dict):
                self._logger.error(f"[ResultHandler:ConvertJSON] Unexpected JSON format in {os.path.basename(json_path)}. Expected dict."); return
            
            sheets_to_write: Dict[str, pd.DataFrame] = {}
            
            # Iterar sobre las posibles claves principales en el JSON del estado del agente
            # (q_tables, visit_counts, baseline_tables, baseline_visit_counts, etc.)
            for main_table_type_key, data_dict_for_type in agent_state_data.items():
                # main_table_type_key puede ser "q_tables", "visit_counts", "baseline_tables", "baseline_visit_counts"
                if isinstance(data_dict_for_type, dict):
                    for gain_or_agent_var, table_data_list in data_dict_for_type.items():
                        # gain_or_agent_var es 'kp', 'ki', 'kd'
                        # table_data_list es la lista de diccionarios para esa tabla/ganancia
                        
                        # Construir un nombre de hoja descriptivo
                        # ej: q_tables_kp, baseline_tables_ki, baseline_visit_counts_kd
                        sheet_name_base = main_table_type_key.replace("_tables", "").replace("_counts","") # "q", "visit", "baseline", "baseline_visit"
                        sheet_name_final = f"{sheet_name_base}_{gain_or_agent_var}"[:31] # Limitar longitud

                        if isinstance(table_data_list, list) and table_data_list:
                            try:
                                df = pd.DataFrame(table_data_list)
                                if not df.empty: 
                                    sheets_to_write[sheet_name_final] = df
                                else: 
                                    self._logger.debug(f"[ResultHandler:ConvertJSON] Empty DataFrame for table '{sheet_name_final}'. Not saving sheet.")
                            except Exception as e_df: 
                                self._logger.warning(f"[ResultHandler:ConvertJSON] Could not convert table data for sheet '{sheet_name_final}' to DataFrame: {e_df}")
                        elif isinstance(table_data_list, list): 
                            self._logger.debug(f"[ResultHandler:ConvertJSON] Table data for sheet '{sheet_name_final}' is an empty list. Not saving sheet.")
                        else: 
                            self._logger.warning(f"[ResultHandler:ConvertJSON] Data for '{main_table_type_key}.{gain_or_agent_var}' (sheet '{sheet_name_final}') not a list. Not saving sheet.")
                else: 
                    self._logger.warning(f"[ResultHandler:ConvertJSON] Top-level key '{main_table_type_key}' in JSON is not a dictionary of tables. Skipping.")

            if sheets_to_write:
                writer = pd.ExcelWriter(excel_path, engine='openpyxl')
                for sheet_name in sheets_to_write:
                    df = sheets_to_write[sheet_name]
                    df.to_excel(writer, sheet_name=sheet_name, index=False); saved_sheets += 1
                writer.close() # close() guarda el archivo con openpyxl
                self._logger.info(f"[ResultHandler:ConvertJSON] Agent state converted ({saved_sheets} sheets) to {os.path.basename(excel_path)}")
            else: 
                self._logger.warning("[ResultHandler:ConvertJSON] No valid data in agent state JSON to convert to Excel sheets.")
        except FileNotFoundError: self._logger.error(f"[ResultHandler:ConvertJSON] Agent JSON file not found: {json_path}")
        except json.JSONDecodeError as e: self._logger.error(f"[ResultHandler:ConvertJSON] JSON decode error in '{os.path.basename(json_path)}': {e}")
        except OSError as e: self._logger.error(f"[ResultHandler:ConvertJSON] OS error saving Excel to {excel_path}: {e}")
        except IndexError as e_excel: self._logger.error(f"[ResultHandler:ConvertJSON] Openpyxl error saving Excel (possibly empty/corrupt internal file): {e_excel}")
        except Exception as e: self._logger.error(f"[ResultHandler:ConvertJSON] Unexpected error converting JSON to Excel: {e}", exc_info=True)
        finally:
             if writer is not None and saved_sheets > 0 and hasattr(writer, 'close') and not getattr(writer, 'closed', True): # getattr para pandas < 1.4
                 try: writer.close()
                 except: pass

    def save_summary_table(self, summary_list: List[Dict], results_folder: str):
        if not os.path.isdir(results_folder):
            self._logger.error(f"[ResultHandler:SaveSummary] Results folder '{results_folder}' not found. Skipping."); return
        if not summary_list: self._logger.warning("[ResultHandler:SaveSummary] Empty summary list. Not saving summary file."); return
        path = os.path.join(results_folder, 'episodes_summary.xlsx')
        self._logger.info(f"[ResultHandler:SaveSummary] Saving summary table to {os.path.basename(path)}")
        try: import openpyxl # type: ignore
        except ImportError: self._logger.error("[ResultHandler:SaveSummary] 'openpyxl' not found. Cannot save summary to Excel."); return
        try:
            df = pd.DataFrame(summary_list)
            if 'episode' in df.columns: df['episode'] = pd.to_numeric(df['episode'], errors='coerce').astype('Int64')
            cols = df.columns.tolist()
            preferred_order = ['episode', 'episode_time', 'total_reward', 'performance', 'termination_reason',
                               'episode_duration_s','avg_stability_score', 'w_stab_kp_cf', 'w_stab_ki_cf', 'w_stab_kd_cf',
                               'total_agent_decisions', 'final_epsilon', 'final_learning_rate', 'final_kp', 'final_ki', 'final_kd']
            ordered_cols = [c for c in preferred_order if c in cols] + sorted([c for c in cols if c not in preferred_order])
            df_ordered = df[ordered_cols]
            df_ordered.to_excel(path, index=False, engine='openpyxl')
            self._logger.info(f"[ResultHandler:SaveSummary] Summary table saved successfully.")
        except OSError as e: self._logger.error(f"[ResultHandler:SaveSummary] OS error saving summary to {path}: {e}")
        except Exception as e: self._logger.error(f"[ResultHandler:SaveSummary] Unexpected error saving summary to {path}: {e}", exc_info=True)

    def finalize(self,
                 config: Dict[str, Any],
                 vis_config: Optional[Dict[str, Any]],
                 summary_data: List[Dict],
                 all_episodes_data: List[Dict], # Normalmente vacío
                 agent: Optional['RLAgent'],
                 results_folder: str):
        if not os.path.isdir(results_folder):
            self._logger.error(f"[ResultHandler:Finalize] Results folder '{results_folder}' not found. Aborting finalization."); return
        self._logger.info(f"[ResultHandler:Finalize] --- Starting ResultHandler Finalization in: {results_folder} ---")

        # Acceder a la configuración desde la nueva estructura
        env_cfg_section = config.get('environment', {})
        sim_params_cfg = env_cfg_section.get('simulation', {})

        # 1. Guardar estado final del agente
        save_final_agent_state = env_cfg_section.get('save_agent_state', False)
        if save_final_agent_state and agent is not None:
            max_episodes = sim_params_cfg.get('max_episodes', 0) # Usar sim_params_cfg
            final_episode_index = max(0, max_episodes - 1)
            self.save_agent_state(agent, final_episode_index, results_folder)
        elif save_final_agent_state: self._logger.warning("[ResultHandler:Finalize] Agent state saving enabled, but agent is None.")
        else: self._logger.info("[ResultHandler:Finalize] Final agent state saving disabled or not applicable.")

        # 2. Convertir último estado guardado a Excel
        if self._last_agent_state_json_path and os.path.exists(self._last_agent_state_json_path):
            excel_path = os.path.join(results_folder, 'agent_state_tables_final.xlsx')
            self.convert_json_agent_state_to_excel(self._last_agent_state_json_path, excel_path)
        elif save_final_agent_state: # Solo advertir si se esperaba guardar
            self._logger.warning("[ResultHandler:Finalize] No final agent state JSON found to convert to Excel (expected if saving was on).")

        # 3. Guardar tabla de resumen
        if summary_data: self.save_summary_table(summary_data, results_folder)
        else: self._logger.info("[ResultHandler:Finalize] No summary data to save.")

        # 4. Guardar datos detallados si no se guardaron por batch (raro)
        if all_episodes_data:
            self._logger.warning("[ResultHandler:Finalize] 'all_episodes_data' contains data. Saving as a final batch.")
            max_ep = sim_params_cfg.get('max_episodes', 0) # Usar sim_params_cfg
            final_ep_idx = max(0, max_ep - 1) if max_ep > 0 else 0
            self.save_episode_batch(all_episodes_data, results_folder, final_ep_idx)

        self._logger.info("[ResultHandler:Finalize] --- ResultHandler Finalization Complete ---")