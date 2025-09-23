import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.numpy_encoder import NumpyEncoder
import heatmap_generator


class ResultHandler:
    """
    Servicio para centralizar todo el guardado de resultados de la simulación:
    - Carpetas de resultados
    - Batches de episodios
    - Metadata
    - Estado final del agente
    - Conversión a Excel
    - Tablas resumen
    - Datos para heatmaps
    """

    def __init__(self):
        self._last_agent_state_json: Optional[str] = None

    def setup_results_folder(self, base_results_folder: str = 'results_history') -> str:
        """
        Crea la carpeta timestamped donde se guardarán todos los archivos de esta corrida.
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        project_root = os.path.dirname(os.path.abspath(__file__))
        logging.info(f"[ResultHandler] Project folder: {project_root}")
        results_folder = os.path.join(project_root, base_results_folder, timestamp)
        try:
            os.makedirs(results_folder, exist_ok=True)
            logging.info(f"[ResultHandler] Results folder: {results_folder}")
            return results_folder
        except OSError as e:
            logging.error(f"[ResultHandler] No se pudo crear carpeta {results_folder}: {e}")
            raise

    def save_episode_batch(self,
                           batch_data: List[Dict],
                           results_folder: str,
                           last_episode: int):
        """
        Guarda un lote de episodios en JSON nombrado con el rango de episodios.
        """
        if not batch_data:
            logging.warning("[ResultHandler] Lote vacío, no se guarda.")
            return
        try:
            first = batch_data[0].get('episode', '0')
            filename = f"simulation_data_{first}_to_{last_episode}.json"
            path = os.path.join(results_folder, filename)
            with open(path, 'w') as f:
                json.dump(batch_data, f, cls=NumpyEncoder, indent=2)
            logging.info(f"[ResultHandler] Saved batch {first}–{last_episode} → {path}")
        except Exception as e:
            logging.error(f"[ResultHandler] Error guardando batch: {e}", exc_info=True)

    def save_metadata(self, metadata: Dict[str, Any], results_folder: str):
        """
        Guarda metadata.json con la configuración y detalles de la simulación.
        """
        try:
            path = os.path.join(results_folder, 'metadata.json')
            with open(path, 'w') as f:
                json.dump(metadata, f, cls=NumpyEncoder, indent=4)
            logging.info(f"[ResultHandler] Metadata saved → {path}")
        except Exception as e:
            logging.error(f"[ResultHandler] Error guardando metadata: {e}", exc_info=True)

    def save_agent_state(self,
                         agent: Any,
                         episode: int,
                         results_folder: str) -> Optional[str]:
        """
        Guarda el estado completo del agente (Q‑tables, contadores, baseline) en JSON.
        """
        if not hasattr(agent, 'get_agent_state_for_saving'):
            logging.error("[ResultHandler] El agente no implementa get_agent_state_for_saving()")
            return None

        try:
            state = agent.get_agent_state_for_saving()
            filename = f"agent_state_episode_{episode}.json"
            path = os.path.join(results_folder, filename)
            with open(path, 'w') as f:
                json.dump(state, f, cls=NumpyEncoder, indent=2)
            logging.info(f"[ResultHandler] Agent state saved → {path}")
            self._last_agent_state_json = path
            return path
        except Exception as e:
            logging.error(f"[ResultHandler] Error guardando estado agente: {e}", exc_info=True)
            return None

    def convert_json_agent_state_to_excel(self,
                                          json_path: str,
                                          excel_path: str):
        """
        Convierte el último JSON de estado de agente a un Excel con hojas separadas
        para Q‑tables, visit_counts y baseline_tables.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Q‑tables
                for gain, table in data.get('q_tables', {}).items():
                    if table:
                        df = pd.DataFrame(table)
                        df.to_excel(writer, sheet_name=f"q_{gain}", index=False)
                # Visit counts
                for gain, table in data.get('visit_counts', {}).items():
                    if table:
                        df = pd.DataFrame(table)
                        df.to_excel(writer, sheet_name=f"visits_{gain}", index=False)
                # Baseline
                for gain, table in data.get('baseline_tables', {}).items():
                    if table:
                        df = pd.DataFrame(table)
                        df.to_excel(writer, sheet_name=f"baseline_{gain}", index=False)
            logging.info(f"[ResultHandler] Agent state Excel → {excel_path}")
        except Exception as e:
            logging.error(f"[ResultHandler] Error convirtiendo JSON a Excel: {e}", exc_info=True)

    def save_summary_table(self,
                           summary_list: List[Dict],
                           path: str):
        """
        Guarda la tabla de resumen de episodios en Excel.
        """
        if not summary_list:
            logging.warning("[ResultHandler] Resumen vacío, no se guarda.")
            return
        try:
            df = pd.DataFrame(summary_list)
            df.to_excel(path, index=False, engine='openpyxl')
            logging.info(f"[ResultHandler] Summary Excel → {path}")
        except Exception as e:
            logging.error(f"[ResultHandler] Error guardando resumen: {e}", exc_info=True)

    def finalize(self,
                 config: Dict[str, Any],
                 summary_data: List[Dict],
                 all_episodes_data: List[Dict],
                 agent: Any,
                 results_folder: str):
        """
        Orquesta el guardado final:
          1) Estado final del agente
          2) Conversión a Excel
          3) Tabla resumen
          4) Datos de heatmaps
        """
        sim_cfg = config.get('simulation', {})
        env_cfg = config.get('environment', {})

        # [1] Estado agente final
        if sim_cfg.get('save_agent_state', False):
            ep_final = env_cfg.get('max_episodes', 1) - 1
            self.save_agent_state(agent, ep_final, results_folder)

        # [2] JSON → Excel del agente
        if self._last_agent_state_json:
            excel_path = os.path.join(results_folder, 'agent_state_tables.xlsx')
            self.convert_json_agent_state_to_excel(self._last_agent_state_json, excel_path)

        # [3] Resumen
        summary_path = os.path.join(results_folder, 'summary.xlsx')
        self.save_summary_table(summary_data, summary_path)

        # [4] Heatmap data
        heatmap_generator()