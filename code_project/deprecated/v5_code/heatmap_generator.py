import os
import json
import yaml
import argparse
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import openpyxl # type: ignore
except ImportError:
    # Log this warning once at module level
    logging.warning("Módulo 'openpyxl' no encontrado. La escritura a Excel fallará si se intenta.")
    pass


class HeatmapGenerator:
    """
    Servicio para generación de datos numéricos para heatmaps a partir de
    resultados detallados de simulación guardados en JSON.
    Extrae datos, calcula histogramas 2D y los exporta a un archivo Excel.
    """
    def __init__(self, logger: logging.Logger):
        """
        Inicializa el HeatmapGenerator con un logger.

        Args:
            logger: Instancia del logger configurado.
        """
        self.logger = logger
        # Usar logger pasado para confirmar instanciación
        if not isinstance(logger, logging.Logger):
             # Fallback MUY improbable si DI falla, pero por seguridad
             logging.error("HeatmapGenerator recibió un logger inválido.")
             self.logger = logging.getLogger(__name__) # Fallback logger
        self.logger.info("HeatmapGenerator instance created.")

    def _extract_heatmap_data( self, detailed_data: List[Dict], x_var: str, y_var: str, filter_reasons: Optional[List[str]] = None ) -> Tuple[np.ndarray, np.ndarray]:
        # ... (código sin cambios) ...
        x_all, y_all = [], []; valid_points_count = 0
        if not isinstance(detailed_data, list): self.logger.error("Formato inválido detailed_data: no es lista."); return np.array([]), np.array([])
        episodes = detailed_data
        if filter_reasons:
            if not isinstance(filter_reasons, list) or not all(isinstance(r, str) for r in filter_reasons): self.logger.warning(f"filter_termination_reason ({filter_reasons}) inválido. Ignorando filtro."); filter_reasons = None
            else:
                try:
                     original_count = len(episodes); episodes = [ ep for ep in episodes if isinstance(ep, dict) and ep.get('termination_reason') in filter_reasons ]; filtered_count = len(episodes);
                     self.logger.info( f"Filtrado por {filter_reasons}: {filtered_count}/{original_count} episodios mantenidos.")
                except Exception as e: self.logger.error(f"Error filtro episodios: {e}", exc_info=True); episodes = detailed_data; filter_reasons = None
        if not episodes: self.logger.warning(f"No quedan episodios tras filtro {filter_reasons}."); return np.array([]), np.array([])
        for i, ep_data in enumerate(episodes):
            if not isinstance(ep_data, dict): continue
            x_raw = ep_data.get(x_var); y_raw = ep_data.get(y_var)
            if x_raw is None or y_raw is None: continue
            if not isinstance(x_raw, (list, np.ndarray)) or not isinstance(y_raw, (list, np.ndarray)): continue
            try:
                min_len = min(len(x_raw), len(y_raw)); x_num = pd.to_numeric(x_raw[:min_len], errors='coerce'); y_num = pd.to_numeric(y_raw[:min_len], errors='coerce'); mask = np.isfinite(x_num) & np.isfinite(y_num); num_valid_in_ep = np.sum(mask)
                if num_valid_in_ep > 0: x_all.append(x_num[mask]); y_all.append(y_num[mask]); valid_points_count += num_valid_in_ep
            except Exception as e: self.logger.warning(f"Error procesando datos numéricos ep {ep_data.get('episode', i)} ({x_var}, {y_var}): {e}"); continue
        if not x_all or not y_all: self.logger.warning(f"No se encontraron puntos válidos para heatmap '{y_var}' vs '{x_var}'{' filtro ' + str(filter_reasons) if filter_reasons else ''}."); return np.array([]), np.array([])
        try:
            x_combined = np.concatenate(x_all); y_combined = np.concatenate(y_all);
            self.logger.info( f"Datos extraídos para heatmap '{y_var}' vs '{x_var}': {len(x_combined)} puntos válidos."); return x_combined, y_combined
        except Exception as e: self.logger.error(f"Error concatenando datos heatmap ({x_var}, {y_var}): {e}", exc_info=True); return np.array([]), np.array([])

    def find_latest_simulation_data(self, results_folder: str) -> Optional[str]:
        # ... (código sin cambios) ...
        latest_file: Optional[str] = None; highest_episode_num: int = -1
        self.logger.debug(f"Buscando último archivo simulation_data en: {results_folder}")
        if not os.path.isdir(results_folder): self.logger.error(f"Carpeta resultados no existe: {results_folder}"); return None
        try:
            for filename in os.listdir(results_folder):
                if filename.startswith('simulation_data_') and filename.endswith('.json'):
                    parts = filename[:-5].split('_') # Remove .json
                    if len(parts) >= 4 and parts[-2] == 'to':
                        try: last_episode = int(parts[-1]);
                        except (ValueError, IndexError): continue
                        if last_episode > highest_episode_num: highest_episode_num = last_episode; latest_file = os.path.join(results_folder, filename)
        except FileNotFoundError: self.logger.error(f"Error: Carpeta no encontrada durante búsqueda: {results_folder}"); return None
        except Exception as e: self.logger.error(f"Error buscando último archivo datos: {e}", exc_info=True); return None
        if latest_file: self.logger.info(f"Último archivo datos encontrado: {os.path.basename(latest_file)} (ep {highest_episode_num})")
        else: self.logger.warning(f"No se encontraron archivos simulation_data_*_to_*.json en {results_folder}")
        return latest_file

    def generate( self, detailed_data_filepath: str, heatmap_configs: List[Dict], output_excel_filepath: str ):
        # ... (código sin cambios) ...
        self.logger.info(f"Iniciando generación datos heatmaps.\n  Origen: {os.path.basename(detailed_data_filepath)}\n  Salida: {os.path.basename(output_excel_filepath)}")
        try:
            with open(detailed_data_filepath, 'r', encoding='utf-8') as f: detailed_data_list = json.load(f)
            if not isinstance(detailed_data_list, list): raise TypeError("Archivo datos detallados no contiene lista JSON.")
            self.logger.info(f"Cargados {len(detailed_data_list)} registros episodios desde {os.path.basename(detailed_data_filepath)}.")
        except FileNotFoundError: self.logger.error(f"Error fatal: Archivo datos detallados no encontrado: {detailed_data_filepath}"); return
        except json.JSONDecodeError as e: self.logger.error(f"Error fatal: Fallo decodificando JSON: {detailed_data_filepath}: {e}"); return
        except Exception as e: self.logger.error(f"Error fatal cargando datos detallados: {detailed_data_filepath}: {e}", exc_info=True); return
        if not detailed_data_list: self.logger.warning("Archivo datos detallados vacío. No se generarán heatmaps."); return

        processed_heatmaps_count = 0
        try:
             # Intentar importar openpyxl justo antes de usarlo
             import openpyxl # type: ignore
        except ImportError:
             self.logger.error("Error fatal: Biblioteca 'openpyxl' no encontrada. Instalar: pip install openpyxl")
             return # No continuar si no se puede escribir Excel

        try:
            with pd.ExcelWriter(output_excel_filepath, engine='openpyxl') as writer:
                for i, cfg in enumerate(heatmap_configs):
                    if not isinstance(cfg, dict): self.logger.warning(f"Config heatmap #{i} inválida. Omitiendo."); continue
                    # Filtrar por tipo y habilitado
                    if cfg.get('type') != 'heatmap' or not cfg.get('enabled', True): continue
                    x_var = cfg.get('x_variable'); y_var = cfg.get('y_variable')
                    # Crear nombre de hoja válido
                    sheet_name_base = cfg.get('output_filename', f"heatmap_{y_var}_vs_{x_var}")
                    sheet_name = "".join(c for c in sheet_name_base if c.isalnum() or c in ('_', '-')).strip()
                    sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name # Limitar longitud
                    sheet_name = sheet_name if sheet_name else f"heatmap_{i}" # Fallback si queda vacío

                    data_cfg = cfg.get('config', {});
                    if not isinstance(data_cfg, dict): self.logger.warning(f"Sub-config 'config' para '{sheet_name}' inválida. Usando defaults."); data_cfg = {}
                    if not x_var or not y_var: self.logger.warning(f"Config heatmap '{sheet_name}': faltan 'x_variable' o 'y_variable'. Omitiendo."); continue

                    self.logger.info(f"Procesando heatmap '{sheet_name}': Y='{y_var}' vs X='{x_var}'")
                    filter_reasons = data_cfg.get('filter_termination_reason')
                    x_data, y_data = self._extract_heatmap_data( detailed_data_list, x_var, y_var, filter_reasons )
                    if x_data.size == 0 or y_data.size == 0: self.logger.warning(f"No se encontraron datos válidos para heatmap '{sheet_name}'. Omitiendo hoja."); continue

                    bins = data_cfg.get('bins', 50);
                    if not isinstance(bins, int) or bins <= 0: self.logger.warning(f"Bins ({bins}) inválido para '{sheet_name}'. Usando 50."); bins = 50
                    xmin, xmax = data_cfg.get('xmin'), data_cfg.get('xmax'); ymin, ymax = data_cfg.get('ymin'), data_cfg.get('ymax'); hist_range = None
                    if all(isinstance(v, (int, float)) and np.isfinite(v) for v in [xmin, xmax, ymin, ymax]):
                         if xmin < xmax and ymin < ymax: hist_range = [[xmin, xmax], [ymin, ymax]]; self.logger.debug(f"Rango definido: X=[{xmin},{xmax}], Y=[{ymin},{ymax}]")
                         else: self.logger.warning(f"Límites rango inválidos {xmin},{xmax},{ymin},{ymax}. Ignorando.")
                    elif any(v is not None for v in [xmin, xmax, ymin, ymax]): self.logger.warning(f"Límites rango incompletos/inválidos. Ignorando.")

                    try:
                        counts, xedges, yedges = np.histogram2d( x_data, y_data, bins=bins, range=hist_range, density=False )
                        self.logger.debug(f"Histograma 2D calculado '{sheet_name}' shape {counts.shape}")
                    except Exception as e_hist: self.logger.error(f"Error calculando histograma 2D '{sheet_name}': {e_hist}", exc_info=True); continue

                    # Crear DataFrames para Excel
                    meta_rows = [ ('X Variable', x_var), ('Y Variable', y_var), ('Bins', bins),
                                  ('X Min Edge', xedges[0]), ('X Max Edge', xedges[-1]),
                                  ('Y Min Edge', yedges[0]), ('Y Max Edge', yedges[-1]),
                                  ('Range Defined', 'Yes' if hist_range else 'No (Auto)'),
                                  ('Filter Reasons', str(filter_reasons) if filter_reasons else 'None'),
                                  ('Total Data Points Used', len(x_data)) ]
                    df_meta = pd.DataFrame(meta_rows, columns=['Parameter', 'Value'])
                    # Transponer counts para que X sea índice y Y columnas (más intuitivo en Excel)
                    df_counts = pd.DataFrame( counts.T,
                                              index=pd.Index(np.round(yedges[:-1], 4), name=f"{y_var}_bin_start"),
                                              columns=pd.Index(np.round(xedges[:-1], 4), name=f"{x_var}_bin_start") )

                    # Escribir en Excel
                    df_meta.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                    title_cell = f"A{len(df_meta) + 2}" # Celda para el título de la tabla de counts
                    writer.sheets[sheet_name][title_cell] = f"Counts (Y={y_var} vs X={x_var})"
                    df_counts.to_excel( writer, sheet_name=sheet_name, startrow=len(df_meta) + 3 ) # Dejar fila para título

                    processed_heatmaps_count += 1; self.logger.info(f"Heatmap '{sheet_name}' añadido a Excel.")

                if processed_heatmaps_count == 0: self.logger.warning("No se procesó ningún heatmap válido para guardar en Excel.")
                else: self.logger.info(f"Total heatmaps procesados y guardados en Excel: {processed_heatmaps_count}")

        except OSError as e: self.logger.error(f"Error de OS escribiendo Excel '{output_excel_filepath}': {e}")
        except Exception as e: self.logger.error(f"Error inesperado guardando datos heatmap en Excel: {e}", exc_info=True)


# --- CLI Section (Sin cambios) ---
if __name__ == "__main__":
    # ... (código sin cambios) ...
    parser = argparse.ArgumentParser( description="Generar datos numéricos para Heatmaps desde resultados RL." )
    parser.add_argument( "-d", "--datafile", help="Ruta al JSON detallado. Si no, busca último en resultsfolder." )
    parser.add_argument( "-r", "--resultsfolder", required=True, help="Carpeta resultados (simulation_data_*.json, sub_config_visualization.yaml)." )
    parser.add_argument( "-v", "--visconfig", default="sub_config_visualization.yaml", help="Nombre archivo config visualización (YAML) en resultsfolder." )
    parser.add_argument( "-o", "--output", default="data_heatmaps.xlsx", help="Nombre archivo Excel salida (en resultsfolder)." )
    parser.add_argument( "--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Nivel logging consola." )
    args = parser.parse_args()
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO); cli_logger = logging.getLogger("HeatmapGeneratorCLI"); cli_logger.setLevel(log_level)
    if not cli_logger.hasHandlers(): ch = logging.StreamHandler(); ch.setLevel(log_level); formatter = logging.Formatter("[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"); ch.setFormatter(formatter); cli_logger.addHandler(ch)
    cli_logger.propagate = False; heatmap_gen_service = HeatmapGenerator(cli_logger); datafile_path = args.datafile
    if not datafile_path: datafile_path = heatmap_gen_service.find_latest_simulation_data(args.resultsfolder);
    if not datafile_path: cli_logger.error(f"No se especificó --datafile y no se encontró archivo simulation_data_*.json en '{args.resultsfolder}'. Abortando."); exit(1)
    elif not os.path.isabs(datafile_path): datafile_path = os.path.join(args.resultsfolder, datafile_path)
    if not os.path.exists(datafile_path): cli_logger.error(f"Archivo datos detallados no encontrado: {datafile_path}. Abortando."); exit(1)
    vis_config_path = os.path.join(args.resultsfolder, args.visconfig); vis_data = None; heatmap_configs_list = []
    if not os.path.exists(vis_config_path): cli_logger.error(f"Archivo config visualización no encontrado: {vis_config_path}. Abortando."); exit(1)
    try:
        with open(vis_config_path, "r", encoding='utf-8') as vf: vis_data = yaml.safe_load(vf)
        if not isinstance(vis_data, dict) or 'plots' not in vis_data: cli_logger.error(f"Archivo config vis '{args.visconfig}' formato inválido. Abortando."); exit(1)
        heatmap_configs_list = [ p for p in vis_data.get("plots", []) if isinstance(p, dict) and p.get("type") == "heatmap" and p.get("enabled", True) ]
        if not heatmap_configs_list: cli_logger.warning(f"No se encontraron configs heatmap habilitadas en '{args.visconfig}'. No se generará Excel."); exit(0)
    except yaml.YAMLError as e: cli_logger.error(f"Error parseando YAML config vis ({args.visconfig}): {e}"); exit(1)
    except Exception as e: cli_logger.error(f"Error cargando/procesando config vis ({args.visconfig}): {e}"); exit(1)
    output_excel_path = args.output;
    if not os.path.isabs(output_excel_path): output_excel_path = os.path.join(args.resultsfolder, output_excel_path)
    cli_logger.info("Iniciando generación datos heatmap desde CLI..."); heatmap_gen_service.generate( datafile_path, heatmap_configs_list, output_excel_path ); cli_logger.info("Proceso CLI completado.")