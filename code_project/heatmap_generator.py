import os
import json
import yaml
import argparse
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Necesita openpyxl para escribir Excel
try:
    import openpyxl
except ImportError:
    # openpyxl no es una dependencia estricta si solo se usa como servicio
    # El error se manejará al intentar escribir si no está instalado
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
        self.logger.info("HeatmapGenerator instance created.")

    def _extract_heatmap_data(
        self,
        detailed_data: List[Dict],
        x_var: str,
        y_var: str,
        filter_reasons: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae y concatena pares de datos (x, y) de la lista de episodios detallados.
        Aplica un filtro opcional por razones de terminación.
        Maneja NaNs y errores de tipo durante la extracción.

        Args:
            detailed_data: Lista de diccionarios, cada uno representa un episodio.
            x_var: Nombre de la clave para la variable X en los diccionarios de episodio.
            y_var: Nombre de la clave para la variable Y.
            filter_reasons: Lista opcional de 'termination_reason' para incluir episodios.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays numpy concatenados de datos X e Y válidos.
                                           Devuelve arrays vacíos si no hay datos válidos.
        """
        x_all, y_all = [], []

        if not isinstance(detailed_data, list):
            self.logger.error("Formato inválido de detailed_data: se esperaba una lista de diccionarios.")
            return np.array([]), np.array([])

        episodes = detailed_data
        if filter_reasons:
            # Asegurar que filter_reasons sea una lista de strings
            if not isinstance(filter_reasons, list) or not all(isinstance(r, str) for r in filter_reasons):
                 self.logger.warning(f"filter_termination_reason ({filter_reasons}) no es una lista de strings válida. Ignorando filtro.")
                 filter_reasons = None # Ignorar filtro si es inválido
            else:
                try:
                    original_count = len(episodes)
                    # Filtrar asegurando que cada episodio es un dict y tiene la clave
                    episodes = [
                        ep for ep in episodes
                        if isinstance(ep, dict) and ep.get('termination_reason') in filter_reasons
                    ]
                    filtered_count = len(episodes)
                    self.logger.info(
                        f"Filtrado por razones de terminación {filter_reasons}: "
                        f"{filtered_count}/{original_count} episodios mantenidos."
                    )
                except Exception as e:
                    self.logger.error(f"Error inesperado durante el filtrado de episodios: {e}", exc_info=True)
                    # Continuar sin filtrar si hay error
                    episodes = detailed_data
                    filter_reasons = None

        if not episodes:
            self.logger.warning(f"No quedan episodios tras filtrar por {filter_reasons}. No se pueden extraer datos.")
            return np.array([]), np.array([])

        # Iterar sobre los episodios (filtrados o no)
        valid_points_count = 0
        for i, ep_data in enumerate(episodes):
            if not isinstance(ep_data, dict):
                # self.logger.debug(f"Item {i} en detailed_data no es un diccionario. Omitiendo.")
                continue

            x_raw = ep_data.get(x_var)
            y_raw = ep_data.get(y_var)

            # Verificar que ambas variables existen y son listas o arrays
            if x_raw is None or y_raw is None:
                # self.logger.debug(f"Episodio {ep_data.get('episode', i)}: Faltan variables '{x_var}' o '{y_var}'.")
                continue
            if not isinstance(x_raw, (list, np.ndarray)) or not isinstance(y_raw, (list, np.ndarray)):
                # self.logger.debug(f"Episodio {ep_data.get('episode', i)}: Variables '{x_var}' o '{y_var}' no son listas/arrays.")
                continue

            # Convertir a numérico y manejar errores/NaN
            try:
                # Alinear longitudes si son diferentes (tomar la mínima)
                min_len = min(len(x_raw), len(y_raw))
                if min_len == 0:
                     continue

                x_num = pd.to_numeric(x_raw[:min_len], errors='coerce')
                y_num = pd.to_numeric(y_raw[:min_len], errors='coerce')

                # Crear máscara para puntos válidos (no NaN/inf)
                mask = np.isfinite(x_num) & np.isfinite(y_num)

                num_valid_in_ep = np.sum(mask)
                if num_valid_in_ep > 0:
                    x_all.append(x_num[mask])
                    y_all.append(y_num[mask])
                    valid_points_count += num_valid_in_ep
                # else:
                    # self.logger.debug(f"Episodio {ep_data.get('episode', i)}: No hay pares (x, y) numéricos válidos.")

            except Exception as e:
                self.logger.warning(f"Error procesando datos numéricos del episodio {ep_data.get('episode', i)} "
                                    f"para ({x_var}, {y_var}): {e}")
                continue # Saltar al siguiente episodio si hay error en conversión/manejo

        # Concatenar todos los puntos válidos
        if not x_all or not y_all:
            self.logger.warning(f"No se encontraron puntos de datos válidos para el heatmap '{y_var}' vs '{x_var}'"
                                f"{' con filtro ' + str(filter_reasons) if filter_reasons else ''}.")
            return np.array([]), np.array([])

        try:
            x_combined = np.concatenate(x_all)
            y_combined = np.concatenate(y_all)
            self.logger.info(
                f"Datos extraídos para heatmap '{y_var}' vs '{x_var}': "
                f"{len(x_combined)} pares válidos."
            )
            return x_combined, y_combined
        except Exception as e:
            self.logger.error(f"Error concatenando datos extraídos para heatmap ({x_var}, {y_var}): {e}", exc_info=True)
            return np.array([]), np.array([])


    def find_latest_simulation_data(self, results_folder: str) -> Optional[str]:
        """
        Busca el archivo `simulation_data_..._to_....json` con el número de episodio
        final más alto dentro de la carpeta de resultados especificada.

        Args:
            results_folder: Ruta a la carpeta donde buscar los archivos JSON.

        Returns:
            Ruta completa al último archivo encontrado, o None si no se encuentra ninguno
            o si ocurre un error.
        """
        latest_file: Optional[str] = None
        highest_episode_num: int = -1

        self.logger.debug(f"Buscando último archivo simulation_data en: {results_folder}")
        if not os.path.isdir(results_folder):
             self.logger.error(f"La carpeta de resultados especificada no existe: {results_folder}")
             return None

        try:
            for filename in os.listdir(results_folder):
                if filename.startswith('simulation_data_') and filename.endswith('.json'):
                    # Extraer números del nombre del archivo
                    parts = filename[:-5].split('_') # Quita .json y divide por _
                    # Espera formato simulation_data_FIRST_to_LAST
                    if len(parts) >= 4 and parts[-2] == 'to':
                        try:
                            # El último elemento debería ser el número final
                            last_episode = int(parts[-1])
                            if last_episode > highest_episode_num:
                                highest_episode_num = last_episode
                                latest_file = os.path.join(results_folder, filename)
                        except (ValueError, IndexError):
                            # Ignorar archivos que no coincidan con el formato numérico esperado
                            # self.logger.debug(f"Ignorando archivo con formato inesperado: {filename}")
                            pass # Continuar con el siguiente archivo
        except FileNotFoundError:
             self.logger.error(f"Error: Carpeta de resultados no encontrada durante la búsqueda: {results_folder}")
             return None
        except Exception as e:
            self.logger.error(f"Error inesperado buscando último archivo de datos detallados: {e}", exc_info=True)
            return None

        if latest_file:
            self.logger.info(f"Último archivo de datos detallados encontrado: {os.path.basename(latest_file)} (hasta episodio {highest_episode_num})")
        else:
            self.logger.warning(
                f"No se encontraron archivos simulation_data_*_to_*.json en {results_folder}"
            )
        return latest_file

    def generate(
        self,
        detailed_data_filepath: str,
        heatmap_configs: List[Dict],
        output_excel_filepath: str
    ):
        """
        Proceso principal para generar el archivo Excel con los datos de los heatmaps.
        Carga los datos detallados, itera sobre las configuraciones de heatmap,
        extrae datos, calcula histogramas 2D y los guarda en hojas separadas del Excel.

        Args:
            detailed_data_filepath: Ruta al archivo JSON con los datos detallados.
            heatmap_configs: Lista de diccionarios, cada uno configurando un heatmap.
                             (Proveniente de la config de visualización, filtrado por tipo='heatmap').
            output_excel_filepath: Ruta completa donde se guardará el archivo Excel.
        """
        self.logger.info(
            f"Iniciando generación de datos para heatmaps."
            f"\n  Origen Datos: {os.path.basename(detailed_data_filepath)}"
            f"\n  Salida Excel: {os.path.basename(output_excel_filepath)}"
        )

        # [1] Cargar datos detallados desde JSON
        try:
            with open(detailed_data_filepath, 'r', encoding='utf-8') as f:
                detailed_data_list = json.load(f)
            if not isinstance(detailed_data_list, list):
                # Podría ser un dict si solo hubo un episodio guardado? Asumimos lista.
                raise TypeError("El archivo de datos detallados no contiene una lista JSON.")
            self.logger.info(f"Cargados {len(detailed_data_list)} registros de episodios desde {os.path.basename(detailed_data_filepath)}.")
        except FileNotFoundError:
            self.logger.error(f"Error fatal: Archivo de datos detallados no encontrado en {detailed_data_filepath}")
            return
        except json.JSONDecodeError as e:
            self.logger.error(f"Error fatal: Fallo al decodificar JSON en {detailed_data_filepath}: {e}")
            return
        except Exception as e:
            self.logger.error(f"Error fatal cargando datos detallados desde {detailed_data_filepath}: {e}", exc_info=True)
            return

        if not detailed_data_list:
            self.logger.warning("El archivo de datos detallados está vacío. No se generarán heatmaps.")
            return

        # [2] Procesar configuraciones y generar Excel
        processed_heatmaps_count = 0
        try:
            # Usar pd.ExcelWriter para escribir en múltiples hojas
            with pd.ExcelWriter(output_excel_filepath, engine='openpyxl') as writer:
                for i, cfg in enumerate(heatmap_configs):
                    # Validar configuración básica del heatmap
                    if not isinstance(cfg, dict):
                         self.logger.warning(f"Configuración de heatmap #{i} inválida (no es diccionario). Omitiendo.")
                         continue
                    # Asegurar que es tipo heatmap y está habilitado (ya filtrado antes, pero doble check)
                    if cfg.get('type') != 'heatmap' or not cfg.get('enabled', True):
                        continue

                    x_var = cfg.get('x_variable')
                    y_var = cfg.get('y_variable')
                    out_fn = cfg.get('output_filename', f"heatmap_{y_var}_vs_{x_var}") # Usar para nombre de hoja
                    # Limitar longitud del nombre de la hoja a 31 caracteres (límite Excel) y sanear nombre
                    sheet_name = "".join(c for c in out_fn if c.isalnum() or c in ('_', '-')).strip()
                    sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
                    if not sheet_name: sheet_name = f"heatmap_{i}" # Fallback si queda vacío

                    data_cfg = cfg.get('config', {}) # Config específica del plot (bins, range, filter)
                    if not isinstance(data_cfg, dict):
                         self.logger.warning(f"Sub-configuración 'config' para heatmap '{sheet_name}' inválida. Usando defaults.")
                         data_cfg = {}

                    if not x_var or not y_var:
                        self.logger.warning(f"Configuración de heatmap '{sheet_name}' inválida: faltan 'x_variable' o 'y_variable'. Omitiendo.")
                        continue

                    self.logger.info(f"Procesando heatmap '{sheet_name}': Y='{y_var}' vs X='{x_var}'")

                    # [2a] Extraer datos X, Y con filtro opcional
                    filter_reasons = data_cfg.get('filter_termination_reason')
                    x_data, y_data = self._extract_heatmap_data(
                        detailed_data_list, x_var, y_var, filter_reasons
                    )

                    # Si no hay datos válidos tras la extracción, saltar este heatmap
                    if x_data.size == 0 or y_data.size == 0:
                        self.logger.warning(f"No se encontraron datos válidos para heatmap '{sheet_name}'. Omitiendo hoja.")
                        continue

                    # [2b] Calcular Histograma 2D
                    # Obtener parámetros del histograma desde data_cfg con defaults
                    bins = data_cfg.get('bins', 50) # Default a 50 bins si no se especifica
                    if not isinstance(bins, int) or bins <= 0:
                         self.logger.warning(f"Número de 'bins' ({bins}) inválido para heatmap '{sheet_name}'. Usando default 50.")
                         bins = 50

                    xmin, xmax = data_cfg.get('xmin'), data_cfg.get('xmax')
                    ymin, ymax = data_cfg.get('ymin'), data_cfg.get('ymax')
                    hist_range = None
                    # Definir rango solo si todos los límites son números válidos
                    if all(isinstance(v, (int, float)) and np.isfinite(v) for v in [xmin, xmax, ymin, ymax]):
                         if xmin < xmax and ymin < ymax:
                              hist_range = [[xmin, xmax], [ymin, ymax]]
                              self.logger.debug(f"Usando rango definido para histograma: X=[{xmin},{xmax}], Y=[{ymin},{ymax}]")
                         else:
                              self.logger.warning(f"Límites de rango inválidos para heatmap '{sheet_name}': xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}. Ignorando rango.")
                    elif any(v is not None for v in [xmin, xmax, ymin, ymax]):
                         # Si algunos están definidos pero no todos o no son válidos
                         self.logger.warning(f"Límites de rango incompletos o inválidos para heatmap '{sheet_name}'. Ignorando rango.")


                    try:
                        # Calcular histograma 2D
                        counts, xedges, yedges = np.histogram2d(
                            x_data, y_data, bins=bins, range=hist_range, density=False
                        )
                        self.logger.debug(f"Histograma 2D calculado para '{sheet_name}' con shape {counts.shape}")
                    except Exception as e_hist:
                        self.logger.error(f"Error calculando histograma 2D para heatmap '{sheet_name}': {e_hist}", exc_info=True)
                        continue # Saltar al siguiente heatmap

                    # [2c] Preparar DataFrames para Excel
                    # Metadata del heatmap
                    meta_rows = [
                        ('X Variable', x_var),
                        ('Y Variable', y_var),
                        ('Bins', bins),
                        ('X Min Edge', xedges[0]),
                        ('X Max Edge', xedges[-1]),
                        ('Y Min Edge', yedges[0]),
                        ('Y Max Edge', yedges[-1]),
                        ('Range Defined', 'Yes' if hist_range else 'No (Auto)'),
                        ('Filter Reasons', str(filter_reasons) if filter_reasons else 'None'),
                        ('Total Data Points Used', len(x_data))
                    ]
                    df_meta = pd.DataFrame(meta_rows, columns=['Parameter', 'Value'])

                    # Counts (histograma) como DataFrame
                    # Usar los bordes izquierdos de los bins como índices/columnas
                    df_counts = pd.DataFrame(
                        counts.T, # Transponer para que Y sea filas, X columnas
                        index=pd.Index(np.round(xedges[:-1], 4), name=f"{x_var}_bin_start"),
                        columns=pd.Index(np.round(yedges[:-1], 4), name=f"{y_var}_bin_start")
                    )
                    # Opcional: Redondear índices/columnas para legibilidad
                    # df_counts.index = df_counts.index.map(lambda x: f"{x:.3f}")
                    # df_counts.columns = df_counts.columns.map(lambda y: f"{y:.3f}")


                    # [2d] Escribir DataFrames a la hoja de Excel
                    df_meta.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                    # Añadir un título antes de la tabla de counts
                    title_cell = f"A{len(df_meta) + 2}" # Celda donde escribir el título
                    writer.sheets[sheet_name][title_cell] = f"Counts (Y={y_var} vs X={x_var})"
                    # Escribir counts debajo de la metadata y el título
                    df_counts.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        startrow=len(df_meta) + 3 # Dejar una fila para el título
                    )

                    processed_heatmaps_count += 1
                    self.logger.info(f"Heatmap '{sheet_name}' procesado y añadido a Excel.")

                if processed_heatmaps_count == 0:
                    self.logger.warning("No se procesó ningún heatmap válido para guardar en Excel.")
                else:
                    self.logger.info(f"Total heatmaps procesados y guardados en Excel: {processed_heatmaps_count}")

        except ImportError:
            self.logger.error("Error fatal: Biblioteca 'openpyxl' no encontrada o no se pudo importar. "
                              "Instalar con: pip install openpyxl")
        except OSError as e:
             self.logger.error(f"Error de OS al escribir el archivo Excel '{output_excel_filepath}': {e}")
        except Exception as e:
            self.logger.error(f"Error inesperado guardando datos de heatmap en Excel: {e}", exc_info=True)


# --- CLI Section (Mantenida para ejecución independiente si se desea) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generar datos numéricos para Heatmaps desde resultados de simulación RL."
    )
    parser.add_argument(
        "-d", "--datafile",
        help="Ruta al archivo JSON de datos detallados. Si no se especifica, busca el último en resultsfolder."
    )
    parser.add_argument(
        "-r", "--resultsfolder", required=True,
        help="Carpeta que contiene los archivos de resultados (simulation_data_*.json, sub_config_visualization.yaml)."
    )
    parser.add_argument(
        "-v", "--visconfig", default="sub_config_visualization.yaml",
        help="Nombre del archivo de configuración de visualización (YAML) dentro de resultsfolder."
    )
    parser.add_argument(
        "-o", "--output", default="data_heatmaps.xlsx",
        help="Nombre del archivo Excel de salida (se guardará en resultsfolder)."
    )
    parser.add_argument(
        "--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de logging para la salida de consola."
    )
    args = parser.parse_args()

    # --- Configurar Logger para CLI ---
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    cli_logger = logging.getLogger("HeatmapGeneratorCLI")
    cli_logger.setLevel(log_level)
    # Evitar duplicar handlers si se ejecuta múltiples veces en el mismo proceso
    if not cli_logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s")
        ch.setFormatter(formatter)
        cli_logger.addHandler(ch)
    cli_logger.propagate = False # No enviar logs al logger raíz

    # Crear instancia del generador con el logger CLI
    heatmap_gen_service = HeatmapGenerator(cli_logger)

    # --- Determinar archivo de datos detallados ---
    datafile_path = args.datafile
    if not datafile_path:
        datafile_path = heatmap_gen_service.find_latest_simulation_data(args.resultsfolder)
        if not datafile_path:
            cli_logger.error(f"No se especificó --datafile y no se encontró ningún archivo simulation_data_*.json en '{args.resultsfolder}'. Abortando.")
            exit(1)
    elif not os.path.isabs(datafile_path):
        # Si es ruta relativa, asumirla relativa a resultsfolder
        datafile_path = os.path.join(args.resultsfolder, datafile_path)

    if not os.path.exists(datafile_path):
        cli_logger.error(f"Archivo de datos detallados no encontrado en: {datafile_path}. Abortando.")
        exit(1)

    # --- Cargar configuración de visualización ---
    vis_config_path = os.path.join(args.resultsfolder, args.visconfig)
    vis_data = None
    heatmap_configs_list = []
    if not os.path.exists(vis_config_path):
        cli_logger.error(f"Archivo de configuración de visualización no encontrado en: {vis_config_path}. Abortando.")
        exit(1)
    try:
        with open(vis_config_path, "r", encoding='utf-8') as vf:
            vis_data = yaml.safe_load(vf)
        if not isinstance(vis_data, dict) or 'plots' not in vis_data:
             cli_logger.error(f"Archivo de config de visualización '{args.visconfig}' no tiene formato esperado (dict con clave 'plots'). Abortando.")
             exit(1)
        # Filtrar solo los heatmaps habilitados
        heatmap_configs_list = [
            p for p in vis_data.get("plots", [])
            if isinstance(p, dict) and p.get("type") == "heatmap" and p.get("enabled", True)
        ]
        if not heatmap_configs_list:
             cli_logger.warning(f"No se encontraron configuraciones de heatmap habilitadas en '{args.visconfig}'. No se generará archivo Excel.")
             exit(0) # Salir limpiamente si no hay nada que hacer
    except yaml.YAMLError as e:
        cli_logger.error(f"Error parseando YAML de config de visualización ({args.visconfig}): {e}")
        exit(1)
    except Exception as e:
        cli_logger.error(f"Error cargando o procesando config de visualización ({args.visconfig}): {e}")
        exit(1)

    # --- Determinar ruta de salida del Excel ---
    output_excel_path = args.output
    if not os.path.isabs(output_excel_path):
        output_excel_path = os.path.join(args.resultsfolder, output_excel_path)

    # --- Ejecutar generación ---
    cli_logger.info("Iniciando generación de datos de heatmap desde CLI...")
    heatmap_gen_service.generate(
        detailed_data_filepath=datafile_path,
        heatmap_configs=heatmap_configs_list,
        output_excel_filepath=output_excel_path
    )
    cli_logger.info("Proceso CLI completado.")