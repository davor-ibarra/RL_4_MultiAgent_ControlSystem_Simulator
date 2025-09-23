# V13_code/external_visualization/external_visualization.py
import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")

import yaml
import pandas as pd
import numpy as np

# --- 1) Rutas hardcodeadas ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # V13_code/
DATA_DIR = PROJECT_ROOT / "results_history_CartPole" / "20250825-0536"  # <-- hardcode
CONFIG_YAML = HERE / "external_visualization_config_CartPole.yaml"               # <-- hardcode
OUTPUT_DIR = HERE  # guarda los .png donde se corre este script

# --- 2) Importar tus clases del proyecto ---
# Añadimos V13_code al PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.plotting.matplotlib_plot_generator import MatplotlibPlotGenerator  # type: ignore

# --- 3) Wrapper para forzar carga de datos desde DATA_DIR pero guardar en OUTPUT_DIR ---
class ExternalMatplotlibPlotGenerator(MatplotlibPlotGenerator):
    def __init__(self, data_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ext_data_dir = Path(data_dir)

    # Sobrescribe carga del summary: ignora 'output_root_path_plot'
    def _load_summary_data(self, _ignored_output_path: str):
        summary_path = self._ext_data_dir / "episodes_summary_data.xlsx"
        if not summary_path.exists():
            logging.error(f"[external] Summary no encontrado: {summary_path}")
            return None
        try:
            return pd.read_excel(summary_path)
        except Exception as e:
            logging.exception(f"[external] Error leyendo summary: {summary_path} -> {e}")
            return None

    # Sobrescribe carga de detallados desde JSON: ignora 'output_root_path_plot'
    def _load_detailed_data(self, _ignored_output_path: str):
    # Cargar lista de archivos parcelados
        try:
            files = [p for p in self._ext_data_dir.iterdir()
                    if p.name.startswith("simulation_data_ep_") and p.suffix == ".json"]
            if not files:
                logging.warning(f"[external] No hay JSON detallados en: {self._ext_data_dir}")
                return None
            # Ordenar por número de episodio inicial si está en el nombre
            try:
                files.sort(key=lambda p: int(p.stem.split('_ep_')[-1].split('_to_')[0]))
            except Exception:
                pass
        except Exception as e:
            logging.error(f"[external] Error listando JSON en {self._ext_data_dir}: {e}")
            return None

        raw_eps: List[dict] = []
        for fp in files:
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    raw_eps.extend([d for d in raw if isinstance(d, dict)])
            except Exception as e:
                logging.error(f"[external] Error cargando {fp.name}: {e}")

        if not raw_eps:
            logging.warning("[external] No se cargaron episodios válidos desde los JSON.")
            return None

        # Alinear por timestep como hace HeatmapGenerator
        all_ep_dfs: List[pd.DataFrame] = []
        for idx, ep in enumerate(raw_eps):
            ep_id = ep.get("episode")
            if isinstance(ep_id, list):
                ep_id = ep_id[0] if ep_id else f"ep_idx_{idx}"
            elif ep_id is None:
                ep_id = f"ep_idx_{idx}"

            time_vals = ep.get("time")
            if not isinstance(time_vals, list) or not time_vals:
                # igual que en HeatmapGenerator: skip si no hay 'time'
                continue

            n = len(time_vals)
            ref_idx = pd.RangeIndex(n)

            tmp: Dict[str, pd.Series] = {"time": pd.Series(time_vals, index=ref_idx)}
            for key, vals in ep.items():
                if key == "time":
                    continue
                if isinstance(vals, list):
                    if len(vals) == n:
                        tmp[key] = pd.Series(vals, index=ref_idx)
                    else:
                        # Alinear con padding NA
                        s = pd.Series(index=ref_idx, dtype=object)
                        m = min(len(vals), n)
                        s.iloc[:m] = vals[:m]
                        tmp[key] = s
                elif vals is not None:
                    # valor escalar → replicar
                    tmp[key] = pd.Series([vals]*n, index=ref_idx)

            try:
                df_ep = pd.DataFrame(tmp)
                df_ep["episode"] = ep_id
                if "termination_reason" not in df_ep.columns:
                    df_ep["termination_reason"] = pd.NA
                all_ep_dfs.append(df_ep)
            except Exception as e:
                logging.error(f"[external] Error creando DF de ep {ep_id}: {e}")

        if not all_ep_dfs:
            logging.warning("[external] No se pudieron construir DataFrames de episodios.")
            return None

        df = pd.concat(all_ep_dfs, ignore_index=True)

        # Coerciones numéricas comunes
        for c in ("time", "step", "level", "angle", "cart_position", "total_reward"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        logging.info(f"[external] Detailed JSON rows: {len(df)} | cols: {sorted(df.columns.tolist())[:25]}")
        return df
    
    def _load_heatmap_data(self, _ignored_output_root_path: str, plot_config_data: Dict) -> Optional[pd.DataFrame]:
        df = self._load_detailed_data(None)
        if df is None or df.empty:
            return None

        cfg = plot_config_data.get("config", {})
        x_var = plot_config_data["x_variable"]
        y_var = plot_config_data["y_variable"]
        val_var = plot_config_data.get("value_variable", x_var)
        agg = str(cfg.get("aggregation", "count")).lower()
        bins = int(cfg.get("bins", 50))

        # Filtros opcionales
        filt_terms = cfg.get("filter_termination_reason")
        if filt_terms and "termination_reason" in df.columns:
            df = df[df["termination_reason"].isin(filt_terms)]

        # --- Chequeo de columnas ---
        if x_var not in df.columns or y_var not in df.columns:
            logging.warning(f"[external] Heatmap: columnas no encontradas x='{x_var}', y='{y_var}'. Disponibles: {list(df.columns)[:15]}...")
            return None

        # Limites opcionales; si no, toma rango de datos válidos
        x_series = pd.to_numeric(df[x_var], errors="coerce")
        y_series = pd.to_numeric(df[y_var], errors="coerce")
        v_series = pd.to_numeric(df[val_var], errors="coerce") if val_var in df.columns else pd.Series(np.ones(len(df)), index=df.index)

        x_valid = x_series.dropna(); y_valid = y_series.dropna(); v_valid = v_series.dropna()

        if x_valid.empty or y_valid.empty:
            logging.warning(f"[external] Heatmap: series vacías tras coerción. x='{x_var}' o y='{y_var}'.")
            return None

        xmin_cfg = cfg.get("xmin"); xmax_cfg = cfg.get("xmax")
        ymin_cfg = cfg.get("ymin"); ymax_cfg = cfg.get("ymax")

        xmin = float(x_valid.min()) if xmin_cfg is None else float(xmin_cfg)
        xmax = float(x_valid.max()) if xmax_cfg is None else float(xmax_cfg)
        ymin = float(y_valid.min()) if ymin_cfg is None else float(ymin_cfg)
        ymax = float(y_valid.max()) if ymax_cfg is None else float(ymax_cfg)

        # Primer recorte
        mask = (x_series >= xmin) & (x_series <= xmax) & (y_series >= ymin) & (y_series <= ymax)
        dfc = pd.DataFrame({"_x": x_series[mask], "_y": y_series[mask], "_v": v_series[mask]}).dropna()

        # --- Fallback automático si quedó vacío por límites demasiado estrictos ---
        if dfc.empty:
            logging.warning(f"[external] Heatmap: 0 puntos dentro de límites (x:[{xmin},{xmax}] y:[{ymin},{ymax}]). Usando rangos automáticos de datos.")
            xmin, xmax = float(x_valid.min()), float(x_valid.max())
            ymin, ymax = float(y_valid.min()), float(y_valid.max())
            mask = (x_series >= xmin) & (x_series <= xmax) & (y_series >= ymin) & (y_series <= ymax)
            dfc = pd.DataFrame({"_x": x_series[mask], "_y": y_series[mask], "_v": v_series[mask]}).dropna()
            if dfc.empty:
                logging.warning("[external] Heatmap: sigue vacío incluso con rangos automáticos. Abortando.")
                return None

        # Bins y centros
        bins = max(1, int(cfg.get("bins", 50)))
        x_edges = np.linspace(xmin, xmax, bins + 1)
        y_edges = np.linspace(ymin, ymax, bins + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

        # Índices de bin
        x_idx = np.clip(np.digitize(dfc["_x"].to_numpy(), x_edges) - 1, 0, bins - 1)
        y_idx = np.clip(np.digitize(dfc["_y"].to_numpy(), y_edges) - 1, 0, bins - 1)

        tmp = pd.DataFrame({"xb": x_idx, "yb": y_idx, "val": dfc["_v"].to_numpy()})

        # Agregación
        aggfunc = {
            "count": "size",
            "sum": "sum",
            "mean": "mean",
            "median": "median",
            "max": "max",
            "min": "min",
        }.get(agg, "size")

        if aggfunc == "size":
            grid = tmp.pivot_table(index="yb", columns="xb", values="val", aggfunc="size", fill_value=0)
        else:
            grid = tmp.pivot_table(index="yb", columns="xb", values="val", aggfunc=aggfunc, fill_value=0)

        # Reindex a grilla completa
        grid = grid.reindex(index=range(bins), columns=range(bins), fill_value=0)

        # Etiquetas = centros de bin + nombres de ejes (lo usa _generate_heatmap_plot)
        grid.index = pd.Index(y_centers, name=plot_config_data.get("y_variable"))
        grid.columns = pd.Index(x_centers, name=plot_config_data.get("x_variable"))

        logging.info(f"[external] Heatmap grid listo: {grid.shape} (min={np.nanmin(grid.values)}, max={np.nanmax(grid.values)})")

        return grid

def main():
    # --- 4) Logging simple ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("[external] Iniciando visualización externa")
    logging.info(f"[external] DATA_DIR = {DATA_DIR}")
    logging.info(f"[external] CONFIG_YAML = {CONFIG_YAML}")
    logging.info(f"[external] OUTPUT_DIR = {OUTPUT_DIR}")

    # --- 5) Cargar YAML de config (hardcode) ---
    if not CONFIG_YAML.exists():
        raise FileNotFoundError(f"No se encontró el YAML de config: {CONFIG_YAML}")
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        vis_cfg = yaml.safe_load(f) or {}
    plots: List[Dict[str, Any]] = [p for p in vis_cfg.get("plots", []) if p.get("enabled", True)]

    # Añade índice interno para nombres por defecto en tu generator
    for i, p in enumerate(plots, 1):
        p["_internal_plot_index"] = i

    # --- 6) Instanciar el generator con wrapper de datos ---
    plot_gen = ExternalMatplotlibPlotGenerator(data_dir=DATA_DIR)

    # --- 7) Generar cada figura guardando en OUTPUT_DIR ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for p in plots:
        try:
            plot_gen.generate_plot(p, str(OUTPUT_DIR))  # <- guarda aquí
            ok += 1
        except Exception as e:
            logging.exception(f"[external] Falló plot '{p.get('name') or p.get('output_filename') or p.get('type')}' -> {e}")
            fail += 1

    logging.info(f"[external] Listo. Éxitos={ok}, Fallos={fail}")

if __name__ == "__main__":
    main()
