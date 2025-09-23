import pandas as pd
import json
import os

# Cargar el archivo JSON
dirname = os.path.dirname(__file__)
folder = 'results_history/20250403-0438'
file_path = os.path.join(folder, 'agent_state_episode_9999.json')
with open(file_path, "r") as f:
    data = json.load(f)

# Extraer las tablas
q_tables = data["q_tables"]
visit_counts = data["visit_counts"]

# Convertir en DataFrames
dfs_q_tables = {}
dfs_visit_counts = {}

for gain_type in q_tables:
    df_q = pd.DataFrame.from_dict(q_tables[gain_type], orient="index")
    df_q.index.name = f"{gain_type}_gain"
    dfs_q_tables[gain_type] = df_q

for gain_type in visit_counts:
    df_v = pd.DataFrame.from_dict(visit_counts[gain_type], orient="index")
    df_v.index.name = f"{gain_type}_gain"
    dfs_visit_counts[gain_type] = df_v

# Guardar en un archivo Excel
excel_path = os.path.join(folder, 'q_tables.json')
with pd.ExcelWriter(excel_path) as writer:
    for gain_type, df in dfs_q_tables.items():
        df.to_excel(writer, sheet_name=f"q_table_{gain_type}")
    for gain_type, df in dfs_visit_counts.items():
        df.to_excel(writer, sheet_name=f"visit_counts_{gain_type}")

excel_path