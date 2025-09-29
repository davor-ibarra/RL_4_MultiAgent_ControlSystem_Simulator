# Q-Learning for On-Line PID Controller Tuning in Continuous Dynamic Systems: An Interpretable Framework for Exploring Multi-Agent System

**Autores:** Davor Ibarra-Pérez, Sergio García-Nieto and Javier Sanchis Saez.

**Journal:** *Mathematics | DOI pendiente*

## 1. Descripción del artículo
Se propone un framework interpretable para sintonía en línea de PID mediante Q-Learning tabular multi-agente: tres agentes (para $K_p, K_i, K_d$​) actúan con baja observabilidad, viendo solo su ganancia discretizada y eligiendo en intervalos de decisión $\Delta T_{\mathrm{dec}}​ entre bajar/mantener/subir en pasos $\Delta k$. El aprendizaje es sin modelo y se guía por una recompensa global que combina: base gaussiana sobre variables de control, penalización temporal, penalización al cambio de control $P_u=(u_t-u_{t-1})^2$, bono por banda operativa y bono de objetivo al estabilizar, acumulada en cada intervalo antes de actualizar Q con Bellman TD.

La validación se realiza en dos bancos no lineales: WaterTank (1er orden) y CartPole (2º orden). Los agentes convergen a combinaciones estabilizantes y muestran patrones coherentes con la dinámica: en WaterTank predominan $K_p$​ altos con $K_i, K_d$​ bajos; en CartPole, $K_p, K_i$​ altos y $K_d$ intermedio. La estructura de recompensas resulta decisiva para guiar conductas (p. ej., $P_u$​ evita sobre-ponderar $K_d$​; con solo error angular en CartPole se estabiliza el péndulo y se frena el carro gracias a componentes de velocidad/bandas/bonus). El enfoque es simple, trazable e interpretable, útil para comprender trayectorias de exploración y políticas aprendidas (tablas Q), y establece principios de diseño para esquemas híbridos PID-RL. Limitaciones: bancos idealizados, sensibilidad a hiperparámetros, y compromiso resolución-eficiencia por discretización y cadencia de decisión.

## 2. Alcance de esta sección del repositorio
Esta sección funciona como respaldo curado de resultados del paper: contiene datos y resúmenes de las simulaciones para CartPole y WaterTank, junto a metadatos que registran la configuración completa de cada simulación.
**Qué puede hacer el lector aquí:**
- **Inspeccionar outputs resumidos** como tablas agregadas, figuras y estados de agente por hitos de entrenamiento.
- **Verificar configuraciones** de simulación en los **metadatos** asociados a cada sistema dinámico.
- **Solicitar JSON históricos de gran tamaño** no incluidos en el repositorio (ver sección 6).

## 3. Contenidos
```
readme.md                          # Este archivo
data/
  results_CartPole/
    20250817-0804/                 # Resumen de resultados + figuras de CartPole
  results_WaterTank/
    20250813-1220/                 # Resumen de resultados + figuras de WaterTank
```
Cada carpeta de resultados incluye:
- `metadata_simulation_run.json` (configuración completa de la simulación).
- Tablas resumen (`episodes_summary_data.xlsx`, `data_heatmaps.xlsx`).
- Estados/snapshots del agente (`agent_state_ep_*.json`, `agent_state_final_tables.xlsx`).
- Figuras clave (`fig*_*.png`).

## 4. Reproducción de resultados
**Requisitos:**
```
pip install -r requirements.txt
```
**Instrucciones:**
1. **Obtener el código:** Descarga/clonéa el repositorio *code_project* (repo general del proyecto).
2. **Identificar parámetros utilizados:** 
	1. En esta misma carpeta *mathematics_1_Q-PID* encontrar el archivo de metadata correspondiente:
	    `data/<ENV>/results_<ENV>/<TIMESTAMP>/metadata_simulation_run.json` 
	2. Abrir e identificar los valores de los parámetros de la llave `main_config` (p. ej., `episodes`, `dt`, `decision_interval`, `epsilon/*`, `alpha/*`, `gamma`, rangos/bins, criterios de parada, etc).
3. **Configurar parámetros:** En *code_project*, sustituye los parámetros del `main_config` identificado en el archivo de configuración del entorno correspondiente, por ejemplo:
```
config/config_CartPole.yaml
config/config_WaterTank.yaml
```
4. **Configurar parámetros:** Ajusta qué guardar y cómo visualizar de la misma forma en:
```
config/sub_config_data_save_<ENV>.yaml
config/sub_config_visualization_<ENV>.yaml
```
5. **Ejecutar la simulación:** Desde la raíz de *code_project* y modificando el archivo `super_config.yaml` al sistema dinámico correspondiente, ejecuta:
```
python main.py
```
6. **Verificar outputs:** Los resultados y figuras se generarán en la ruta de salida definida por tu configuración.  Además, los metadatos de la simulación producida deben reflejar los parámetros utilizados (verificar trazabilidad 1:1 con `metadata_simulation_run.json`).

*** Considerar que al intentar reproducir los resultados, estos no serán exactamente iguales, ya que es un proceso dinámico de entrenamiento, sin embargo serán lo suficientemente similares y congruentes con las conclusiones obtenidas.

## 5.Salidas esperadas al ejecutar simulación
- **Carpeta de resultados con timestamp** -> Se crea automáticamente en la dirección: `/<output_root>/<YYYYMMDD-HHMM>/`.
- **Metadatos de la corrida** -> Al iniciar la simulación se guarda `metadata_simulation_run.json` - incluye versión del framework, timestamp, ruta de salida, versión de Python y **todas** las configuraciones usadas (main, visualización, logging y directivas de datos).
- **Datos detallados por lotes de episodios** -> Los nombres se almacenan y registran según el rango de episodios seleccionado, en la forma: `simulation_data_ep_<i>_to_<j>.json` compuesto por listas de episodios con series temporales alineadas (p. ej., `time`, variables del sistema, `termination_reason`, etc.).
- **Resumen por episodio (tabla agregada)** -> Al finalizar la simulación, se escribe `episodes_summary_data.xlsx` que contiene una hoja única con filas por episodio y métricas resúmenes (e.g., recompensa total, duración, performance, estabilidad media, y columnas directas configuradas).
- **Estado del agente (opcional, si está habilitado)**  -> En función a la cantidad de episodios asignados en la configuración, se registra una snapshot `agent_state_ep_<N>.json` del agente (Q-tables, contadores de visita, etc.) guardado con el índice del último episodio ejecutado. Adicionalmente, se convierte el último a un archivo Excel `agent_state_final_tables.xlsx` para facilitar su inspección.
- **Datos preprocesados para heatmaps (si visualización habilitada)** -> Se almacena la suma de trayectorias discretizadas de las variables en `data_heatmaps.xlsx`, donde cada hoja contiene la grilla binned (centros de bins como índices/columnas) del valor agregado seleccionado.
- **Figuras** -> Archivos de imagen (por defecto `.png`) generados por `MatplotlibPlotGenerator` según las entradas de `visualization.plots` (tipo, fuente `summary`/`detailed`/`heatmap`, nombre de salida opcional).
    - Para *heatmaps*, las figuras consumen las hojas de `data_heatmaps.xlsx`.
    - Para plots de *resumen* o *detallados*, los datos se cargan desde `episodes_summary_data.xlsx` o desde los `simulation_data_ep_*.json`, respectivamente.
    - El nombre del archivo puede fijarse en `output_filename`; si no, se usa `plot_<type>_<index>.png`.
> Nota: los nombres concretos de figuras dependen de la configuración de visualización (p. ej., `fig1_*.png`). El sistema garantiza trazabilidad vía `metadata_simulation_run.json` y las rutas de salida.

## 6. JSON de historial por lotes de episodios (bajo solicitud)
**¿Qué es?** -> Historial completo de las simulaciones asignadas al archivo de configuración, tanto como del estado del sistema, variables de entrenamiento y métricas agregadas. (Ver **Ejemplo** para conocer la estructura esperada)
**¿Por qué no está en el repo?** -> Los archivos `simulation_data_ep_<i>_to_<j>.json` no se suben por tamaño elevado; si requiere el historial asociado a la publicación, solicítelo por correo (ver punto 9) indicando sistema dinámico, rango de episodios y propósitos de uso.

**Ejemplo**:
```json
[
  {
    "episode": [0, 0, 0, . . .],
    "time": [0.000, 0.001, 0.002, . . .],
    "x": [0.0, 0.01, 0.02, . . .],
    "x_dot": [0.0, 0.5, 0.4, . . .],
    "theta": [0.05, 0.03, 0.01, . . .],
    "theta_dot": [0.2, 0.15, 0.10, . . .],
    "u": [0.0, 1.0, 1.0, . . .],
    "reward": [0.0, 0.8, 0.9, . . .],
    "termination_reason": ["time_excedded", "time_excedded", "time_excedded", . . .]
  },
  {
    "episode": [1, 1, 1, . . .],
    "time": [0.00, 0.02, 0.04, . . .],
    "x": [0.0, -0.01, -0.02, . . .],
    "x_dot": [0.0, -0.6, -0.3, . . .],
    "theta": [0.06, 0.04, 0.02, . . .],
    "theta_dot": [0.25, 0.18, 0.12, . . .],
    "u": [0.0, -1.0, -1.0, . . .],
    "reward": [0.0, 0.7, 0.85, . . .],
    "termination_reason": ["goal_reached", "goal_reached", "goal_reached", . . .]
  },
  . . .
]
```

## 7. Citación
Si utiliza este repositorio o reproduce sus resultados, cite el manuscrito:

**Formato breve**:
Ibarra-Pérez, D., García-Nieto, S., \& Sanchis Saez, J. (2025). _Q-Learning for On-Line PID Controller Tuning in Continuous Dynamic Systems: An Interpretable Framework for Exploring Multi-Agent System_. _Mathematics_. _MDPI_. DOI: [pendiente].

**BibTeX (plantilla)**
```
@article{IbarraPerez2025QLearningPID,
  title   = {Q-Learning for On-Line PID Controller Tuning in Continuous Dynamic Systems:
             An Interpretable Framework for Exploring Multi-Agent System},
  author  = {Ibarra-P{\'e}rez, D. and Garc{\'i}a-Nieto, S. and Sanchis Saez, J.},
  journal = {Mathematics},
  year    = {2025},
  doi     = {<pendiente>}
}
```

## Licencia

**Código del proyecto (carpeta `code_project/`)**
Licencia ->`PolyForm-Noncommercial-1.0.0`.
- Se permite **uso, modificación y redistribución no comercial** (docencia, investigación académica, evaluación).
- **No esta permitido uso comercial** (p. ej., integración en productos/servicios, consultoría remunerada, SaaS), sin acuerdo **de licencia separada** con el autor.
**Datos, figuras y artefactos derivados de artículos (carpeta `scientific_articles/`)**
Licencia -> `CC-BY-NC-4.0`.
- Requiere **atribución** a los autores y **prohíbe usos comerciales**; se permite compartir y adaptar bajo las mismas condiciones.

## 9. Nota de contacto
Para permisos comerciales del código, reproducción, utilización o para aclaraciones sobre validación de datos vinculados a las publicaciones, escribir a davor.ibarra@usach.cl