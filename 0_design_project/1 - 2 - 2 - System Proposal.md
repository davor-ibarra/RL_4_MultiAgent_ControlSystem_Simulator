---
created: 20250306 05:03
update: 20250324-08:58
summary: 
status: 
link: 
tags:
  - content
---
# File Structure

```
RL-Agent-Factory/
├── core/
│   ├── orchestrator.py
│   ├── registry.py
│   └── config_manager.py
│
├── interfaces/
│   ├── dynamic_system.py
│   ├── controller.py
│   ├── rl_agent.py
│   ├── environment.py
│   ├── reward_function.py
│   ├── metrics_collector.py
│   └── config_validator.py
│
├── factories/
│   ├── system_factory.py
│   ├── controller_factory.py
│   ├── agent_factory.py
│   ├── environment_factory.py
│   └── reward_factory.py
│
├── components/
│   ├── agents/
│   ├── controllers/
│   ├── systems/
│   ├── environments/
│   └── rewards/
│
├── utils/
│   ├── logger.py
│   ├── event_bus.py
│   └── validators.py
│
├── analysis/
│   ├── metrics_analyzer.py
│   └── report_generator.py
│
├── data/
│   ├── configs/
│   │   └── default_config.yaml
│   └── logs/
│
├── tests/
│   └── ...
│
├── requirements.txt
└── main.py

```


# Role and responsibilities of modules and classes

## 📂 **1. core/**

### 📌 **orchestrator.py**

Clase `SimulationOrchestrator`

- **Objetivo:** Controlar la ejecución y ciclo de vida completo de simulaciones definidas por workflows.
- **Responsabilidades:**
	- Ejecutar workflows desde configuración.
	- Gestionar dependencias de componentes y recursos.
	- Manejar excepciones durante la ejecución.

| Método                       | Responsabilidad Específica                          |
| ---------------------------- | --------------------------------------------------- |
| `__init__(config, registry)` | Inicializa configuración y registro                 |
| `load_workflow()`            | Lee y carga workflow desde configuración            |
| `validate_workflow()`        | Valida componentes y dependencias del workflow      |
| `execute_workflow()`         | Ejecuta workflow en orden lógico definido           |
| `terminate_workflow()`       | Termina ejecución liberando recursos                |
| **`pause_workflow()`**       | Pausa workflow y conserva estado actual             |
| **`resume_workflow()`**      | Reanuda workflow desde estado pausado               |
| **`handle_exception(e)`**    | Manejo robusto y reporte inteligente de excepciones |

### 📌 **registry.py**

Clase `ComponentRegistry`

- **Objetivo:** Mantener un registro validado y organizado de componentes.
- **Responsabilidades:**
	- Autodescubrimiento de componentes.
	- Registro de interfaces/componentes.
	- Gestión de dependencias internas.

| Método                                                      | Responsabilidad Específica                     |
| ----------------------------------------------------------- | ---------------------------------------------- |
| `register(component_type, component_name, component_class)` | Registra componentes específicos               |
| `get(component_type, component_name)`                       | Devuelve una instancia de componente           |
| `validate_registry()`                                       | Verifica validez componentes registrados       |
| `discover_components(component_dir)`                        | Autodescubrimiento componentes desde carpeta   |
| **`unregister(component_type, component_name)`**            | Desregistra componentes en tiempo de ejecución |
| **`list_components(component_type)`**                       | Lista componentes registrados por categoría    |

### 📌 **config_manager.py**

Clase `ConfigurationManager`

- **Objetivo:** Administración centralizada y dinámica de configuraciones del proyecto.
- **Responsabilidades:**
	- Validar dinámicamente parámetros.
	- Actualizar configuraciones durante ejecución.

| Método                                 | Responsabilidad Específica                      |
| -------------------------------------- | ----------------------------------------------- |
| `load_config(config_path)`             | Lee configuraciones desde YAML                  |
| `validate_config(schema_path)`         | Valida configuración según esquema              |
| `get_param(param_path)`                | Obtiene valor de parámetro específico           |
| `update_config(param_path, new_value)` | Actualiza parámetro durante ejecución           |
| **`save_config(output_path)`**         | Guarda configuración actualizada a archivo YAML |
| **`reset_config(default_config)`**     | Resetea configuración al estado por defecto     |

---

## 📂 **2. interfaces/**

### 📌 **DynamicSystem**

Interfaz genérica para sistemas dinámicos.

|Método|Responsabilidad Específica|
|---|---|
|`apply_action(action)`|Aplica acción al sistema dinámico, actualizando el estado.|
|`reset(initial_conditions)`|Reinicia el sistema a condiciones iniciales especificadas.|
|`get_state()`|Devuelve el estado actual del sistema dinámico.|
|**`get_state_space()`**|Devuelve características del espacio de estados.|
|**`get_action_space()`**|Devuelve características del espacio de acciones.|

### 📌 **Controller**

Interfaz genérica para controladores adaptativos y tradicionales.

|Método|Responsabilidad Específica|
|---|---|
|`compute_action(state)`|Calcula y devuelve la acción de control dada el estado actual.|
|`update_params(params)`|Actualiza parámetros internos del controlador en ejecución.|
|`reset()`|Reinicia controlador a valores iniciales predefinidos.|
|**`get_params()`**|Obtiene parámetros actuales del controlador.|
|**`auto_tune(target_state)`**|Autoajuste dinámico de parámetros del controlador (opcional).|

### 📌 **RLAgent**

Interfaz genérica para agentes de aprendizaje por refuerzo.

|Método|Responsabilidad Específica|
|---|---|
|`select_action(state)`|Selecciona y devuelve acción dada el estado actual.|
|`learn(batch)`|Entrena el modelo con un lote de experiencias.|
|`store_experience(state, action, reward, next_state)`|Almacena experiencia individual en memoria interna.|
|**`save_model(path)`**|Guarda modelo entrenado en archivo especificado.|
|**`load_model(path)`**|Carga modelo previamente entrenado desde archivo.|
|**`reset_agent(initial_conditions)`**|Reinicia parámetros internos del agente al estado inicial.|

### 📌 **Environment**

Interfaz genérica para entornos de simulación.

|Método|Responsabilidad Específica|
|---|---|
|`step(action)`|Ejecuta acción en entorno, retornando estado siguiente.|
|`reset(initial_conditions)`|Reinicia el entorno al estado inicial especificado.|
|`get_reward(state, action, next_state)`|Calcula recompensa para una transición dada.|
|**`render(mode='human')`**|Visualiza entorno actual de manera gráfica.|
|**`close()`**|Libera recursos asignados al entorno.|
|**`get_observation_space()`**|Devuelve detalles del espacio de observación (estados).|
|**`get_action_space()`**|Devuelve detalles del espacio de acciones.|

### 📌 **RewardFunction**

Interfaz para funciones de recompensa.

|Método|Responsabilidad Específica|
|---|---|
|`calculate(state, action, next_state)`|Calcula recompensa basada en transición especificada.|

### 📌 **MetricsCollector**

Interfaz para recopilación de métricas durante simulaciones.

|Método|Responsabilidad Específica|
|---|---|
|`log(metric_name, metric_value)`|Almacena métricas individuales durante ejecución.|
|`get_metrics()`|Retorna todas las métricas recopiladas hasta el momento.|
|`reset_metrics()`|Reinicia almacenamiento interno de métricas.|
|**`export_metrics(path)`**|Exporta métricas recopiladas a un archivo externo.|

### 📌 **ConfigValidator**

Interfaz para validadores de configuraciones del sistema.

| Método                                | Responsabilidad Específica                                  |
| ------------------------------------- | ----------------------------------------------------------- |
| `validate(config, schema)`            | Valida configuración según un esquema definido previamente. |
| **`generate_schema(example_config)`** | Genera esquema a partir de una configuración ejemplo dada.  |

---

## 📂 **3. factories/**

Cada Factory instancia componentes según configuración específica.

### 📌 **SystemFactory**
Fábrica para instanciar sistemas dinámicos específicos.

| Método                                               | Responsabilidad Específica                              |
| ---------------------------------------------------- | ------------------------------------------------------- |
| `create_system(system_config)`                       | Instancia un sistema dinámico según configuración dada. |
| **`register_custom_system(system_name, class_ref)`** | Registra sistemas dinámicos personalizados al registro. |
| **`list_available_systems()`**                       | Lista sistemas dinámicos disponibles en el registro.    |

### 📌 **ControllerFactory**
Fábrica para instanciar controladores específicos (clásicos o híbridos).

| Método                                                       | Responsabilidad Específica                               |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| `create_controller(controller_config)`                       | Instancia controlador según configuración proporcionada. |
| **`register_custom_controller(controller_name, class_ref)`** | Registra controladores personalizados al registro.       |
| **`list_available_controllers()`**                           | Lista controladores disponibles en el registro.          |

### 📌 **AgentFactory**

Fábrica para instanciar agentes de aprendizaje por refuerzo específicos.

| Método                                             | Responsabilidad Específica                         |
| -------------------------------------------------- | -------------------------------------------------- |
| `create_agent(agent_config)`                       | Instancia agente RL según configuración dada.      |
| **`register_custom_agent(agent_name, class_ref)`** | Registra agentes RL personalizados en el registro. |
| **`list_available_agents()`**                      | Lista agentes disponibles en el registro.          |

### 📌 **EnvironmentFactory**

Fábrica para instanciar entornos de simulación específicos.

| Método                                                 | Responsabilidad Específica                                |
| ------------------------------------------------------ | --------------------------------------------------------- |
| `create_environment(env_config)`                       | Instancia entorno de simulación según configuración dada. |
| **`register_custom_environment(env_name, class_ref)`** | Registra entornos personalizados en el registro.          |
| **`list_available_environments()`**                    | Lista entornos disponibles en el registro.                |

### 📌 **RewardFactory**

Fábrica para instanciar funciones de recompensa específicas.

| Método                                               | Responsabilidad Específica                       |
| ---------------------------------------------------- | ------------------------------------------------ |
| `create_reward_function(reward_config)`              | Instancia función recompensa según configuración |
| **`register_custom_reward(reward_name, class_ref)`** | Registra funciones recompensa personalizadas     |

---

## 📂 **4. components/**

Implementaciones específicas según interfaces definidas previamente:

## 📂 **agents/**

###  📌Clase **QLearningAgent**

|Método|Responsabilidad Específica|
|---|---|
|`select_action(state)`|Selecciona acción usando política ε-greedy|
|`learn(batch)`|Actualiza la tabla Q según lote de experiencias|
|`store_experience(state, action, reward, next_state)`|Almacena experiencia individual en memoria|
|**`epsilon_decay()`**|Controla decaimiento del factor de exploración ε|
|**`update_q_values(state, action, reward, next_state)`**|Método específico de actualización de valores Q|
|**`save_model(path)`**|Guarda tabla Q actual en archivo|
|**`load_model(path)`**|Carga tabla Q desde archivo|
|**`reset_agent(initial_conditions)`**|Reinicia tabla Q y parámetros internos|

### 📌 Clase **DQNAgent**

|Método|Responsabilidad Específica|
|---|---|
|`select_action(state)`|Selecciona acción usando red neuronal|
|`learn(batch)`|Entrena red neuronal según lote de experiencias|
|`store_experience(state, action, reward, next_state)`|Guarda experiencia individual en Replay Buffer|
|**`update_target_network()`**|Actualiza la red objetivo (target network) periódicamente|
|**`save_model(path)`**|Guarda modelo neuronal entrenado en archivo|
|**`load_model(path)`**|Carga modelo neuronal previamente entrenado|
|**`reset_agent(initial_conditions)`**|Reinicia pesos y parámetros del modelo neuronal|

### 📌 Clase **ActorCriticAgent**

|Método|Responsabilidad Específica|
|---|---|
|`select_action(state)`|Selecciona acción según la política del actor|
|`learn(batch)`|Entrena redes del actor y crítico simultáneamente|
|`store_experience(state, action, reward, next_state)`|Almacena experiencia en memoria interna|
|**`update_actor()`**|Actualiza parámetros específicos del actor|
|**`update_critic()`**|Actualiza parámetros específicos del crítico|
|**`save_model(path)`**|Guarda ambos modelos (actor y crítico) en archivo|
|**`load_model(path)`**|Carga modelos actor y crítico desde archivo|
|**`reset_agent(initial_conditions)`**|Reinicia parámetros internos del actor y crítico|

### 📌 Clase **ReplayBuffer**

| Método                                     | Responsabilidad Específica                         |
| ------------------------------------------ | -------------------------------------------------- |
| `store(state, action, reward, next_state)` | Guarda experiencias individuales                   |
| `sample(batch_size)`                       | Retorna lote aleatorio de experiencias almacenadas |
| `clear()`                                  | Limpia todas las experiencias almacenadas          |
| `get_size()`                               | Retorna número actual de experiencias almacenadas  |

## 📂 **controllers/**

### 📌 Clase **PIDController**

|Método|Responsabilidad Específica|
|---|---|
|`compute_action(state)`|Calcula acción PID a partir del error actual|
|`update_params(params)`|Actualiza dinámicamente parámetros PID (Kp, Ki, Kd)|
|`reset()`|Reinicia variables internas y acumuladas del controlador|
|**`auto_tune(target_state)`**|Ajusta dinámicamente parámetros PID|
|**`anti_windup()`**|Aplica técnica anti-windup para evitar saturación|
|**`get_params()`**|Retorna parámetros PID actuales|

### 📌 Clase **LQRController**

|Método|Responsabilidad Específica|
|---|---|
|`compute_action(state)`|Calcula acción usando control óptimo LQR|
|`update_params(params)`|Actualiza matrices internas (Q, R) de controlador|
|`reset()`|Reinicia controlador a valores iniciales|
|**`calculate_gain()`**|Calcula ganancias óptimas LQR en tiempo de ejecución|
|**`get_params()`**|Retorna parámetros actuales del controlador|

### 📌 Clase **HybridRLController**

|Método|Responsabilidad Específica|
|---|---|
|`compute_action(state)`|Calcula acción híbrida combinando control clásico y RL|
|`update_params(params)`|Actualiza parámetros híbridos|
|`reset()`|Reinicia parámetros internos del controlador híbrido|
|**`switch_mode(mode)`**|Cambia dinámicamente entre control clásico y RL|
|**`get_params()`**|Retorna parámetros híbridos actuales|

## 📂 **systems/**

### 📌 Clase **PendulumSystem**

| Método                      | Responsabilidad Específica                 |
| --------------------------- | ------------------------------------------ |
| `apply_action(action)`      | Aplica acción sobre sistema                |
| `reset(initial_conditions)` | Reinicia condiciones iniciales del sistema |
| `get_state()`               | Retorna estado actual del sistema          |
| **`get_state_space()`**     | Retorna espacio posible de estados         |
| **`get_action_space()`**    | Retorna espacio posible de acciones        |

_(Similar para `ElectricalMotorSystem`, `ThermalSystem`)_

## 📂 **environments/**

### 📌 Clase **SimulationEnvironment**

|Método|Responsabilidad Específica|
|---|---|
|`step(action)`|Ejecuta acción y retorna nuevo estado y recompensa|
|`reset(initial_conditions)`|Reinicia simulación a condiciones iniciales|
|`get_reward(state, action, next_state)`|Calcula recompensa asociada a transición|
|**`render(mode='human')`**|Visualización gráfica del entorno|
|**`close()`**|Libera recursos del entorno|
|**`set_time_limit(time_limit)`**|Establece duración máxima de episodios|
|**`set_state_space(normalizer)`**|Normaliza espacio de estados|
|**`set_action_space(normalizer)`**|Normaliza espacio de acciones|

## 📂 **rewards/**

### 📌 Clase **GaussianReward**

|Método|Responsabilidad Específica|
|---|---|
|`calculate(state, action, next_state)`|Calcula recompensa utilizando función gaussiana|
|**`update_parameters(params)`**|Actualiza parámetros (media, desviación) en tiempo real|

### 📌 Clase **LinearReward**

| Método                                 | Responsabilidad Específica                          |
| -------------------------------------- | --------------------------------------------------- |
| `calculate(state, action, next_state)` | Calcula recompensa utilizando función lineal        |
| **`update_weights(weights)`**          | Actualiza pesos de la función lineal en tiempo real |

Cada clase implementa estrictamente métodos definidos en su interfaz correspondiente.

---

## 📂 **5. utils/**

### 📌 **logger.py**

Clase `Logger`
- Clase para el registro centralizado de eventos, advertencias, errores y depuración.

| Método                     | Responsabilidad Específica                                        |
| -------------------------- | ----------------------------------------------------------------- |
| `log_info(message)`        | Registra mensajes informativos estándar del sistema.              |
| `log_error(message)`       | Registra errores o problemas críticos del sistema.                |
| `log_debug(message)`       | Registra mensajes detallados para depuración.                     |
| **`log_warning(message)`** | Registra advertencias no críticas.                                |
| **`set_log_level(level)`** | Configura nivel de detalle de logs (DEBUG, INFO, WARNING, ERROR). |
| **`export_log(path)`**     | Exporta todos los registros almacenados en un archivo externo.    |
| **`clear_logs()`**         | Limpia los registros almacenados en memoria.                      |

### 📌 **event_bus.py**

Clase `EventBus`
- Clase para la gestión asíncrona y desacoplada de eventos internos del sistema.

| Método                                           | Responsabilidad Específica                                    |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `publish(event_type, data)`                      | Publica evento en el bus interno del sistema.                 |
| `subscribe(event_type, listener_callback)`       | Suscribe componentes a eventos específicos publicados.        |
| **`unsubscribe(event_type, listener_callback)`** | Elimina suscripción de componentes a eventos.                 |
| **`clear_subscribers(event_type)`**              | Remueve todos los suscriptores de un tipo de evento dado.     |
| **`list_subscribers(event_type)`**               | Lista todos los componentes suscritos a un evento específico. |

### 📌 **validators.py**

Clase `validator`
- Clase para métodos generalizados de validación de configuraciones, parámetros y tipos de datos.

| Método                                                 | Responsabilidad Específica                                          |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| `validate_numeric_range(value, min, max)`              | Valida que un valor numérico se encuentre en un rango específico.   |
| `validate_schema(config_dict, schema_path)`            | Valida diccionario de configuración según un esquema JSON/YAML.     |
| `validate_type(variable, expected_type)`               | Valida tipo específico de una variable dada.                        |
| **`validate_list_elements_type(list, expected_type)`** | Valida que todos los elementos de una lista sean del tipo esperado. |
| **`validate_non_empty(variable)`**                     | Valida que la variable proporcionada no esté vacía o nula.          |

---

## 📂 **6. analysis/**

### 📌 **metrics_analyzer.py**

Clase `MetricsAnalyzer`
- Procesar métricas de simulaciones.

| Método                                     | Responsabilidad Específica                     |
| ------------------------------------------ | ---------------------------------------------- |
| `aggregate(metrics_list)`                  | Agrega múltiples resultados métricos           |
| `evaluate_performance(aggregated_metrics)` | Evalúa desempeño según criterios               |
| `compare_algorithms(metrics_a, metrics_b)` | Compara desempeño entre algoritmos o agentes   |
| `calculate_statistics(metrics)`            | Calcula estadísticas (media, desviación, etc.) |

### 📌 **report_generator.py**

Clase `ReportGenerator`
- Generar reportes automatizados.

| Método                                           | Responsabilidad Específica                     |
| ------------------------------------------------ | ---------------------------------------------- |
| `create_report(evaluated_metrics, report_path)`  | Genera reporte detallado en archivo            |
| `generate_summary(evaluated_metrics)`            | Genera resumen ejecutivo de resultados         |
| `visualize_report(evaluated_metrics, save_path)` | Visualización gráfica automatizada de métricas |

---

## ✅ **7. MAIN.PY (Entry Point)**

Archivo principal que ejecuta la aplicación:
- Inicializa `ConfigurationManager`, `ComponentRegistry` y `SimulationOrchestrator`.
- Carga configuraciones desde archivo externo.
- Inicia workflows y monitorea ejecución.

| Método                                      | Responsabilidad Específica                                        |
| ------------------------------------------- | ----------------------------------------------------------------- |
| `main()`                                    | Punto de entrada del programa; ejecuta el workflow principal      |
| `setup_logging(log_level)`                  | Inicializa y configura sistema centralizado de logging            |
| `setup_event_bus()`                         | Inicializa sistema centralizado de eventos internos               |
| `initialize_config(config_path)`            | Inicializa y valida configuraciones desde archivos externos       |
| `initialize_registry(components_dir)`       | Descubre, registra y valida automáticamente componentes           |
| `initialize_orchestrator(config, registry)` | Crea y configura instancia del orquestador de simulaciones        |
| `run_simulation_loop(iterations)`           | Ejecuta múltiples simulaciones o workflows de forma automatizada  |
| `save_simulation_results(output_dir)`       | Guarda resultados finales de las simulaciones                     |
| `handle_exceptions(e)`                      | Manejo inteligente y estructurado de excepciones del sistema      |
| `shutdown_system()`                         | Finaliza ejecución limpiamente, libera recursos y cierra procesos |

Ejemplo:

```
from core.config_manager import ConfigurationManager
from core.registry import ComponentRegistry
from core.orchestrator import SimulationOrchestrator
from utils.logger import Logger
from utils.event_bus import EventBus

def setup_logging(log_level="INFO"):
    Logger.set_log_level(log_level)
    Logger.log_info("📌 Logging inicializado correctamente.")

def setup_event_bus():
    event_bus = EventBus()
    Logger.log_info("📌 Event bus inicializado correctamente.")
    return event_bus

def initialize_config(config_path='data/configs/default_config.yaml', schema_path='data/configs/schema.yaml'):
    config_manager = ConfigurationManager()
    config = config_manager.load_config(config_path)
    config_manager.validate_config(schema_path)
    Logger.log_info("📌 Configuración cargada y validada correctamente.")
    return config_manager, config

def initialize_registry(components_dir='components/'):
    registry = ComponentRegistry()
    registry.discover_components(components_dir)
    registry.validate_registry()
    Logger.log_info("📌 Componentes descubiertos y validados correctamente.")
    return registry

def initialize_orchestrator(config, registry):
    orchestrator = SimulationOrchestrator(config, registry)
    orchestrator.load_workflow()
    orchestrator.validate_workflow()
    Logger.log_info("📌 Orquestador inicializado y workflow validado.")
    return orchestrator

def run_simulation_loop(orchestrator, iterations=1):
    Logger.log_info(f"🚀 Iniciando ciclo de simulaciones ({iterations} iteraciones).")
    for iteration in range(iterations):
        Logger.log_info(f"🔄 Ejecutando simulación iteración {iteration+1}/{iterations}.")
        orchestrator.execute_workflow()
        orchestrator.terminate_workflow()
        Logger.log_info(f"✅ Simulación {iteration+1} completada.")

def save_simulation_results(output_dir='data/results/'):
    Logger.log_info(f"💾 Resultados guardados en {output_dir}.")

def handle_exceptions(e):
    Logger.log_error(f"⚠️ Excepción detectada: {str(e)}")
    Logger.export_log('data/logs/error_log.txt')

def shutdown_system():
    Logger.log_info("🛑 Sistema finalizado correctamente.")

def main():
    try:
        setup_logging(log_level="DEBUG")
        event_bus = setup_event_bus()
        config_manager, config = initialize_config()
        registry = initialize_registry()
        orchestrator = initialize_orchestrator(config, registry)

        run_simulation_loop(orchestrator, iterations=config_manager.get_param('simulation.iterations'))
        save_simulation_results()

    except Exception as e:
        handle_exceptions(e)

    finally:
        shutdown_system()

if __name__ == "__main__":
    main()

```

---
## 📋 **requirements.txt**

```
numpy
scipy
pyyaml
jsonschema
matplotlib
tensorboard    # Para monitoreo avanzado
gym
```


---

# UML Structure

![[system proposal.png]]