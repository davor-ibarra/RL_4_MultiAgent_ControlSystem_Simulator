---
created: 20250305 09:03
update: 20250410-11:16
summary: Release and Sprint Planning
status: 
link: 
tags:
  - content
---
# Product Backlog

Items de alto nivel que representan el valor a entregar para alcanzar la visión del proyecto.

- **PBI1: MVP Funcional de Sintonización PID Péndulo:** Establecer una base funcional y validada para la sintonización del PID del péndulo invertido usando Q-Learning, con guardado de estado y métricas robustas. 
	- **(`*` completado en Sprint 1.1)**
- **PBI2: Control Avanzado de Péndulo y Adaptabilidad:** Implementar control dual (ángulo y posición del carro) y mecanismos básicos de adaptación de hiperparámetros, mejorando la estructura del código.
- **PBI3: Implementación del Núcleo del Framework Extensible:** Construir los componentes centrales (Orchestrator, Registry, ConfigManager) y refactorizar la base de código existente para adherirse estrictamente a la nueva arquitectura propuesta.
- **PBI4: Demostración de Extensibilidad del Framework:** Integrar un nuevo tipo de sistema dinámico y un nuevo tipo de agente RL utilizando únicamente configuración, validando la flexibilidad del framework.

---
# Release 1: MVP y Control Avanzado del Péndulo

**Objetivo del Release:** Entregar un sistema capaz de sintonizar un controlador PID para el péndulo invertido (ángulo y posición), con mejoras en la adaptabilidad y una estructura de código más organizada como paso intermedio hacia el framework completo.  

**Resultado Esperado:** Una base de código validada y funcional que implementa las características de control avanzadas para el péndulo, lista para ser refactorizada hacia la arquitectura final en el Release 2.
## Sprint 1.1: Establecimiento de Línea Base y Validación MVP Inicial

**Objetivo:** Mejorar y validar que la implementación actual produce resultados consistentes y esperados para la sintonización del PID del ángulo del péndulo. Establecer una línea base fiable.

**Pruebas de validación:** Comparación de gráficos de recompensa, tiempo, heatmaps y métricas de resumen con ejecuciones anteriores (si existen) o expectativas teóricas. Asegurar que el guardado/carga de estado del agente funciona.

**Diseño de la solución técnica:** Modular el sistema completo y mejorar la implementación de tablas Q basadas en NumPy, mecanismo de guardado/transformación de estado del agente, refactorización inicial del bucle principal y utilidades.

### Backlog Sprint 1.1

- Implementar PIDQLearningAgent con almacenamiento interno NumPy.
- Crear agent_state_manager para guardar/cargar estado del agente (formato JSON).
- Implementar transformación NumPy -> Dict en PIDQLearningAgent para guardado.
- Refactorizar main.py para manejar el flujo de simulación, guardado periódico y errores.
- Validar la generación de métricas, resúmenes (summarize_episode, save_summary_table) y visualizaciones.
- Ejecutar simulaciones y validar consistencia de resultados (Informe de estado actual).

## Sprint 1.2: Recompensas Diferenciales y Evaluación del Impacto de Acciones

**Objetivo:** Implementar un mecanismo que permita medir y diferenciar el impacto de las acciones individuales de cada agente (Kp, Ki, Kd) sobre la recompensa global, mediante estrategias contrafactuales y baselines internos. Esta etapa sienta las bases para control dual y futuras mejoras de adaptabilidad.

**Pruebas de validación:** 
- Evaluación cualitativa de simulaciones donde se observa si los agentes aprenden a estabilizar el péndulo.
- Verificación de que las recompensas diferenciales por agente se calculan correctamente (por logs, gráficas).
- Simulaciones en modo `echo-baseline` y `shadow-baseline` funcionando sin errores, con métricas de rendimiento razonables.


**Diseño de la solución técnica:**
- Se reemplaza la recompensa global única por un sistema donde cada agente recibe una señal individual que estima su contribución.
- Implementación de dos estrategias:
  - **Echo Baseline:** recompensa contrafactual simulando qué habría ocurrido si el agente no hubiera cambiado su ganancia.
  - **Shadow Baseline:** recompensa diferencial calculada como la diferencia entre la recompensa real y una baseline local por estado `B(s)`, ponderada por una métrica de estabilidad.
- Inclusión de `reward_mode` en `config.yaml` con las opciones:
  - `global` (actual),
  - `echo-baseline`,
  - `shadow-baseline`.
- Integración del `PendulumVirtualSimulator` para el cálculo de recompensas contrafactuales.
- Refactorización mínima del entorno y del agente para soportar estos nuevos modos.

### Backlog Sprint 1.2

- Modificar `PIDQLearningAgent` para aceptar recompensas diferenciales por ganancia (`kp`, `ki`, `kd`).
- Implementar modo `echo-baseline` con simulaciones contrafactuales por agente.
- Implementar modo `shadow-baseline` con tabla de baseline local por estado y métrica de estabilidad (`w_stab`).
- Actualizar `GaussianReward` para soportar estabilidad externa (`IRA`, `SimpleExponential`).
- Añadir `reward_mode` en `config.yaml` y lógica correspondiente en `main.py`.
- Validar correcto funcionamiento del `PendulumVirtualSimulator`.
- Loggear métricas diferenciales por agente y `w_stab` por intervalo.
- Validar el aprendizaje diferencial y la convergencia individual de Q-Tables por ganancia.

---
# Release 2: Framework Extensible Completo

**Objetivo del Release:** Implementar la arquitectura central propuesta (Orchestrator, Registry, etc.) y demostrar su flexibilidad añadiendo nuevos componentes (sistema y agente) mediante configuración.

**Resultado Esperado:** Un framework robusto donde las simulaciones se definen y ejecutan a través de workflows configurables, y la adición de nuevas funcionalidades (sistemas, agentes, controladores) se realiza de forma modular y desacoplada.
## Sprint 2.1: Construcción del Núcleo del Framework

**Objetivo:** Implementar los componentes centrales (core, utils mejorados) y refactorizar la base de código existente (del Release 1) para que se ajuste estrictamente a las interfaces y la estructura definidas en la propuesta técnica (1 - 2 - 1 - Technical System Documentation.md y 1 - 2 - 2 - System Proposal.md).

**Pruebas de validación:** Ejecución exitosa de la simulación del péndulo invertido (control dual del Sprint 1.2) utilizando el nuevo Orchestrator y Registry. Los componentes deben ser cargados y gestionados a través del framework.

**Diseño de la solución técnica:** Definir las implementaciones concretas de SimulationOrchestrator, ComponentRegistry, ConfigurationManager. Planificar la refactorización de todas las fábricas y componentes existentes para cumplir las nuevas interfaces (interfaces/) y usar el Registry y EventBus. Adaptar main.py para ser el punto de entrada que inicializa y ejecuta el Orchestrator.

### Backlog Sprint 2.1

- Implementar core.ConfigurationManager (carga, validación, acceso a parámetros).
- Implementar core.ComponentRegistry (registro, descubrimiento, obtención de componentes).
- Implementar core.Orchestrator (carga y ejecución de workflows básicos).
- Implementar utils.Logger y utils.EventBus (si se decide usarlos explícitamente).
- Refactorizar todas las factories/* para interactuar con el Registry.
- Refactorizar todas las components/* (agentes, controladores, sistemas, etc. existentes) para implementar estrictamente las interfaces/*.
- Actualizar main.py para inicializar el core y ejecutar el Orchestrator.
- Adaptar config.yaml a un formato de workflow compatible con el Orchestrator.
- Crear/actualizar pruebas unitarias/integración para el core y componentes refactorizados.

## Sprint 2.2: Demostración de Extensibilidad

**Objetivo:** Validar la flexibilidad del framework implementado añadiendo un nuevo tipo de DynamicSystem (ej. Motor DC simple) y un nuevo tipo de RLAgent (ej. Sistema Multi-Agente o un DQN básico), ejecutando una simulación con ellos únicamente mediante cambios en la configuración.

**Pruebas de validación:** Ejecución exitosa (aunque no necesariamente óptima) de una simulación con el nuevo sistema y agente, orquestada completamente por el framework. El código del core no debe requerir modificaciones.

**Diseño de la solución técnica:** Implementar las clases concretas para el nuevo sistema y agente, asegurando que implementen las interfaces correspondientes. Crear un archivo de configuración (config_motor_random.yaml) que defina un workflow utilizando estos nuevos componentes.

### Backlog Sprint 2.2

- Implementar components/systems/dc_motor_system.py (o similar) implementando interfaces.DynamicSystem.
- Implementar components/agents/multi_agent.py (o un DQN básico) implementando interfaces.RLAgent.
- Asegurar que el Registry descubra los nuevos componentes.
- Crear data/configs/config_motor_random.yaml definiendo un workflow con los nuevos componentes.
- Ejecutar main.py con la nueva configuración y verificar que la simulación se ejecuta sin errores (validar flujo de datos básico).
- Añadir pruebas unitarias básicas para los nuevos componentes.

