---
created: 20250305 10:03
update: 20250416-02:37
summary: Release and Sprint control
status: 
link: 
tags:
  - High-Level-Note
---
# WIP

**Tablero Scrum:**

[[]]

# Check List
## Product Backlog
- [~] PBI1: MVP Funcional de Sintonización PID Péndulo
- [~] PBI2: Control Avanzado de Péndulo y Adaptabilidad
- [ ] PBI3: Implementación del Núcleo del Framework Extensible
- [ ] PBI4: Demostración de Extensibilidad del Framework
## Stage Check
- [ ] **Release 1: MVP y Control Avanzado del Péndulo** (En Progreso)
	- [~] **Sprint 1.1: Establecimiento de Línea Base y Validación MVP Inicial** (Completado)
		- **Objetivo:** Validar implementación actual (post-NumPy/estado) y establecer línea base fiable.
		- **Resultado Esperado:** Código base validado, estado del agente guardable/cargable, métricas consistentes.
		- **Outcome:** Se genera informe de estado actual detallando la funcionalidad, flujo de datos y validación de la implementación con tablas NumPy y guardado de estado. Código base estable para iniciar Sprint 1.2.
	- [~] **Sprint 1.2: Recompensas Diferenciales y Evaluación del Impacto de Acciones
		- **Objetivo:** Reemplazar el modelo de recompensa global por un mecanismo que permita medir el impacto individual de las acciones de cada agente (Kp, Ki, Kd) en la estabilidad del sistema, utilizando simulaciones virtuales y baselines por estado.
		- **Resultado Esperado:** Sistema entrenando con recompensas diferenciadas, permitiendo evaluar qué tan beneficiosa fue la acción de cada agente. Lógica de entrenamiento más explicativa y trazable para futuras mejoras.
		- **Outcome:** Los nuevos modos `echo-baseline` y `shadow-baseline` fueron integrados exitosamente, permitiendo descomponer el efecto de cada ganancia del PID sobre la recompensa global. Esto allana el camino para entrenamientos más estables y agentes con roles diferenciados. El código fue adaptado con mínima refactorización, preservando la modularidad.
- [ ] **Release 2: Framework Extensible Completo** (Planificado)
	- [ ] **Sprint 2.1: **Construcción del Núcleo del Framework** (Planificado)
		- **Objetivo:** Implementar core (Orchestrator, Registry, etc.) y refactorizar componentes existentes a nuevas interfaces.
		- **Resultado Esperado:** Simulación del péndulo ejecutada a través del nuevo framework central.
		- **Outcome:** 
	- [ ] **Sprint 2.2: Demostración de Extensibilidad** (Planificado)
		- **Objetivo:** Añadir nuevo sistema y agente vía configuración para validar flexibilidad.
		- **Resultado Esperado:** Simulación de nuevo sistema/agente funcionando bajo el framework sin cambios en el core.
		- **Outcome:**

---
## Release 1

---
### Sprint 1.1

**Objetivo:** Validar que la implementación actual produce resultados consistentes y esperados para la sintonización del PID del ángulo del péndulo. Establecer una línea base fiable.

**Backlog:**
- [x] Implementación de PIDQLearningAgent con tablas Q y contadores de visita basados en NumPy.
- [x] Creación de agent_state_manager y lógica de transformación en el agente para guardar estado en JSON.
- [x] Refactorización de main.py para mejorar flujo, manejo de errores y guardado periódico.
- [x] Validación de utilidades de procesamiento de datos
	- [x] summarize_episode
	- [x] save_summary_table
	- [x] visualization.
- [x] Informe de Estado Actual del Proyecto, validando la funcionalidad y el flujo de datos.
- [ ] Ejecución de simulaciones e Informe de Resultados


**Historial:**
Dev:
[[20250324 - Sprint 1.1 - Escalamiento Tecnológico a MVP - Parte 1]]
[[20250401 - Sprint 1.1 - Escalamiento Tecnológico a MVP - Parte 2]]
Test:
[[20250325 - Validación del Escalamiento tecnológico ]]
Prod:
[[20250326 - Informe de resultados]]  (en lienzo)


![[Sprint 1.1 - Actual System State.png]]

**Retrospectiva:**

- ✅**Qué fue bien:** 
	- La estructura general es comprensible y flexible.
	- La implementación de NumPy mejoró la eficiencia. 
	- El guardado de estados y de datos ha sido validado. 
	- La visualización de datos configurable junto con la simulación.
- 🛠️**Qué podría mejorar:** 
	- La separación de responsabilidades entre 
		- main.py, 
		- PendulumEnvironment
		- PIDQLearningAgent.
- 🧭**Acciones:** 
	- Implementar controlador dual. (realizado pero se recomienda revalidar)
	- Inicialización Aleatoria de Condiciones iniciales
	- Terminar deuda técnica
		- Nombres de variables
		- Separar algunas responsabilidades
		- Extracción de Q-tables a utils
		- Optimizar y homologar gráficos en app
	- Planificar refactorización explícita en Sprint 1.2. 
	- Ideas:
		- [[2 - 1 - 1 - Sprint 1.1 - Retrospective Results]]
		- Limitar al agente a solo las acciones disponibles
		- Diseño de la Recompensa
			- Función de Costo con Ponderación Dinámica
			- Recompensa Intermedia por Sub-objetivos
				- Reducir el error de ángulo por debajo de un umbral.
				- Mantenerse dentro de una franja estable durante cierto número de pasos.
		- Abordar el entrenamiento guiado mediante estrategias de exploración/explotación adaptativas.
			- Agente con modo explorador o explotador
			- Exploración
				- Upper Confidence Bound
				- Exploración Dirigida por la Recompensa
			- Explotación
				- ε Decay adaptativa en función al rendimiento del agente
			- Entrenamiento individualizado, combinado, finalizando con uno conjunto
		- Modos de agentes para el ajuste fino o grueso
		- Memoria de las políticas (aunque sería optimizar la política)
			- Almacenar trayectorias exitosas y amplificar recompensas de las acciones que llevaron a un episodio exitoso

---
### Sprint 1.2

**Objetivo:** ...

**Backlog:**
- [x] Implementar modo `echo-baseline` con simulaciones contrafactuales y diferenciales por ganancia.
- [x] Implementar modo `shadow-baseline` con baseline local `B(s)` y métrica de estabilidad `w_stab`.
- [x] Refactorizar `PIDQLearningAgent` para aceptar recompensas por agente.
- [x] Añadir soporte de `reward_mode` en `config.yaml` y lógica correspondiente en `main.py`.
- [x] Validar `PendulumVirtualSimulator` para simulaciones de recompensa virtual.
- [x] Loggear `w_stab`, recompensas diferenciales y baseline para trazabilidad.

**Historial:**
Dev:
[[20250408 - Sprint 1.2 - Propuesta de Control de Péndulo Avanzado]]
[[20250410 - Sprint 1.2 - Informe de Escalamiento Parte 1 - Control de Péndulo Avanzado]]
[[20250411 - Sprint 1.2 - Mejoras al Proyecto Actual]]
[[20250414 - Sprint 1.2 - Informe de Escalamiento Parte 2 - Estado Actual del Proyecto]]
Test:
[[20250425 - Sprint 1.2 - Plataforma para la Evaluación de Resultados]]
Prod:
[[20250410 - Informe Implementación y Evaluación Recompensas Diferenciadas]]




**Retrospectiva:**

- ✅ **Qué fue bien:**
	- **Integración exitosa de lógica avanzada de recompensa** (`echo-baseline` y `shadow-baseline`), sin romper compatibilidad con el modo `global`.
		- Se adiciona un calculador de reward del tipo `gaussian` o del tipo `stability_calculator`
	- **PendulumVirtualSimulator** implementado correctamente, reutilizando el controlador y sistema sin afectar el entorno real.
	- **Separación clara de responsabilidades** entre `main.py`, `PendulumEnvironment`, `PIDQLearningAgent` y `GaussianReward`, facilitando la trazabilidad de cada flujo de decisión y aprendizaje.
	- **Sistema robusto de métricas y logging**, permitiendo análisis detallado del comportamiento del agente y del sistema.
	- **Flexibilidad total desde el `config.yaml`**, lo que permitirá experimentar fácilmente con distintos modos y parámetros de recompensa.
- 🛠️ **Qué podría mejorar:**
	- **Costo computacional elevado en modo `echo-baseline`**
	- **Dificultad para interpretar el impacto individual de cada ganancia sin herramientas de visualización específicas.**
	- **Dependencia fuerte de la configuración correcta**
	- **Complejidad técnica creciente**: la integración de múltiples agentes y modos de recompensa complica la mantenibilidad sin una refactorización del núcleo orientada a control multiagente.
- 🧭 **Acciones para los próximos sprints:**
	- Desarrollar un **modo de entrenamiento dual (ángulo + posición)** con separación clara de agentes.
	- Incorporar **herramientas de análisis visual del impacto de cada acción** en la estabilidad.
	- **Optimizar el rendimiento del modo `echo-baseline`**, por ejemplo, paralelizando simulaciones virtuales o limitando su frecuencia.
	- Estudiar la integración de una **estrategia de entrenamiento jerárquico o cooperativo entre agentes**.
	- Planificar una **refactorización hacia arquitectura orientada a componentes multi-agente**, manteniendo los principios actuales.



## Release 2

### Sprint 2.1

**Objetivo:** ...

**Backlog:**
- 

**Historial:**
Dev:

Test:

Prod:

### Sprint 2.2

**Objetivo:** ...

**Backlog:**
- 

**Historial:**
Dev:

Test:

Prod: