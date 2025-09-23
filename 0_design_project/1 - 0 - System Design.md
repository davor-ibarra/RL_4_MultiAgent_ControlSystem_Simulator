---
created: 20241005-18:28
update: 20250312-18:10
summary: 
status: 
link: 
tags:
  - High-Level-Note
---

# System Design

1. **Requirement Analysis:**
	- [[1 - 1 - 1 - Requirements Lifting]]
	- [[1 - 1 - 2 - Business Rules]]
	- [[1 - 1 - 3 - Architecture and Design]]
2. **Implementation Planning:**
	- [[1 - 3 - 1 - Functional Division]]
	- [[1 - 3 - 2 - Scrum Planning]]
3. **Control and Monitoring**
	- [[2 - 0 - Scrum Review]]

# System Proposal

- **Description and definitions of system:** [[1 - 2 - 1 - Technical System Documentation]]
- **General Figure:**
![[MARL Factory.drawio (1).png]]

- **Technical Proposal:** [[1 - 2 - 2 - System Proposal]]
- **File Structural:**

```
project/
│
├── config/
│   ├── user_config.json          # Configuración con todos los parámetros
│   └── config.py                 # Procesamiento y distribución de config
│
├── agents_factory/               # Módulo de agentes RL
│   ├── __init__.py
│   ├── base_agent.py             # Clase interfaz para los agentes
│   ├── agent_factory.py          # Fábrica para creación de agentes RL
│   ├── q_learning_agent.py       # Implementación del agente Q-Learning
│   └── ...                       # Otros agentes (DQN, PPO, etc.)
│
├── controllers_factory/          # Módulo de controladores
│   ├── __init__.py
│   ├── base_controller.py        # Clase con métodos comunes de controladores
│   ├── controller_factory.py     # Fábrica para creación de controladores
│   ├── pid_controller.py         # Implementación del controlador PID
│   └── ...                       # Otros controladores
│
├── environments_factory/         # Módulo de entornos de simulación
│   ├── __init__.py
│   ├── base_environment.py       # Interfaz para definir entornos
│   ├── env_factory.py            # Fábrica para creación de entornos
│   ├── inverted_pendulum.py      # Ejemplo de entorno
│   └── ...                       # Otros entornos
│
├── simulation/                   # Módulo para la lógica de simulación
│   ├── __init__.py
│   ├── simulation_engine.py      # Motor de simulación: gestiona episodios, pasos de tiempo y evaluación de recompensas
│   └── episode_manager.py        # Gestión y separación de episodios (por ejemplo, agrupación en archivos JSON)
│
├── evaluation/                   # Módulo de evaluación y reportes
│   ├── __init__.py
│   ├── evaluator.py              # Herramientas para evaluar el desempeño
│   └── metrics.py                # Funciones para el cálculo de métricas (convergencia, estabilidad, etc.)
│
├── utils/                        # Utilidades comunes
│   ├── __init__.py
│   ├── logger.py                 # Gestión de logs y depuración
│   └── helpers.py                # Funciones auxiliares varias (manejo de errores, validaciones, etc.)
│
├── main.py                      # Orquesta la configuración, instanciación y ejecución de la simulación
├── README.md                    # Documentación general del proyecto
└── requirements.txt             # Dependencias y requerimientos del proyecto

```

- **:**

