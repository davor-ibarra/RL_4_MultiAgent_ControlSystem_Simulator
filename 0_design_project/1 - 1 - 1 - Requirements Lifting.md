---
created: 20241010 05:10
update: 20250305-06:50
summary: 
status: done
link: 
tags:
  - content
---
# Preguntas Claves para la definición del sistema
## Requerimientos Generales y Objetivos del Sistema

- ¿Cuál es el objetivo principal del sistema? ¿Se centra únicamente en la optimización de parámetros de controladores PID o se espera que sea extensible a otros tipos de controladores?
	- El objetivo principal del sistema es la optimización de controladores en sistemas de control multiobjetivos, Por lo que se debería poder utilizar a los multiagentes como controladores de forma directa, o incluso el caso de coexistir con otros controladores, por lo que se podría escalar en el futuro a la inclusión de otros controladores pero comenzaremos con el controlador PID.
- ¿Cuál es la visión a largo plazo del sistema? ¿Se busca un prototipo para pruebas o una solución lista para producción que integre múltiples entornos y agentes?
	- A largo plazo, se podrían construir variados entornos, se podrían intregrar variados tipos de controladores, y para variados algoritmos de RL o ML para la optimización dinámica en linea con sistemas multiagentes de estos controladores o al sistema directo. Por lo que, se requiere un prototipo minimo viables para su escalamiento progresivo.

## Especificaciones del Entorno y Simulación

- ¿Cómo se define el entorno a controlar? ¿Se trata de sistemas físicos reales, simulaciones puras o una combinación de ambos?
	- Simulaciones mediante ecuaciones diferenciales ordinarias de sistemas físicos, de procesos y cualquiera sea la lógica que se quiera establecer mientras cuente con una entrada para la acción y una salida para el estado siguiente.
- ¿Cuáles son los parámetros del entorno (por ejemplo, tiempo de simulación, frecuencia de actualización, condiciones iniciales) y cómo se espera que interactúe con los agentes?
	- Se espera que la clase de Simulación logre establecer una lógica flexible de entrenamiento por episodios que incluya ventanas de tiempo para la evaluación del comportamiento del controlador despues de la acción de optimización del sistema multiagente, la duración de cada episodio será determinado por un tiempo máximo, y el paso de tiempo por un pequeño delta de tiempo. Respecto a las condiciones iniciales estas pueden ser configuradas para que se inicialicen en cada episodio de entrenamiento, o que cada episodio comience exactamente igual a como se encontraba en el último tiempo del episodio anterior. Finalmente, el agente podría aplicar una determinada acción "u" sobre la función que determina el comportamiento del entorno, aunque en principio la idea es que el entorno del agente sean un rango de valores de los hiperparametros del controlador, y las acciones de bajar, mantener o subir una cierta cantidad de valor; por tanto el controlador accionaría sobre el entorno, y durante la ventana de tiempo luego de esa acción, se evalúa una función temporal de recompensa acumulada, permitiendo al agente mejorar su acción sobre el controlador. En el caso específico del PID sería una tabla Q para cada ganancia (kp, ki, kd) y la posibilidad de adicionar agentes de consenso, comunicación, directores, y otros, para la optimización de las ganancias del controlador.
- ¿La simulación se ejecutará en tiempo real o en modo acelerado? ¿Qué nivel de fidelidad se requiere?
	- La escala de tiempo se establece por la variable que determina el delta de tiempo "dt". Alta fidelidad

## Controladores y Ajuste PID

- ¿Qué tipos de controladores se espera soportar? ¿Solo PID o también otros tipos (adaptativos, robustos, etc.)?
	- Se espera soportar todo tipo de controladores, incluso al mismo sistema multiagente se le podrían asignar como variables de estado el error o del entorno, para que luego pueda realizar una determinada acción sobre el entorno.
- Podrían ser otros, pero comenzaremos con el PID
- ¿Cuáles son los parámetros específicos del controlador PID que deben ser ajustables (ganancias proporcional, integral y derivativa, límites, etc.)?
	- Ganancias proporcional, integral y derivativa y límites
- ¿Existen restricciones o rangos esperados para estos parámetros?
	- Deben poder ser determinados por el usuario

## Agentes y Algoritmos de Aprendizaje por Refuerzo

- ¿Qué algoritmos de RL se deben soportar inicialmente (por ejemplo, Q-learning, DQN, PPO, SAC, TD3, etc.)? ¿Se prevé la inclusión de algoritmos adicionales en el futuro?
	- Q-Learning en principio, luego se podrán incluir todos esos y más
- ¿Cuál es el criterio de selección o el método de comparación entre diferentes algoritmos? ¿Qué métricas serán prioritarias (velocidad de convergencia, estabilidad, consumo de recursos, etc.)?
	- Velocidad de convergencia y estabilidad
- ¿Cómo se gestionará la interacción y coordinación en el entorno multiagente? ¿Se opta por una arquitectura centralizada, distribuida o híbrida?
	- Debe poder ser determinada por el usuario, incluso podría ser un agente Q que actúe mediante diferentes agentes en base a ciertas variables como estados

## Arquitectura y Diseño del Sistema

- ¿Existen estándares o restricciones en cuanto a la estructura de carpetas y la organización de módulos?
	- No
- ¿Se debe partir de algún framework o librería existente, o la intención es desarrollar la solución desde cero?
	- No se utilizarán librerías para la simulación, entorno o algoritmos de control, la intención es desarrollar la solución desde 0, sin perjuicio que se puedan utilizar algunas librerías elementales para el procesamiento, validación y gestión del código y rendimientos, mas no para las lógicas del sistema. Aunque la idea es que sea lo suficientemente flexible para que en el futuro se puedan integrar entornos externos, como los que se pudieran crear con Gym o Matlab.
- ¿Cómo se integrará el patrón de fábrica en la creación de agentes y controladores? ¿Se requiere algún nivel de abstracción o interfaz común para facilitar la extensión?
	- Todo debe ser configurable por un archivo JSON con los parámetros necesarios
- ¿Qué principios SOLID y de Clean Code son prioritarios en este contexto y cómo se espera que se reflejen en la implementación?
	- Que el sistema sea lo más modular posible para aumentar la flexibilidad por lo que se espera que cada uno cuente con responsabilidades y roles bien definidas.

## Configuración y Extensibilidad

- ¿Qué parámetros y configuraciones se esperan incluir en el archivo `config.json`? ¿Se contempla que estos parámetros se puedan actualizar en caliente sin reiniciar el sistema?
	- Todos los necesarios. No es necesario que se actualicen en caliente.
- ¿Se requiere un sistema de plugins o módulos externos para permitir la incorporación de nuevos algoritmos o controladores sin modificar la base del código?
	- Claro.
- ¿Cómo se gestionarán las dependencias y la compatibilidad con futuras versiones de Python o de las librerías utilizadas?
	- No es necesario abordar esto.

## Escalabilidad y Paralelización

- ¿Cuál es el volumen esperado de agentes y simulaciones concurrentes? ¿Existen requerimientos específicos de rendimiento y consumo de recursos (CPU, memoria)?
	- Volumen no determinado y una simulación cada vez en principio. Los recursos limites deben poder ser configurados por el usuario.
- ¿El sistema debe estar preparado para ejecutarse en entornos distribuidos o en la nube, además del uso de multiprocessing en entornos locales?
	- Solo en entornos locales con la posibilidad de activar el multiprocessing
- ¿Qué estrategia se utilizará para el registro de logs y métricas? ¿Se prefiere TensorBoard, MLflow u otra herramienta?
	- Ninguno, se requiere extraer toda la metadata junto a la historia de la simulación por cada paso de tiempo, y luego nos ocupamos de su análisis.

## Módulos de Evaluación y Reporte

- ¿Qué métricas y criterios de evaluación son críticos para medir el desempeño de los agentes y el impacto en la optimización de los controladores?
	- Función de Recompensa y otros relevantes.
- ¿Cómo se estructurará la generación de reportes en archivos JSON? ¿Cada archivo contendrá un número fijo de episodios o se basará en un criterio de tamaño o tiempo?
	- Por cantidad de episodios, en principio 1000 pero podría ser configurado por el usuario.
- ¿Se requiere visualización en tiempo real de los resultados, o se tratará únicamente de análisis post-simulación?
	- Post-simulación

## Integración, Interfaz y Comunicación

- ¿Qué tipo de interfaz se espera para interactuar con el sistema: una API REST, una interfaz de línea de comandos, una interfaz gráfica, o una combinación de estas?
	- En principio se pasaría un json con todas las configuraciones necesarias, pero imagino que en el futuro integraré una interfaz gráfica y utilizaré este json para poder manejar la simulación.
- ¿Cómo se gestionará la comunicación entre los distintos módulos (Agente, Controlador, Entorno, Simulación, Evaluación)? ¿Se utilizará un sistema de eventos, colas de mensajes o APIs internas?
	- APIs internas.
- ¿Existen requisitos específicos de seguridad, autenticación o autorización en la interacción de módulos o con usuarios externos?
	- No.

## Mantenimiento y Documentación

- ¿Cuál es el nivel de documentación esperado en cuanto a docstrings, comentarios y guías de usuario? ¿Se utilizarán herramientas de documentación automática?
	- Lo más breve posible, descripción y valor esperado por cada variable
- ¿Cómo se gestionará el versionado y la integración continua (CI/CD) del proyecto?
	- Github
- ¿Qué estrategias se tienen previstas para el manejo de errores y la recuperación ante fallos en el sistema?
	- Debugging y validación continua.
