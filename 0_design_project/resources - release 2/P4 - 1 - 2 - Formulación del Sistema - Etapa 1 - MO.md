---
created: 20251202 01:12
update: 20251202-03:01
summary:
status:
link:
tags:
---
# R2 - E1 - 2 - Documentación del Sistema  - Multi-objective

## Versión inicial
La transición desde un sistema de aprendizaje local con recompensas compuestas a un sistema de optimización genuinamente multiobjetivo (MORL) se formaliza como un **Proceso de Decisión de Markov Multiobjetivo (MOMDP)**.

En específico, la optimización multiobjetivo se formula como una capa externa sobre la Lagrangiana de lazo ya definida, sin alterar ni la semántica física de las métricas ni el esquema de asignación de crédito entre agentes. El punto de partida es reconocer que el sistema no optimiza un único escalar abstracto, sino diferentes objetivos físicos bien diferenciados, como rapidez en el seguimiento, suavidad del esfuerzo de control y seguridad frente a saturaciones.

### 1. Objetos formales del problema multiobjetivo
El lazo se organiza en intervalos $(t^-,t]$ sobre los que se calculan métricas intensivas:

$$L_e,\;L_{\dot e},\;L_I,\;L_u,\;L_{\text{mov}},\;L_{\tilde{s}}$$

definidas como promedios o acumulados normalizados a lo largo del intervalo. A partir de ellas se definen tres objetivos:
$$\begin{aligned} J_{\text{fast}} &= f_{\text{fast}}(L_e, L_{\dot e}, L_I),\\ J_{\text{smooth}} &= f_{\text{smooth}}(L_u, L_{\text{mov}}),\\ J_{\text{safety}} &= f_{\text{safety}}(L_{\tilde{s}}), \end{aligned}$$​donde cada $f_{(\bullet)}$​ es una combinación lineal. *(Quizás se debería cambiar $L_{mov}$ por $L_{\dot{e}}$)???*

El desempeño de un intervalo queda así recogido por el vector:
$$\mathbf{J}_t = \big(J_{\text{fast},t}, J_{\text{smooth},t}, J_{\text{safety},t}\big)$$
El módulo de asignación de crédito conserva la misma estructura que en el caso escalar. La pérdida total del lazo se escribe como:
$$\ell_t = \alpha L_e+\beta L_{\dot e}+\gamma L_I+\eta L_u+ \zeta L^{(g)}_{\text{mov}} +\psi L_\tilde{s}$$

y se descompone en tres pérdidas específicas para los agentes proporcional, integral y derivativo:
$$\mathcal{L}_t^{(g)}=\alpha\,w_{e}^{(g)}L_e+\beta\,w_{\dot e}^{(g)}L_{\dot e}+\gamma\,w_{I}^{(g)}L_I+\eta\,w_u^{(g)}L_u+\zeta\,L_{\text{mov}}^{(g)}+\psi\,w_{s}^{(g)}L_\tilde{s}$$
con $g\in\{p,i,d\}$, y partición de unidad por métrica, tal que $w_{e}^{(p)}+w_{e}^{(i)}+w_{e}^{(d)}=1$ 

Por construcción, $\mathcal{L}_t = \sum_g \mathcal{L}_t^{(g)}$ y las recompensas específicas se mantienen como $r_t^{(g)} = -\mathcal{L}_t^{(g)}$​.

El nuevo objeto formal que introduce la formulación multiobjetivo es un vector de pesos entre objetivos:
$$\boldsymbol{\lambda} = (\lambda_{\text{fast}}, \lambda_{\text{smooth}}, \lambda_{\text{safety}})$$

con $\lambda_j \ge 0$ y $\sum_j \lambda_j=1$. Este vector parametriza la preferencia de alto nivel entre rapidez, suavidad y seguridad.

Para un $\boldsymbol{\lambda}$ dado, la pérdida escalar de intervalo que ve el agente ahora se define como:

$$\ell_t(\boldsymbol{\lambda}) = \lambda_{\text{fast}} J_{\text{fast},t} + \lambda_{\text{smooth}} J_{\text{smooth},t} + \lambda_{\text{safety}} J_{\text{safety},t}$$
y la recompensa de lazo sigue siendo $r_t = -\ell_t(\boldsymbol{\lambda})$. La asignación de crédito simplemente reemplaza $\ell_t$​ por la versión que incorpora estos pesos, sin cambiar la estructura de $\mathcal{L}_t^{(g)}$​ ni los $w_{(\bullet)}^{(g)}$​. *(debieran ser fijados por sensibilidad canónica?? y/o por régimen??, analizar)*.

Finalmente, para cada $\boldsymbol{\lambda}$ y política $\pi$ inducida por el entrenamiento con esa configuración, se define el vector de objetivos de largo plazo como
$$\bar{\mathbf{J}}(\boldsymbol{\lambda},\pi) = \Big( \mathbb{E}[J_{\text{fast}}],\, \mathbb{E}[J_{\text{smooth}}],\, \mathbb{E}[J_{\text{safety}}] \Big)$$
donde la esperanza se toma con respecto a la dinámica del lazo bajo la política estacionaria $\pi$ y sobre un conjunto estándar de condiciones iniciales y regímenes de operación.

### 2. Criterio de optimalidad multiobjetivo
Dado que el controlador PID opera de forma continua y la política del agente se ejecuta repetidamente para ajustar las ganancias, el criterio de optimización apropiado es el **Retorno Esperado Escalarizado (SER)**. El SER se calcula primero computando el retorno vectorial esperado de una política, y luego aplicando la función de utilidad u a esa expectativa: $Vuπ​=u(E[i=0∑∞​γiRi​∣π,s0​])$

En este marco, el problema fundamental ya no es minimizar una única $\ell_t$​, sino decidir qué compromisos entre $(J_{\text{fast}},J_{\text{smooth}},J_{\text{safety}})$ son aceptables o deseables.

Desde una perspectiva formal, el criterio de optimalidad se plantea en dos niveles:
1. A nivel interno, para un $\boldsymbol{\lambda}$ fijado, el algoritmo de RL maximiza el retorno escalar
$$\mathcal{R}(\boldsymbol{\lambda},\pi) = \mathbb{E}\Big[\sum_{t} \gamma^t r_t(\boldsymbol{\lambda})\Big] = - \mathbb{E}\Big[\sum_t \gamma^t \ell_t(\boldsymbol{\lambda})\Big]$$
	lo que equivale a optimizar una escalarización lineal de los tres objetivos. Esto define, para cada $\boldsymbol{\lambda}$, un problema estándar de control óptimo con una única función de recompensa.
2. A nivel externo, el interés recae en el conjunto de pares $(\boldsymbol{\lambda},\bar{\mathbf{J}}(\boldsymbol{\lambda},\pi_{\boldsymbol{\lambda}}))$, donde $\pi_{\boldsymbol{\lambda}}$​ es la política que maximiza $\mathcal{R}(\boldsymbol{\lambda},\cdot)$. Entonces el criterio de optimalidad multiobjetivo se formula como la identificación de soluciones no dominadas: aquellas para las cuales no existe otra configuración de pesos y política que mejore simultáneamente en rápido, suave y seguro sin empeorar alguno de ellos.

De este modo, la noción de óptimo se desplaza de un único valor escalar a una frontera de Pareto en el espacio de $\bar{\mathbf{J}}$. El problema se convierte en construir y caracterizar esa frontera, y luego seleccionar (o parametrizar) puntos sobre ella según restricciones de ingeniería y preferencias del diseñador.

## Problema operativo
La propuesta en este punto, requiere de un proceso iterativo off-line:
1. Se asigna un determinado vector de pesos $\boldsymbol{\lambda}$.
2. Se ejecuta el entrenamiento hasta obtener una política estacionaria (o suficientemente estable) para ese $\boldsymbol{\lambda}$.
3. Se evalúa esa política sobre varios episodios y se estiman los **retornos medios de cada objetivo**:
$$\bar{\mathbf{J}}(\boldsymbol{\lambda},\pi) \approx \Big( \mathbb{E}[J_{\text{fast}}|\boldsymbol{\lambda}],\, \mathbb{E}[J_{\text{smooth}}|\boldsymbol{\lambda}],\, \mathbb{E}[J_{\text{safety}}|\boldsymbol{\lambda}] \Big)$$
Donde, este vector es un punto en el espacio de objetivos, es decir, bajo ese esquema de pesos, ¿qué tan rápido, suave y seguro se comporta el lazo?

El problema, es que esto requiere explorar $\boldsymbol{\lambda}$ a mano (mediante barrido grueso), y así construir un conjunto de puntos en el espacio $(J_{\text{fast}},J_{\text{smooth}},J_{\text{safety}})$ que aproximen la frontera de Pareto, pero el coste en simulaciones y entrenamientos asincrónicos sería alto, sin mencionar los riesgos de no converger hacia políticas óptimas con pesos límites.

## Versión Extendida
Puesto que la combinación exacta de las preferencias entre $(J_{\text{fast}},J_{\text{smooth}},J_{\text{safety}})$​ es desconocida *a priori*, se podrían adoptar tanto **algoritmos de soporte de decisión** o incluso propuestas de **escenarios de función de utilidad dinámica**. En cualquier caso, el entrenamiento de sistemas en contextos multi-objetivo requiere un enfoque de múltiples políticas (multi-policy algorithm) para producir un conjunto de soluciones, en lugar de una única política. Donde, el agente busca aproximar la Frontera de Pareto (PF), que representa el conjunto de políticas no dominadas para cualquier función de utilidad monótonamente creciente.

### 1. El Proceso Gaussiano como meta-modelo sobre los pesos
La idea central es tratar a $\boldsymbol{\lambda}$ como variable de diseño y a $\bar{\mathbf{J}}(\boldsymbol{\lambda})$ como respuesta costosa de evaluar (porque detrás hay un entrenamiento RL completo). El Proceso Gaussiano (GP) se usa como mecanismo para aproximar el mapeo:
$$\boldsymbol{\lambda} \;\mapsto\; \bar{\mathbf{J}}(\boldsymbol{\lambda})$$
#### Implementación de Alto nivel
En la práctica, el bucle externo, principalmente se compone de:
1. **Datos iniciales.**
    - Se elige un pequeño conjunto de vectores de pesos $\{\boldsymbol{\lambda}^{(k)}\}_{k=1}^{K_0}$ bien espaciados en el simplex (por ejemplo, combinaciones que prioricen casi exclusivamente cada objetivo y algunos puntos intermedios).
    - Para cada $\boldsymbol{\lambda}^{(k)}$, se entrena el sistema y se registran $\bar{\mathbf{J}}^{(k)} = \bar{\mathbf{J}}(\boldsymbol{\lambda}^{(k)})$.
2. **Ajuste del GP.**
    - Para cada objetivo (fast, smooth, safety) se ajusta un GP independiente o un GP multi-salida:
$$J_{\text{fast}} \sim GP(m_{\text{fast}},k_{\text{fast}}), \dots$$
    - La entrada del GP es $\boldsymbol{\lambda}$ (representada en $\mathbb{R}^2$, por ejemplo $\lambda_{\text{fast}},\lambda_{\text{smooth}}$​ y $\lambda_{\text{safety}} = 1 - \lambda_{\text{fast}} - \lambda_{\text{smooth}}$.
    - El GP modela tanto la media esperada de cada objetivo como la incertidumbre debida al ruido de simulación y a la variabilidad entre ejecuciones.
3. **Modelo de utilidad implícita.**
    - A partir del GP se define un criterio de adquisición que combina el deseo de mejorar la frontera Pareto (por ejemplo, aumentar el hipervolumen dominado) con el interés en reducir la incertidumbre en las regiones relevantes. Ese criterio propone un nuevo vector de pesos $\boldsymbol{\lambda}^{\text{nuevo}}$.
        - sobre $\bar{\mathbf{J}}$, se puede definir un criterio escalar que resuma el “valor” de cada combinación $\boldsymbol{\lambda}$: utilidad del decisor, indicadores tipo hipervolumen, métricas de no-dominancia, etc.
            - en la literatura de multiobjetivo se han usado GP para modelar directamente funciones de utilidad desconocidas del usuario y guiar la exploración de preferencia *(por ejemplo, esquemas tipo Gaussian-process Utility Thompson Sampling GUTS [Zintgraf et al, 2020])*.
4. **Se repite el ciclo.**
	- Entrenamiento de $\pi_{\boldsymbol{\lambda}^{\text{nuevo}}}$, evaluación de $\bar{\mathbf{J}}(\boldsymbol{\lambda}^{\text{nuevo}},\pi_{\boldsymbol{\lambda}^{\text{nuevo}}})$, actualización del GP y refinamiento de la aproximación de la frontera.

El GP así guía la exploración multi-objetivo, y dice, con incertidumbre cuantificada, qué se espera obtener en rápido/suave/seguro si se modifican los pesos entre objetivos, sin necesidad de re-entrenar para todos los $\boldsymbol{\lambda}$ posibles. De esta forma, el proceso de entrenamiento utiliza la información del GP para enfocarse en la región de la Frontera de Pareto más relevante, produciendo un Conjunto de Cobertura Convexo (CCS) de políticas deterministas. Este enfoque asegura que se obtengan políticas que son óptimas para un rango de ponderaciones de utilidad (preferencias)

#### Consideraciones
##### 1. Política única frente a conjunto de políticas
Esta formulación admite dos estrategias de alto nivel respecto a la política, las cuales no son excluyentes:
- En una primera opción, se busca una política única de compromiso. En este caso, el proceso Gaussiano se utiliza para explorar el espacio de $\boldsymbol{\lambda}$, localizar una configuración $\boldsymbol{\lambda}^\star$ que satisfaga las restricciones de los objetivos, y seleccionar la política $\pi_{\boldsymbol{\lambda}^\star}$​ como solución final.
- En una segunda opción, se acepta explícitamente la naturaleza multiobjetivo del problema y se mantiene un conjunto de políticas representativas de la frontera de Pareto. Cada política $\pi_{\boldsymbol{\lambda}^{(k)}}$​ se asocia a una región del simplex de pesos y a un punto característico de la frontera. El sistema puede entonces:
	- seleccionar la política apropiada según el régimen operativo (por ejemplo, regímenes que requieren rapidez frente a regímenes que priorizan robustez y seguridad), o
	- ofrecer al finalizar las simulaciones, un menú de soluciones etiquetadas con sus métricas $\bar{\mathbf{J}}$ para una caracterización y selección informada.

##### 2. Restricciones de seguridad y determinismo
Las restricciones de seguridad se formalizan directamente sobre el objetivo $J_{\text{safety}}$​. Dado que $J_{\text{safety}}$​ recoge el coste asociado a saturaciones de actuadores, o en el futuro, a cercanía de condiciones de operación críticas, se imponen cotas del tipo:
$$\bar{J}_{\text{safety}}(\boldsymbol{\lambda},\pi) \leq J_{\text{safety}}^{\text{max}}$$
donde $J_{\text{safety}}^{\text{max}}$ representa el nivel máximo tolerable de riesgo. Esta restricción puede implementarse de dos formas: como restricción dura, descartando cualquier solución que supere el umbral, o como penalización fuerte incorporada en la función de utilidad que guía la selección de $\boldsymbol{\lambda}$ en el nivel externo.

En cuanto al determinismo, la política operativa se define como una política determinista por construcción en fase de explotación. Durante el entrenamiento, se utilizan esquemas exploratorios clásicos (por ejemplo, $\varepsilon$-greedy) sobre las tablas Q; sin embargo, la política final $\pi_{\boldsymbol{\lambda}}$ se obtiene fijando $\varepsilon = 0$, de modo que para cada estado observado, la acción seleccionada es la que maximiza la Q estimada. Esto asegura reproducibilidad y coherencia con los requisitos de control industrial, donde las decisiones del lazo deben ser trazables y no aleatorias en operación nominal.

Además, la propia estructura de la función de recompensa refuerza la seguridad y el determinismo: el término de saturación $L_{\tilde{s}}$ y los términos específicos de cada agente como $L_{\text{mov}}^{(g)}$ penalizan políticas que abusen del actuador o cambien de bin de manera errática, induciendo políticas que, en la práctica, son más suaves y predecibles.

#### Protocolo de Evaluación
El protocolo de evaluación tiene como misión convertir cada configuración de pesos 
$$\boldsymbol{\lambda}\in\mathcal{S}_\lambda$$
en un conjunto de observables empíricos bien definidos:
1. El vector de desempeño estacionario por objetivo
$$\bar{\mathbf{J}}(\boldsymbol{\lambda})= \big(\bar{J}_{\text{fast}}(\boldsymbol{\lambda}), \bar{J}_{\text{smooth}}(\boldsymbol{\lambda}), \bar{J}_{\text{safety}}(\boldsymbol{\lambda})\big)$$
2. La utilidad escalar asociada
$$U(\boldsymbol{\lambda}) = F\big(\bar{\mathbf{J}}(\boldsymbol{\lambda})\big)$$
3. La decisión de si la política entrenada para esos pesos pasa a formar parte del Conjunto de Cobertura Convexo (CCS) que aproxima la frontera de Pareto.

Todo se construye sobre los mismos objetos de lazo ya definidos: las métricas de intervalo $L_{e},\; L_{\dot e},\;L_I,\;L_u,\;L_{\text{mov}},\;L_{\tilde{s}}$ y los objetivos de intervalo $(J_{\text{fast},t},\; J_{\text{smooth},t},\; J_{\text{safety},t})$ obtenidos como combinaciones lineales de esas métricas.

**Recapitulamos y volvemos a definir:**

##### 1. Desempeño de la política 
Para una configuración de pesos $\boldsymbol{\lambda}$ fija, se entrena una política estacionaria $\pi_{\boldsymbol{\lambda}}$
como ya se definió. La evaluación se realiza fuera del bucle de entrenamiento, sobre un conjunto de episodios generados de manera controlada.

Se considera un conjunto de episodios de evaluación  
$$\mathcal{E} = \{1,\dots,N_{\text{eval}}\}$$
Sea $\nu$ una distribución de condiciones de evaluación (combinaciones de condiciones iniciales, setpoints, perturbaciones, etc.). Para cada $\boldsymbol{\lambda}$ se generan $N_{\text{eval}}$ episodios independientes
$$\omega_E \sim \nu,\quad E=1,\dots$$
En el episodio $E$, la política $\pi_{\boldsymbol{\lambda}}$​ induce una secuencia de intervalos de decisión $W_{E,\,t}=(t^{-},t]$, y para cada intervalo se calculan los objetivos de intervalo:
$$\mathbf{J}_{\text{w},\,t}(\boldsymbol{\lambda}) = \big(J_{\text{fast},\, \text{w},\,t}, J_{\text{smooth},\, \text{w},\,t}, J_{\text{safety},\,\text{w},\,t}\big)$$
Los cuales, permiten establecer una medida resumen de mediano y largo plazo. Por ejemplo, a mediano plazo se pueden obtener los promedios temporales (o sumas normalizadas, etc) de cada objetivo en el episodio $E$:
$$\mathbf{J}_{E}(\boldsymbol{\lambda}) = \frac{1}{\#W_{E}}\sum_{\text{w}=1}^{\#W_{E}} \mathbf{J}_{E,\,\text{w}} = \left( J_{\text{fast},\,E}, J_{\text{smooth},\,E}, J_{\text{safety},\,E} \right)$$
donde $\#W_{\text{E}}$​ es la duración (en intervalos) del episodio. Y por otro lado, se define un estimador empírico del vector de objetivos a largo plazo como:
$$\hat{\bar{\mathbf{J}}}_{E}(\boldsymbol{\lambda}) = \frac{1}{N_{\text{eval}}}\sum_{E=1}^{N_{\text{eval}}} \mathbf{J}_{E}(\boldsymbol{\lambda}) = \left( \hat{\bar{J}}_{\text{fast},\,E}(\boldsymbol{\lambda}), \hat{\bar{J}}_{\text{smooth},\,E}(\boldsymbol{\lambda}), \hat{\bar{J}}_{\text{safety},\,E}(\boldsymbol{\lambda}) \right)$$
Estimador que converge a la evaluación estacionaria de la política $\bar{\mathbf{J}}(\boldsymbol{\lambda})$ para ese $\boldsymbol{\lambda}$ cuando $N_{\text{eval}}$ crece lo suficiente, y que por ende, representa el valor esperado de $\mathbf{J}_{E}$ bajo la distribución conjunta de condiciones iniciales, ruido y dinámica del lazo:
$$\bar{\mathbf{J}}(\boldsymbol{\lambda}) = \mathbb{E}_{{\nu},\,  P(\cdot\,|\pi_{\boldsymbol{\lambda}})} \big[\,\mathbf{J}_{E}(\boldsymbol{\lambda})\,\big] \approx \hat{\bar{\mathbf{J}}}_{E}(\boldsymbol{\lambda})$$
En paralelo, para cuantificar la incertidumbre asociada a la evaluación, se calculan también las varianzas y covarianzas empíricas de los objetivos:
$$\hat{\Sigma}(\boldsymbol{\lambda}) = \frac{1}{N_{\text{eval}}-1}\sum_{E=1}^{N_{\text{eval}}} \left(\mathbf{J}_{E}(\boldsymbol{\lambda}) - \hat{\bar{\mathbf{J}}}(\boldsymbol{\lambda})\right) \left(\mathbf{J}_{E}(\boldsymbol{\lambda}) - \hat{\bar{\mathbf{J}}}(\boldsymbol{\lambda})\right)^{\top}$$
lo que permite añadir intervalos de confianza y tomar decisiones sobre dominancia o no dominancia de forma robusta frente al ruido de simulación.

##### 2. Función de utilidad y restricciones de seguridad
Entonces, y volviendo a la optimización multiobjetivo. La misión del algoritmo de proceso gaussiano propuesto es aprender una función de utilidad sobre el simplex de pesos y, a partir de ella, seleccionar la combinación de rapidez, suavidad y seguridad que debe usar el sistema.

Por lo tanto, con $\mathcal{S_{\lambda}}$ como el simplex de pesos entre objetivos:
$$\mathcal{S_{\lambda}} = \left\{ \boldsymbol{\lambda} = (\lambda_{\text{fast}},\lambda_{\text{smooth}},\lambda_{\text{safety}}) : \lambda_j \ge 0,\ \sum_j \lambda_j = 1 \right\}$$
La función de utilidad global:
$$U : \mathcal{S_{\lambda}} \to \mathbb{R}$$
que asigna a cada $\boldsymbol{\lambda}$ una medida de calidad global de la solución resultante, es resumida por el escalar multiobjetivo resultante del desempeño estacionario de la política entrenada bajo esos pesos:
$$U(\boldsymbol{\lambda}) = F\big(\bar{\mathbf{J}}(\boldsymbol{\lambda})\big)$$
El funcional $F$ es el lugar donde se incorporan explícitamente las preferencias de diseño y las restricciones de seguridad. Además, recordar que todos los $J_{(\bullet)}$​ son costes, por lo que $F$ debe ser monótonamente decreciente en cada componente. Una construcción genérica podría ser:
$$F(\bar{\mathbf{J}}) = f_{\text{pref}}(\bar{J}_{\text{fast}},\bar{J}_{\text{smooth}}) + M \,\max\big(0,\;\bar{J}_{\text{safety}} - J_{\text{safety}}^{\max}\big)$$
donde:
- $f_{\text{pref}}$​ es un funcional que recoge las preferencias entre rapidez y suavidad (por ejemplo, una combinación lineal que refleja la importancia relativa de cada objetivo al seleccionar soluciones dentro de la frontera de Pareto, complementado con transformaciones monótonas como reescalados o función convexa que modelan sensibilidad no lineal a cada objetivo, o simplemente un esquema lexicográfico, etc),
- $⁡J_{\text{safety}}^{\max}$​ es el nivel máximo tolerable de coste de seguridad, definido por ingeniería,
- $M\gg 1$ es una penalización suficientemente grande para que cualquier violación de seguridad domine la utilidad.

##### 3. Dominancia, conjunto de cobertura y aceptación de políticas
El tercer elemento del protocolo es decidir si la política entrenada con $\boldsymbol{\lambda}$ debe incorporarse al CCS que aproxima la frontera de Pareto.

Sea $\mathcal{C}$ el conjunto actual de candidatos aceptados, almacenando pares
$$\mathcal{C} = \big\{ (\boldsymbol{\lambda}^{(k)},\hat{\bar{\mathbf{J}}}_E(\boldsymbol{\lambda}^{(k)})) \big\}_{k\in\mathcal{K}}$$
Se trabaja en el espacio de costos, por lo que “mejor” significa “menor o igual en todos los objetivos y estrictamente menor en al menos uno”. Entonces, dado el candidato actual
$$\mathbf{j}(\boldsymbol{\lambda}) \;\equiv\; \hat{\bar{\mathbf{J}}}_E(\boldsymbol{\lambda}) = \big( \hat{\bar{J}}_{\text{fast},\, E}(\boldsymbol{\lambda}), \hat{\bar{J}}_{\text{smooth},\, E}(\boldsymbol{\lambda}), \hat{\bar{J}}_{\text{safety},\, E}(\boldsymbol{\lambda}) \big)$$
se dice que un punto existente $\mathbf{j}(\boldsymbol{\lambda}^{(k)})$ domina al candidato si
$$\mathbf{j}(\boldsymbol{\lambda}^{(k)}) \preceq \mathbf{j}(\boldsymbol{\lambda}) \quad\text{y}\quad \mathbf{j}(\boldsymbol{\lambda}^{(k)}) \neq \mathbf{j}(\boldsymbol{\lambda})$$

es decir,
$$\hat{\bar{J}}_{j,E}(\boldsymbol{\lambda}^{(k)}) \;\le\; \hat{\bar{J}}_{j,E}(\boldsymbol{\lambda})\quad \wedge \quad \hat{\bar{J}}_{j,E}(\boldsymbol{\lambda}^{(k)}) < \hat{\bar{J}}_{j,E}(\boldsymbol{\lambda}) \quad \forall j \in \{\text{fast},\text{smooth},\text{safety}\}$$

El protocolo implementa entonces la regla:
1. **Rechazo por dominancia:**  
    Si $\exists\, k\in\mathcal{K}$ tal que, $\mathbf{j}(\boldsymbol{\lambda}^{(k)}) \preceq \mathbf{j}(\boldsymbol{\lambda})$ entonces la política $\pi_{\boldsymbol{\lambda}}$ se considera dominada y no se incorpora al CCS.
2. **Aceptación y poda:**  
    Si ningún punto en $\mathcal{C}$ domina a $\mathbf{j}(\boldsymbol{\lambda})$, el candidato se incorpora:
$$\mathcal{C} \leftarrow \mathcal{C} \cup \{(\boldsymbol{\lambda},\mathbf{j}(\boldsymbol{\lambda}))\}$$
    Además, se eliminan todos los puntos que resulten dominados por el nuevo candidato:
$$\mathcal{C} \leftarrow \big\{(\boldsymbol{\lambda}^{(k)},\mathbf{j}(\boldsymbol{\lambda}^{(k)}))\in\mathcal{C} \;:\; \mathbf{j}(\boldsymbol{\lambda}) \not\preceq \mathbf{j}(\boldsymbol{\lambda}^{(k)}) \big\}$$
Con esto, el CCS mantiene siempre un conjunto de políticas no dominadas en el espacio de costos $(J_{\text{fast}},J_{\text{smooth}},J_{\text{safety}})$, y el protocolo de evaluación provee, para cada $\boldsymbol{\lambda}$ visitado, el conjunto de métricas operacionales:
$$\mathcal{E}(\boldsymbol{\lambda}) = \Big( \hat{\bar{\mathbf{J}}}_E(\boldsymbol{\lambda}), \hat{U}(\boldsymbol{\lambda}), \delta_{\text{CCS}}(\boldsymbol{\lambda}) \Big)$$
donde $\delta_{\text{CCS}}(\boldsymbol{\lambda})\in \{0,1\}$ indica si la política asociada a $\boldsymbol{\lambda}$ ha sido finalmente incluida ($1$) o no ($0$) en el conjunto de cobertura convexo que aproxima la frontera de Pareto del sistema multiobjetivo.

Adicionalmente, para cuantificar el progreso global en la exploración de la frontera, el protocolo puede calcular indicadores de calidad de aproximación como el **hipervolumen dominado**. Dado un vector de referencia $\mathbf{r}$ que representa un límite superior admisible de costes, el hipervolumen del conjunto no dominado $\mathcal{P}$ se define como la medida de volumen en $\mathbb{R}^3$ del conjunto:
$$\text{HV}(\mathcal{P}) = \mu\left( \bigcup_{(\boldsymbol{\lambda}^{(k)},\hat{\bar{\mathbf{J}}}^{(k)})\in\mathcal{P}} [\hat{\bar{\mathbf{J}}}^{(k)},\,\mathbf{r}] \right)$$

donde $[\hat{\bar{\mathbf{J}}}^{(k)},\,\mathbf{r}]$ es la región del espacio delimitado por $\hat{\bar{\mathbf{J}}}^{(k)}$ y $\mathbf{r}$. A mayor hipervolumen, mejor cobertura de la región de interés de la frontera.

##### 4. Integración del GP
Sea $\mathcal{D}_n = {(\boldsymbol{\lambda}^{(k)}, \hat{\bar{\mathbf{J}}}_E(\boldsymbol{\lambda}^{(k)}))}_{k=1}^n$ el conjunto de configuraciones ya entrenadas y evaluadas por el protocolo, y sea el GP ajustado el que entrega, para cada nuevo $\boldsymbol{\lambda}$, una distribución posterior gaussiana aproximada para el vector de objetivos de largo plazo:
$$\bar{\mathbf{J}}(\boldsymbol{\lambda}) \mid \mathcal{D}_n \approx \mathcal{N}\big(\boldsymbol{\mu}_J(\boldsymbol{\lambda}),\,\Sigma_J(\boldsymbol{\lambda})\big)$$
donde $\boldsymbol{\mu}_J(\boldsymbol{\lambda})$ y $\Sigma_J(\boldsymbol{\lambda})$ son, respectivamente, la media (medida de largo plazo) y la covarianza predichas por el GP para los tres objetivos rápido/suave/seguro.

Luego, sobre estos objetivos se definió una función de utilidad escalar $F:\mathbb{R}^3\to\mathbb{R}$, monótona decreciente en cada componente, que resume el compromiso entre rapidez, suavidad y seguridad en una única magnitud $U(\boldsymbol{\lambda}) = F(\bar{\mathbf{J}}(\boldsymbol{\lambda}))$. A partir del GP vectorial se obtiene una aproximación gaussiana para la utilidad:
$$U(\boldsymbol{\lambda}) \mid \mathcal{D}_n \approx \mathcal{N}\big(\mu_U(\boldsymbol{\lambda}),\,\sigma_U^2(\boldsymbol{\lambda})\big)$$
donde la media se aproxima evaluando $F$ en la media de los objetivos,
$$\mu_U(\boldsymbol{\lambda}) \;\approx\; F\big(\boldsymbol{\mu}_J(\boldsymbol{\lambda})\big)$$
y la varianza se obtiene por linealización de primer orden de $F$ alrededor de $\boldsymbol{\mu}_J(\boldsymbol{\lambda})$:
$$\sigma_U^2(\boldsymbol{\lambda}) \;\approx\; \nabla_{\mathbf{z}}F\big(\boldsymbol{\mu}_J(\boldsymbol{\lambda})\big)^{\top} \,\Sigma_J(\boldsymbol{\lambda})\, \nabla_{\mathbf{z}}F\big(\boldsymbol{\mu}_J(\boldsymbol{\lambda})\big)$$
siendo $\nabla_{\mathbf{z}}F$ el gradiente de la utilidad con respecto al vector de objetivos. Esta aproximación es estándar en modelos de GP con utilidad compuesta y permite trabajar con una sola variable aleatoria escalar $U(\boldsymbol{\lambda})$ aun cuando el GP original es multi-salida.

Con esta distribución aproximada se define el mejor valor de utilidad observado hasta el momento:
$$U_n^\star \;=\; \max_{1\le k\le n} F\big(\hat{\bar{\mathbf{J}}}_E(\boldsymbol{\lambda}^{(k)})\big)$$
y se introduce la **mejora esperada** (Expected Improvement, EI) de una nueva configuración de pesos $\boldsymbol{\lambda}$ respecto de ese máximo actual. La mejora puntual es:
$$I(\boldsymbol{\lambda}) = \big(U(\boldsymbol{\lambda}) - U_n^\star\big)_+ = \max\big(U(\boldsymbol{\lambda}) - U_n^\star,\;0\big)$$
y el criterio de adquisición es su esperanza bajo la distribución posterior del GP:
$$\alpha_{\text{EI}}(\boldsymbol{\lambda}) = \mathbb{E}\big[I(\boldsymbol{\lambda}) \mid \mathcal{D}_n\big]$$
Bajo la aproximación gaussiana $U(\boldsymbol{\lambda})\sim\mathcal{N}(\mu_U(\boldsymbol{\lambda}),\sigma_U^2(\boldsymbol{\lambda}))$, esta esperanza se puede escribir en forma cerrada como:
$$\alpha_{\text{EI}}(\boldsymbol{\lambda}) = \big(\mu_U(\boldsymbol{\lambda}) - U_n^\star\big)\,\Omega\big(z(\boldsymbol{\lambda})\big) + \sigma_U(\boldsymbol{\lambda})\,\Lambda\big(z(\boldsymbol{\lambda})\big)$$
donde
$$z(\boldsymbol{\lambda}) = \frac{\mu_U(\boldsymbol{\lambda}) - U_n^\star}{\sigma_U(\boldsymbol{\lambda})}$$
$\Omega$ es la función de distribución acumulada de una normal estándar y $\Lambda$ su densidad. Cuando $\sigma_U(\boldsymbol{\lambda})$ es muy pequeña (punto ya casi conocido), el término de exploración se apaga; cuando es grande, la segunda componente domina y empuja a explorar regiones de alta incertidumbre. El criterio $\alpha_{\text{EI}}$ codifica de forma automática el compromiso de **exploración-explotación**, ya que, busca pesos con alta utilidad esperada (primer término) pero que todavía no estén bien caracterizados (segundo término).

Para incorporar las restricciones de seguridad que ya se expresan sobre el objetivo asociado a saturaciones, se define una versión “segura” del criterio, anulando la mejora esperada en pesos cuya predicción sea incompatible con el umbral de riesgo aceptable. Sea $\mu_{\text{safety}}(\boldsymbol{\lambda})$ y $\sigma_{\text{safety}}(\boldsymbol{\lambda})$ la media y desviación estándar predichas por el GP para el objetivo de seguridad y $J_{\text{safety}}^{\max}$ el límite permitido; entonces se introduce un test de factibilidad probabilística
$$\text{safe}(\boldsymbol{\lambda}) = \mathbf{1}\!\left[ \mu_{\text{safety}}(\boldsymbol{\lambda}) + \kappa\,\sigma_{\text{safety}}(\boldsymbol{\lambda}) \;\le\; J_{\text{safety}}^{\max} \right]$$
con $\kappa>0$ un parámetro de confianza (por ejemplo, $\kappa=2$ para exigir que incluso un intervalo de confianza amplio respete el umbral). Así, el criterio de adquisición efectivo queda
$$\alpha_{\text{safe-EI}}(\boldsymbol{\lambda}) = \text{safe}(\boldsymbol{\lambda})\; \alpha_{\text{EI}}(\boldsymbol{\lambda})$$
de modo que cualquier vector de pesos cuya predicción apunte hacia regiones inseguras recibe adquisición cero y no se propone para entrenamiento.

Entonces, el bucle externo selecciona el siguiente vector de pesos resolviendo
$$\boldsymbol{\lambda}^{(n+1)} \in \arg\max_{\boldsymbol{\lambda}\in\mathcal{S}_\lambda} \alpha_{\text{safe-EI}}(\boldsymbol{\lambda})$$
generalmente sobre un conjunto finito de candidatos muestreados en el simplex. Ese $\boldsymbol{\lambda}^{(n+1)}$ define la nueva configuración multiobjetivo para la que se entrena una política $\pi_{\boldsymbol{\lambda}^{(n+1)}}$, se evalúa mediante el protocolo de episodios y se incorpora a $\mathcal{D}_{n+1}$, actualizando el GP y el CCS asociado. De esta forma, el proceso gaussiano funciona como un meta-modelo que sugiere de manera sistemática las configuraciones de pesos más informativas y prometedoras, guiando el escalamiento multiobjetivo del sistema sin necesidad de barridos manuales densos sobre el espacio de $\boldsymbol{\lambda}$.
