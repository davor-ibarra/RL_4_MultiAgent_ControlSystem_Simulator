---
created: 20251202 03:12
update: 20251205-16:45
summary:
status:
link:
tags:
---
# R2 - E1 - 3 - Documentación del Sistema  - Medidas de Variabilidad
## Entropía de Shannon
### 1. Contexto del sistema y espacio discreto
El contexto operativo que se observa en cada lazo no está determinado por un único valor, sino por un conjunto de estados y parámetros visitados durante ese intervalo. 

Sea
$$\mathbb{x}_t \in \mathbb{R}^d$$
el vector de estado observable del entorno en el instante $t$ (por ejemplo, error, variación del error, error acumulado, ganancias, acción de control, etc.), y sea
$$\mathbb{z}_t = \chi(\mathbb{x}_t)$$
el vector de características de contexto, donde el mapeo $\chi$ puede ser tan simple como seleccionar algunas de las variables observables o combinar estas con métricas del intervalo.

Luego, para poder definir una entropía sobre el contexto, el espacio continuo de características $\mathbb{Z} \subset \mathbb{R}^q$ se discretiza en una partición finita:
$$\mathbb{Z} = \bigcup_{i=1}^{N_{reg}} \mathbb{C}_i,\quad \mathbb{C}_i \cap \mathbb{C}_j = \varnothing\ (i\neq j)$$
donde cada celda $\mathbb{C}_i$​ representa un régimen local de contexto (por ejemplo, zona de error pequeño sin saturación, zona de error grande con alta variación de control, etc.). La elección de esta partición puede ser una malla cartesiana simple sobre unas pocas variables normalizadas, o incluso un conjunto de celdas obtenidas por clustering; lo relevante es que al definir la cantidad de etiquetas $N_{reg}$, estas podrán ser utilizadas para poder comparar entropías entre intervalos y políticas.

### 2. Distribución empírica de contexto en una ventana
El lazo que se organiza en ventanas de decisión $W=(t^{-},t]\, |\, \text{w} \in W$ sobre las que se calculan las métricas intensivas $L_e,L_{\dot e},L_I,L_u,\dots$ utilizadas en la Lagrangiana de intervalo, ahora también se definen distribuciones empíricas de contexto a partir de las muestras $\mathbb{z}_{\text{w}}$​ observadas en cada intervalo.

Sea $N_{\text{w}} = t - t^{-}$ el número de pasos de tiempo de la ventana. Para cada celda $\mathbb{C}_i$​ se cuenta cuántas veces el contexto coincide dentro de esos criterios:
$$n_i(\text{w}) = \sum_{\tau=t^{-}+1}^{t} \mathbf{1}\big[\mathbb{z}_\tau \in \mathbb{C}_i\big], \qquad i=1,\dots$$
y se define la distribución empírica de contexto
$$p_i(\text{w}) = \frac{n_i(\text{w})}{N}, \qquad i=1,\dots,N_{\text{w}}$$
Por construcción, $p_i(\text{w})\ge 0$ y $\sum_{i=1}^{N} p_i(\text{w}) = 1$. Esta distribución describe qué fracción de la ventana el lazo estuvo en cada región de contexto en el intervalo, de forma análoga a cómo las métricas intensivas describen el desempeño medio del intervalo. El contexto se convierte así en una variable aleatoria discreta
$$\mathbb{C}_\text{w} \sim p(\text{w}) = \big(p_1(\text{w},\dots,p_N(\text{w})\big)$$
donde $\mathbb{C}_\text{w}$ indica “en qué tipo de régimen” se encuentra el lazo en una muestra típica de la ventana $\text{w}$.

### 3. Entropía de Shannon como medida de variabilidad
Sobre la distribución empírica $p(\text{w})$ se define la entropía de Shannon como medida de variabilidad del contexto en la ventana:
$$\mathcal{H}(\text{w}) \;=\; \mathcal{H}\big(p(\text{w})\big) \;=\; -\sum_{i=1}^{N_{\text{w}}} p_i(\text{w})\,\log p_i(\text{w})$$
donde se adopta el logaritmo natural, expresión regular como medida de entropía. Los términos con $p_i(\text{w})=0$ se toman como $0\log 0 := 0$.

Esta construcción es efectiva ya que cumple con las propiedades y características de un buen indicador de variabilidad, dado que:
1. **Continuidad en $p_i$​**. Pequeñas perturbaciones de la distribución $p(\text{w})$ producen variaciones pequeñas en $\mathcal{H}(\text{w})$, lo que garantiza que cambios graduales en la mezcla de regímenes se reflejen de forma suave en el indicador de variabilidad.
2. **Monotonía con el número de celdas efectivamente ocupadas.** Si todas las probabilidades son iguales $p_i(\text{w})=1/N_{reg}$, se obtiene la entropía máxima
$$\mathcal{H}_{\max}(N_{reg}) = \log N_{reg}$$
	de modo que aumentar el número de celdas relevantes (más regímenes equiprobables) incrementa necesariamente la entropía.
3. **Descomposición en subconjuntos.** Si se agrupa una celda $\mathbb{C}_a$​ en subceldas más finas con distribución interna $q_j$​, la entropía total puede escribirse como
$$\mathcal{H}(r) = \mathcal{H}(p) + p_a\,\mathcal{H}(q)$$
	lo que permite descomponer la variabilidad global en contribuciones de regímenes principales y variabilidad interna dentro de ellos.

Estas propiedades hacen que la entropía de Shannon sea una excelente propuesta para cuantificar la variabilidad de contexto en el marco propuesto.

Luego, para comparar ventanas con diferentes números de celdas $N_{reg}$ (o para hacer el indicador adimensional y acotado), se introduce una versión normalizada:
$$V(\text{w}) \;=\; \frac{\mathcal{H}(\text{w})}{\log N_{reg}} \;\in\; [0,1]$$
Con esta normalización:
- $V(\text{w})=0$ indica que toda la masa está concentrada en un único tipo de contexto (régimen prácticamente único).
- $V(\text{w})\approx 1$ indica que el lazo reparte su tiempo de forma aproximadamente uniforme entre los $N_{reg}$ regímenes definidos.

El escalar $V(\text{w})$ es, por tanto, una mejor medida de variabilidad del contexto en la ventana de decisión $\text{w}$.

### 4. Agregación por episodio y por política
Al igual que las métricas de desempeño $J_{\text{fast}},J_{\text{smooth}},J_{\text{safety}}$​ se agregan desde intervalos a episodios y desde episodios a medidas de largo plazo, la variabilidad de contexto se integra en dos niveles.

Sea un episodio de evaluación $E$ compuesto por una secuencia de ventanas $W_{E,1},\dots,W_{E,\#W_E}$​​. Para cada una se dispone de $V(W_{E,\text{w}})$. Se define la variabilidad media de contexto del episodio como:
$$V_E = \frac{1}{\#W_E}\sum_{\text{w}=1}^{\#W_E} V\big(W_{E,\text{w}}\big)$$
Este escalar resume cuánto cambia el régimen de operación a lo largo de un episodio típico bajo una política dada. Episodios con trayectorias muy confinadas (por ejemplo, cerca de un punto de equilibrio con pequeñas perturbaciones) tendrán $V_E$ cercano a 0, mientras que episodios con grandes transitorios, saturaciones intermitentes y cambios de setpoint mostrarán $V_E$​ más altos.

A nivel de política, dado un conjunto de evaluación $\mathcal{E}=\{1,\dots,N_{\text{eval}}\}$ y una política estacionaria $\pi$, se define la variabilidad esperada de contexto de la política como:
$$\bar{V}(\pi) \;=\; \mathbb{E}_{\nu,\;P(\cdot|\pi)}\big[V_E\big] \;\approx\; \hat{\bar{V}}(\pi) \;=\; \frac{1}{N_{\text{eval}}}\sum_{E=1}^{N_{\text{eval}}} V_E$$
donde la esperanza se toma respecto de la distribución de condiciones de evaluación $\nu$ y de la dinámica inducida por $\pi$. Este valor resume, con un único escalar, qué tan “rico” o “concentrado” es el conjunto de contextos que recorre la política durante la evaluación estándar.

### 5. Medidas de variabilidad específicas dentro del sistema
Se desarrolla un conjunto de 4 medidas que convierten el proceso completo de operación en un objeto contextual cuantificable. Donde, cada ventana y lazo se puede caracterizar, ya sea por: 
1. Regimen operativo de control
	- Start-Up -> SU
	- Transient Regime-> TR
	- Steady Regime -> SR
2. Grado de no estacionariedad relativa (respecto a entropía del intervalo anterior)
3. Estado del proceso de aprendizaje (entropía de acciones, tasa de cambio, subceldas de bajo riesgo)
4. Detectabilidad marginal de las decisiones
5. Nivel de ruido. 

Por lo tanto, para cada lazo $\ell_{\theta} \in \{\text{lazo de ángulo}, \text{lazo de posición del carro}, \dots\}$, se considera en cada instante $t$ un vector de baja observabilidad:
$$\mathbb{x}^{(\ell_{\theta})}_t = \big( e^{(\ell_{\theta})}_t,\, \dot{e}^{(\ell_{\theta})}_t,\,  e_{I,t}^{(\ell_{\theta})},\, k^{(\ell_{\theta})}_{p,t},\, k^{(\ell_{\theta})}_{i,t},\, k^{(\ell_{\theta})}_{d,t},\, u^{(\ell_{\theta})}_t,\, \Delta u^{(\ell_{\theta})}_t,\, \text{sat}^{(\ell_{\theta})}_t \big)$$
A partir de $\mathbb{x}^{(\ell_{\theta})}_t$​ se define el vector de características de contexto
$$\mathbb{z}^{(\ell_{\theta})}_t = \chi\big(\mathbb{x}^{(\ell_{\theta})}_t\big)$$
y se reutiliza la partición discreta en celdas $\{\mathbb{C}^{(\ell_{\theta})}_i\}_{i=1}^{N_{\text{reg}}}$ y la distribución empírica de contexto por ventana descrita previamente. Todo lo que sigue se construye sobre estas mismas ventanas w\text{w}w y celdas de contexto.

### 5.1. Regímenes operativos de control: Start-up, Transient y Steady
Para caracterizar los regímenes operativos se introduce, para cada lazo $\ell_{\theta}$, una variable de régimen instantáneo:
$$\Xi^{(\ell_{\theta})}_t \in \{\text{SU},\ \text{TR},\ \text{SR}\}$$
correspondiente a Start-up, Transient Regime y Steady Regime, respectivamente.

Se parte de variables normalizadas, obtenidas a partir de límites operativos conocidos:
$$\tilde{e}^{(\ell_{\theta})}_t = \frac{|e^{(\ell_{\theta})}_t|}{e_{\max}^{(\ell_{\theta})}},\quad \widetilde{\dot{e}}^{(\ell_{\theta})}_t = \frac{|\dot{e}^{(\ell_{\theta})}_t|}{\dot{e}_{\max}^{(\ell_{\theta})}},\quad \widetilde{\Delta u}^{(\ell_{\theta})}_t = \frac{|\Delta u^{(\ell_{\theta})}_t|}{\Delta u_{\max}^{(\ell_{\theta})}}$$
Se eligen umbrales
$$\theta_e^{\text{low}} < \theta_e^{\text{high}}$$
$$\theta_{\dot{e}}^{\text{low}} < \theta_{\dot{e}}^{\text{high}}$$
$$\theta_{\Delta u}^{\text{low}} < \theta_{\Delta u}^{\text{high}}$$ 
en $(0,1)$, y se define:
- **Start-up (SU)**: régimen con errores y variaciones claramente altos, típico del arranque o de cambios abruptos de consigna:
$$\Xi^{(\ell_{\theta})}_t = \text{SU} \quad\text{si}\quad \tilde{e}^{(\ell_{\theta})}_t \ge \theta_e^{\text{high}} \ \text{y}\ \big( \widetilde{\dot{e}}^{(\ell_{\theta})}_t \ge \theta_{\dot{e}}^{\text{high}} \ \text{o}\ \widetilde{\Delta u}^{(\ell_{\theta})}_t \ge \theta_{\Delta u}^{\text{high}} \big)$$
- **Steady State (SR)**: régimen con error, variación de error y variación de acción bajos y sin saturación:
$$\Xi^{(\ell_{\theta})}_t = \text{SR} \quad\text{si}\quad \tilde{e}^{(\ell_{\theta})}_t \le \theta_e^{\text{low}}, \ \widetilde{\dot{e}}^{(\ell_{\theta})}_t \le \theta_{\dot{e}}^{\text{low}}, \ \widetilde{\Delta u}^{(\ell_{\theta})}_t \le \theta_{\Delta u}^{\text{low}}, \ \text{sat}^{(\ell_{\theta})}_t = 0$$
- **Transient (TR)**: cualquier situación intermedia:
$$\Xi^{(\ell_{\theta})}_t = \text{TR} \quad\text{en otro caso}$$
Sobre una ventana específica $\text{w} \in W \, | \, W=(t^{-},t]$ se define la distribución empírica de regímenes para el lazo $\ell_{\theta}$:
$$\pi^{(\ell_{\theta})}_\mathbb{r}(\text{w}) = \frac{1}{N_{\text{w}}} \sum_{\tau=t^{-}+1}^{t} \mathbf{1}\big[ \Xi^{(\ell_{\theta})}_\tau = \mathbb{r} \big], \quad \mathbb{r}\in\{\text{SU},\text{TR},\text{SR}\}$$
con $N_{\text{w}} = t - t^{-}$. Esta distribución describe qué fracción de la ventana el lazo opera en cada régimen.

De forma análoga a la entropía de contexto, se puede definir una entropía de régimen:
$$\mathcal{H}^{(\ell_{\theta})}_\Xi(\text{w}) = -\sum_{\mathbb{r}} \pi^{(\ell_{\theta})}_\mathbb{r}(\text{w})\log \pi^{(\ell_{\theta})}_\mathbb{r}(\text{w})$$
que mide cuán mezclado está el episodio entre SU, TR y SR dentro de la ventana. En combinación con $V(\text{w})$, la pareja $\big(V(\text{w}),\mathcal{H}^{(\ell_{\theta})}_\Xi(\text{w})\big)$ caracteriza simultáneamente variabilidad de contexto y mezcla de regímenes.

### 5.2. No estacionariedad relativa por lazo
Dado que la idea es que el lazo no conozca información alguna de otros lazos, la no estacionariedad se define como una medida relativa, que mide los cambios en la distribución de contextos de un lazo $\ell_{\theta}$ a lo largo del tiempo. Para cada lazo se dispone, en cada ventana $\text{w}$, de la distribución empírica de contexto
$$p^{(\ell_{\theta})}_i(\text{w}) = \frac{n^{(\ell_{\theta})}_i(\text{w})}{N_{\text{w}}}, \quad i=1,\dots,N_{\text{reg}}$$
definida solo a partir de las muestras $\mathbb{z}^{(\ell)}_t$​ de ese lazo.

Para cuantificar la no estacionariedad local se comparan distribuciones en ventanas consecutivas:
$$D^{(\ell_{\theta})}_{\text{NS}}(\text{w}) = D\big(p^{(\ell_{\theta})}(\text{w})\;\|\;p^{(\ell_{\theta})}(\text{w}-1)\big)$$
donde $D(\cdot\|\cdot)$ puede ser una distancia o divergencia elegida (por ejemplo, divergencia KL, distancia total de variación o una versión suavizada para evitar inestabilidades cuando hay celdas sin visitas). Conceptualmente:
- Valores bajos de $D^{(\ell_{\theta})}_{\text{NS}}(\text{w})$ indican que la mezcla de contextos para el lazo $\ell_{\theta}$ es similar a la de la ventana previa, coherente con un régimen casi estacionario.
- Valores altos indican que el lazo ha cambiado significativamente la región del espacio de contextos que recorre (por ejemplo, cambio de consigna, perturbación fuerte, cambio de régimen SU↔TR↔SR).

Si se quiere hacer explícito el rol de los regímenes, se puede refinar la medida definiendo distribuciones condicionadas:
$$p^{(\ell_{\theta})}_{i|\mathbb{r}}(\text{w}) = \frac{ \sum_{\tau} \mathbf{1}\big[\mathbb{z}^{(\ell_{\theta})}_\tau \in \mathbb{C}^{(\ell_{\theta})}_i,\ \Xi^{(\ell_{\theta})}_\tau = \mathbb{r}\big] }{ \sum_{\tau} \mathbf{1}\big[\Xi^{(\ell_{\theta})}_\tau = \mathbb{r}\big] }$$
y comparar, por ejemplo, $p^{(\ell_{\theta})}_{\cdot|\mathbb{r}}(\text{w})$ con $p^{(\ell_{\theta})}_{\cdot|\mathbb{r}}(\text{w}-1)$ solo dentro de cada régimen. Esto permite identificar si la no estacionariedad proviene de cambios de mezcla de regímenes (pasar de TR a SR) o de modificaciones internas del comportamiento dentro de un mismo régimen.

### 5.3. Adquisición y utilización de conocimiento efectivo
El proceso de aprendizaje se caracteriza desde la perspectiva de cómo varían las decisiones de los agentes (subir, mantener o bajar cada ganancia) en los distintos regímenes y contextos, y cómo esas decisiones tienden a converger hacia un comportamiento estacionario de explotación, manteniendo zonas de exploración de bajo riesgo.

Para cada agente $g \in \{k_p,k_i,k_d\}$ asociado al lazo $\ell_{\theta}$, se considera la acción discreta:
$$a^{(g,\ell_{\theta})}_t \in \{\uparrow,\ \circ,\ \downarrow\}$$
correspondiente a subir, mantener o bajar una ganancia en un paso de decisión.

Se definen tres tipos de descriptores:

#### (a) Distribución de acciones por contexto y régimen
Para cada celda de contexto $\mathbb{C}^{(\ell_{\theta})}_i$ y régimen $\mathbb{r}$, se estima la distribución empírica de acciones del agente $g$:
$$\pi^{(g,\ell_{\theta})}_{a|i,\mathbb{r}} = \frac{ \sum_t \mathbf{1}\big[\mathbb{z}^{(\ell_{\theta})}_t \in \mathbb{C}^{(\ell_{\theta})}_i,\ \Xi^{(\ell_{\theta})}_t=\mathbb{r},\ a^{(g,\ell_{\theta})}_t = a\big] }{ \sum_t \mathbf{1}\big[\mathbb{z}^{(\ell_{\theta})}_t \in \mathbb{C}^{(\ell_{\theta})}_i,\ \Xi^{(\ell_{\theta})}_t=\mathbb{r}\big] }$$
La entropía de acción local
$$\mathcal{H}^{(g,\ell_{\theta})}_{A}(i,\mathbb{r}) = -\sum_{a} \pi^{(g,\ell_{\theta})}_{a|i,\mathbb{r}}\log \pi^{(g,\ell_{\theta})}_{a|i,\mathbb{r}}$$
mide, en cada celda y régimen, cuán “mezcladas” están las decisiones del agente:
- Entropía alta sugiere exploración activa (acciones variadas sin una preferencia clara).
- Entropía baja indica que el agente ha convergido a un patrón de decisión estable (explotación).

La evolución de $\mathcal{H}^{(g,\ell_{\theta})}_{A}(i,\mathbb{r})$ a lo largo de episodios permite diagnosticar el proceso de adquisición de conocimiento en cada zona del espacio de contexto.

#### (b) Tasa de cambio de decisión como indicador de convergencia
Dentro de una ventana $\text{w}$, se puede definir la tasa de cambio de acción del agente $g$ en el lazo $\ell_{\theta}$:
$$\Gamma^{(g,\ell_{\theta})}_{\mathbb{r}}(\text{w}) = \frac{1}{N_{\text{w}}-1} \sum_{\tau=t^{-}+1}^{t-1} \mathbf{1}\big[a^{(g,\ell_{\theta})}_{\tau+1} \neq a^{(g,\ell_{\theta})}_{\tau}\big]$$
Valores decrecientes de $\Gamma^{(g,\ell_{\theta})}(\text{w})$ indican que las decisiones del agente se estabilizan en cada régimen, lo que es coherente con una convergencia progresiva hacia una política más estacionaria.

#### (c) Subceldas de exploración de bajo riesgo
Para identificar regiones del contexto donde la exploración tiene bajo impacto adverso, se define un índice de riesgo local por celda y régimen:
$$\rho^{(\ell_{\theta})}(i,\mathbb{r}) = \mathbb{E}\big[ \alpha_e\,\tilde{e}^{(\ell_{\theta})}_t + \alpha_{\Delta u}\,\widetilde{\Delta u}^{(\ell_{\theta})}_t + \alpha_{\text{sat}}\,\text{sat}^{(\ell_{\theta})}_t \,\big|\, \mathbb{z}^{(\ell_{\theta})}_t\in \mathbb{C}^{(\ell_{\theta})}_i,\ \Xi^{(\ell_{\theta})}_t=\mathbb{r} \big]$$

donde $\alpha_e,\alpha_{\Delta u},\alpha_{\text{sat}}\ge 0$ son pesos fijados a nivel de diseño (no como objetivos multiobjetivo o combinación lineal de la lagrangiana, sino como parámetros de diagnóstico). Celdas y regímenes con $\rho^{(\ell_{\theta})}(i,\mathbb{r})$ pequeño se interpretan como subceldas de exploración de bajo riesgo, adecuadas para mantener cierta entropía de acción $\mathcal{H}^{(g,\ell_{\theta})}_A(i,\mathbb{r})$ alta sin comprometer estabilidad, mientras que celdas con riesgo alto deberían asociarse con patrones de decisión más conservadores.

### 5.4. Detectabilidad marginal de la decisión
La detectabilidad marginal mide cuánto se diferencia, a nivel de métricas observables, el efecto de elegir subir, mantener o bajar una ganancia en un contexto y régimen dados. Si el impacto de la decisión es muy pequeño comparado con la variabilidad del entorno, el agente tendrá dificultades para atribuir crédito.

Para cada lazo $\ell_{\theta}$, celda $i$, régimen $\mathbb{r}$ y agente $g$, se considera un vector de métricas intensivas de lazo calculadas en la ventana posterior a la decisión (por ejemplo, las utilizadas en la Lagrangiana: $L_e,L_{\dot e},L_I,L_u$ etc.). Se denota:
$$\mathbf{Y}^{(\ell_{\theta})} = (L_e, L_{\dot e}, L_I, L_u, \dots)$$
Condicionando por acción, se estima la media y covarianza locales:
$$\boldsymbol{\mu}^{(g,\ell)}_{i,r,a} = \mathbb{E}\big[\mathbf{Y}^{(\ell)} \mid \mathbb{z}^{(\ell)}_t\in\mathbb{C}^{(\ell)}_i,\ R^{(\ell)}_t=r,\ a^{(g,\ell)}_t = a\big]$$
$$\Sigma^{(g,\ell)}_{i,r,a} = \text{Cov}\big[\mathbf{Y}^{(\ell)} \mid \mathbb{z}^{(\ell)}_t\in\mathbb{C}^{(\ell)}_i,\ R^{(\ell)}_t=r,\ a^{(g,\ell)}_t = a\big]$$
La detectabilidad marginal entre dos acciones $a$ y $a'$ en $(i,r)$ se puede medir, por ejemplo, mediante una distancia de Mahalanobis de medias:
$$D^{(g,\ell)}_{i,r}(a,a') = \big(\boldsymbol{\mu}^{(g,\ell)}_{i,r,a} - \boldsymbol{\mu}^{(g,\ell)}_{i,r,a'}\big)^\top \Sigma^{-1}_{i,r}\, \big(\boldsymbol{\mu}^{(g,\ell)}_{i,r,a} - \boldsymbol{\mu}^{(g,\ell)}_{i,r,a'}\big)$$
donde $\Sigma_{i,r}$​ es una covarianza promedio (por ejemplo, la media de $\Sigma^{(g,\ell)}_{i,r,a}$ sobre las acciones con suficientes muestras).

Un índice agregado de detectabilidad local se define como:
$$\mathcal{D}^{(g,\ell)}_{i,r} = \frac{1}{|\mathcal{A}|(|\mathcal{A}|-1)} \sum_{a\neq a'} D^{(g,\ell)}_{i,r}(a,a'), \quad \mathcal{A}=\{\uparrow,\circ,\downarrow\}$$
Interpretación:
- $\mathcal{D}^{(g,\ell)}_{i,r}$ grande indica que cambiar la acción del agente produce efectos claramente diferenciables en las métricas de lazo en ese contexto y régimen; la decisión es fácilmente detectable para el proceso de aprendizaje.
- $\mathcal{D}^{(g,\ell)}_{i,r}$ pequeña indica que las acciones son prácticamente indistinguibles a nivel de métricas, lo que dificulta la asignación de crédito.

Al analizar $\mathcal{D}^{(g,\ell)}_{i,r}$​ entre regímenes, se puede observar, por ejemplo, que en Transient las decisiones sobre $k_p$​ tienen alta detectabilidad sobre $L_e$y $L_{\dot e}$​, mientras que en Steady State su impacto es menor y la detectabilidad se desplaza hacia métricas ligadas a suavidad o esfuerzo de control.


### 5.5. Ruido en el entorno de aprendizaje

El ruido en el entorno de aprendizaje se refleja en la variabilidad residual de las métricas observables que no se explica por el contexto y la acción elegidos. Se distingue entre:

- Ruido **estructural** (perturbaciones del entorno, cambios de consigna imprevistos).
    
- Ruido **inducido por saturaciones** (acciones limitadas repetidamente en el borde).
    
- Ruido **intrínseco** (fluctuaciones pequeñas en las respuestas del sistema).
    

Para cada lazo ℓ\ellℓ, celda iii, régimen rrr y acción aaa del agente ggg, se puede cuantificar el ruido como la traza de la covarianza:

Ni,r,a(g,ℓ)=tr(Σi,r,a(g,ℓ)),\mathcal{N}^{(g,\ell)}_{i,r,a} = \text{tr}\big(\Sigma^{(g,\ell)}_{i,r,a}\big),Ni,r,a(g,ℓ)​=tr(Σi,r,a(g,ℓ)​),

donde Σi,r,a(g,ℓ)\Sigma^{(g,\ell)}_{i,r,a}Σi,r,a(g,ℓ)​ es la covarianza de Y(ℓ)\mathbf{Y}^{(\ell)}Y(ℓ) definida más arriba. Valores altos de Ni,r,a(g,ℓ)\mathcal{N}^{(g,\ell)}_{i,r,a}Ni,r,a(g,ℓ)​ señalan que, incluso fijando contexto, régimen y acción, las métricas de lazo presentan alta variabilidad, lo que reduce la señal de aprendizaje.

A nivel de ventana w\text{w}w, se puede definir un **índice de ruido de lazo** más grueso:

N(ℓ)(w)=βe Var(et(ℓ)∣t∈w)+βΔu Var(Δut(ℓ)∣t∈w)+βsat 1Nw∑t∈wsatt(ℓ),\mathcal{N}^{(\ell)}(\text{w}) = \beta_e\,\text{Var}\big(e^{(\ell)}_t \mid t\in\text{w}\big) + \beta_{\Delta u}\,\text{Var}\big(\Delta u^{(\ell)}_t \mid t\in\text{w}\big) + \beta_{\text{sat}}\, \frac{1}{N_{\text{w}}}\sum_{t\in\text{w}}\text{sat}^{(\ell)}_t,N(ℓ)(w)=βe​Var(et(ℓ)​∣t∈w)+βΔu​Var(Δut(ℓ)​∣t∈w)+βsat​Nw​1​t∈w∑​satt(ℓ)​,

con pesos βe,βΔu,βsat≥0\beta_e,\beta_{\Delta u},\beta_{\text{sat}}\ge 0βe​,βΔu​,βsat​≥0. Este indicador resume, en cada ventana, cuánto “ruido operativo” ve el lazo en términos de dispersión de error, variaciones de acción y frecuencia de saturaciones.

Analizado junto con la no estacionariedad DNS(ℓ)(w)D^{(\ell)}_{\text{NS}}(\text{w})DNS(ℓ)​(w), la distribución de regímenes πr(ℓ)(w)\pi^{(\ell)}_r(\text{w})πr(ℓ)​(w) y la detectabilidad Di,r(g,ℓ)\mathcal{D}^{(g,\ell)}_{i,r}Di,r(g,ℓ)​, este índice permite distinguir si el sistema está aprendiendo en un entorno:

- **estable y poco ruidoso**, favorable para explotación y ajuste fino de ganancias;
    
- **variable pero con ruido moderado**, adecuado para exploración controlada;
    
- **altamente ruidoso o saturado**, donde las ganancias de exploración adicional son bajas y la robustez del controlador resulta prioritaria.