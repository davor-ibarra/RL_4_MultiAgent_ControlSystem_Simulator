---
created: 20251202 01:12
update: 20251207-14:21
summary:
status:
link:
tags:
---
# R2 - E1 - 1 - Documentación del Sistema  - Recompensas
## Lagrangiana como Recompensa
Se diseña una función de recompensa basada en el principio variacional que permita la minimización de la perdida del intervalo al establecer sus términos como funciones de costo:
$$r_{t}^{g} = -\,\mathcal{L}_t^{(g)}$$
Por lo que, la lagrangiana se define como:
$$\mathcal{L}_t = \underbrace{\alpha\,\overline{e^2}_{t} + \beta\,\overline{\dot e^{\,2}}_{t} + \gamma\,\overline{I^{2}}_{t}}_{\text{término potencial (desempeño)}} \;+ \underbrace{\underbrace{\eta\,\overline{E^{\Delta u}}_{t}}_{\text{esfuerzo de control}} + \underbrace{\zeta\,J^{(g)}_{t}}_{\text{inercia al cambio}}}_{\text{término cinético (suavidad)}} +\underbrace{\psi\,\phi(\tilde{s}_{t})}_{\text{barrera de saturación (seguridad)}}$$
donde: 
- $\alpha,\beta,\gamma,\eta,\zeta,\kappa,\psi>0$ son pesos.
- $\overline{(\cdot)}_{t}$​ denota promedio o acumulado normalizado en $(t^{-},t]$.
- $\tilde{s}_{t}\in[0,1]$ es la fracción de saturación (fracción del tiempo saturado en la ventana) y $\phi(\tilde{s}_{t})=\frac{\tilde{s}_{t}}{1-\tilde{s}_{t}}$ es creciente y convexa para elevar costo cerca de $s\!\to\!1$. 
- $J^{(g)}_{t}$ modera el ajuste de la ganancia de los agentes, con un primer término que castiga mover y el segundo que penaliza oscilar, es decir, cambiar de subir a bajar y viceversa (con $\mu>0$) tal que: 
$$J^{(g)}_{t} \;=\; \mathbf{1}[a_t\neq \circ] \;+\; \mu\,\mathbf{1}[a_t \text{ cambia el signo respecto a } a_{t^{-}}]$$
### Recompensas Base
#### Energía Potencial (Desempeño)
##### 1) Magnitud del error (proporcional)
- **Descripción:** energía de seguimiento por **tamaño del error** en la ventana.
- **Señales:** 
	- $e_t=r_t-y_t$
	- $|e_t|$
- **Métricas en intervalo:**
	- $\overline{e^2}_t=\tfrac{1}{N_t}\sum_{\tau=t^{-}+1}^{t}\tfrac{e_\tau^2}{\sigma_e^2}$, 
	- $\langle|e|\rangle$, 
	- $q_{90}(|e|)$.
- **Efecto:** mejora precisión de seguimiento.
- **Categoría:** global (base).
- **Objetivos:** seguimiento, estabilidad.
- **Temporalidad:** intervalo (intensiva).
- **Justificación:** mide desvío observable con baja información; robusto con normalización.
- **Controversias:** demasiado peso puede aumentar energía u overshoot; sensible a ruido si no se filtra.
- **Formas:**
    - **static:** $r_t=-c\cdot \tfrac{e_t^2}{\sigma_e^2}$.
    - **decay (exp/linear):** $r_t=-c\cdot \tfrac{e_t^2}{\sigma_e^2}\cdot \gamma^{k}$ si hubo $k$ pasos previos con mejora $\Delta|e|<0$.
    - **incremental (tanh/linear):** $r_t=-c\cdot \tanh(\alpha\,\overline{e^2}_t)$ o $-c\cdot \overline{e^2}_t$.
    - **conditional:** suavizar si $|sat_{err,t}|>0$; endurecer en **zona fina** $|e_t|\le \text{banda}$.

##### 2) Variación del error (derivativa)
- **Descripción:** energía por **cambio del error** (pendiente/oscilación).
- **Señales:** $\dot e_t=e_t-e_{t-1}$
- **Métricas en intervalo:**
	- $\overline{\dot e^{\,2}}_t=\tfrac{1}{N_t}\sum_{\tau}\tfrac{(e_\tau-e_{\tau-1})^2}{\sigma_{\dot e}^2}$,
	- $\langle|\dot e|\rangle$,
	- n° de picos $\#\{|\dot e|>\theta\}$.
- **Efecto:** aumenta suavidad y amortiguamiento.
- **Categoría:** global (base).
- **Objetivos:** estabilidad, suavidad de control.
- **Temporalidad:** intervalo (intensiva).
- **Justificación:** limita pendientes altas y oscilación sin requerir observables externos.
- **Controversias:** puede frenar rapidez necesaria en cambios bruscos; sensible a ruido de medición.
- **Formas:**
    - **static:** $r_t=-c\cdot \tfrac{(e_t-e_{t-1})^2}{\sigma_{\dot e}^2}$.
    - **decay (exp/linear):** $r_t=-c\cdot \tfrac{\dot e_t^{\,2}}{\sigma_{\dot e}^2}\cdot \gamma^{k}$ si $|\Delta u^{raw}|$ venía alto y cae.
    - **incremental (tanh/linear):** $r_t=-c\cdot \tanh(\alpha\,\overline{\dot e^{\,2}}_t)$ o $-c\cdot \overline{\dot e^{\,2}}_t$.
    - **conditional:** endurecer si $|\Delta u^{raw}_t|$ alto (evitar HF); relajar en grandes transitorios de setpoint.

##### 3) Error sostenido (integral)
- **Descripción:** energía por **persistencia del error** (sesgo en régimen).
- **Señales:** $I_t=\sum_{\tau=t^{-}+1}^{t} e_\tau$ (o filtrado), $|I_t|$.
- **Métricas en intervalo:** $\overline{I^{2}}_t=\tfrac{1}{N_t}\sum_{\tau}\tfrac{I_\tau^2}{\sigma_I^2}$, $\langle|I|\rangle$, racha sobre umbral.
- **Efecto:** reduce sesgo y mejora cierre exacto.
- **Categoría:** global (base).
- **Objetivos:** seguimiento en régimen, estabilidad.
- **Temporalidad:** intervalo (intensiva).
- **Justificación:** captura error remanente prolongado con observabilidad limitada.
- **Controversias:** incentiva subir $K_i$ y puede provocar windup si no se combina con barreras de saturación.
- **Formas:**
    - **static:** $r_t=-c\cdot \tfrac{I_t^2}{\sigma_I^2}$.
    - **decay (exp/linear):** $r_t=-c\cdot \tfrac{I_t^2}{\sigma_I^2}\cdot \gamma^{k}$ si $|I|$ viene descendiendo en racha.
    - **incremental (tanh/linear):** $r_t=-c\cdot \tanh(\alpha\,\overline{I^{2}}_t)$ o $-c\cdot \overline{I^{2}}_t$.
    - **conditional:** endurecer si $|sat_{err,t}|>0$ (evitar windup); relajar si $\Delta|e_t|<0$ sostenido.

#### Energía Cinética (Suavidad)
##### 4) Esfuerzo de control (cinético)
- **Descripción:** penalizar **magnitud y variación** de la acción para promover señales de control contenidas entre intervalos consecutivos.
- **Señales:** $u_{\text{total},t}$, $\Delta u_{\text{total},t}=u_{\text{total},t}-u_{\text{total},t-1}$.
- **Métricas en intervalo:**
- $\overline{E^{\Delta u}}_t=\tfrac{1}{N}\sum |\Delta u_{\text{total}}|$,
- RMS $\sqrt{\tfrac{1}{N}\sum (\Delta u_{\text{total}})^2}$,
- TV $\sum |\Delta u_{\text{total}}|$.
- **Efecto:** mayor suavidad y menor energía aplicada.
- **Categoría:** global (base).
- **Objetivos:** suavidad de control, energía, estabilidad.
- **Temporalidad:** intervalo (intensiva).
- **Justificación:** limitar la variación evita excitar no linealidades y reduce ruido en la recompensa.
- **Controversias:** penalizar en exceso puede ralentizar respuestas necesarias en grandes transitorios.
- **Formas:**
    - **static:** $r_t=-c\cdot \overline{E^{\Delta u}}_t$ (o $-c\cdot \text{RMS}(\Delta u)$).
    - **decay (exp/linear):** $r_t=-c\cdot \overline{E^{\Delta u}}_t\cdot \gamma^{k}$ si $\overline{E^{\Delta u}}$ viene bajando en racha.
    - **incremental (tanh/linear):** $r_t=-c\cdot \tanh(\alpha\,\overline{E^{\Delta u}}_t)$ o $-c\cdot \sum |\Delta u_{\text{total}}|$.
    - **conditional:** relajar si $|e_t|$ es alto o $|sat_{err,t}|>0$; endurecer en zona fina.

##### 5) Inercia al cambio de ganancias (cinético)
- **Descripción:** penalizar mover la ganancia y oscilar (cambiar de subir↔bajar) para estabilizar la sintonía.
- **Señales:**
	- acción $a_t\in\{\uparrow =+1,\circ =0,\downarrow =-1\}$,
	- indicadores $\mathbb{1}[a_t\neq \circ]$,$\mathbb{1}[\operatorname{sgn}(a_t)\neq \operatorname{sgn}(a_{t-1})]$, 
	- $\text{dwell}_t$.
- **Métricas en intervalo:** 
	- nº de movimientos $\sum \mathbb{1}[a_t\neq \circ]$,
	- nº de flips de signo,
	- $\text{dwell}_{\max}$,
	- $\overline{\text{dwell}}$.
- **Efecto:** reduce oscilaciones de sintonía y el serrucho paramétrico.
- **Categoría:** asignación (costo).
- **Objetivos:** estabilidad, suavidad de control, energía.
- **Temporalidad:** instantánea (evento) + agregado en intervalo.
- **Justificación:** imponer costo al movimiento y a la inversión de sentido disminuye el ruido paramétrico y mejora la convergencia.
- **Controversias:** umbrales altos pueden impedir ajustes legítimos ante perturbaciones.
- **Formas:**
    - **static:** $r_t=-c\cdot \mathbb{1}[a_t\neq \circ] - c'\cdot \mathbb{1}[\operatorname{sgn}(a_t)\neq \operatorname{sgn}(a_{t-1})]$.
    - **decay (exp/linear):** penalización creciente con racha de movimientos: $r_t=-c\cdot (1-e^{-\alpha\,n_{\text{move}}})$.
    - **incremental (tanh):** $r_t=-c\cdot \tanh(\beta\,n_{\text{flip}})$; descuento adicional si $\text{dwell}_t<\tau_{\min}$.
    - **conditional:** relajar si $|e_t|$ alto o aumentar si $|e_t|\le \text{banda}\ \wedge\ |sat_{err,t}|=0$ (zona fina).

### Recompensas Adicionales
#### Barrera de Saturación (Seguridad)
##### 6) Barrera convexa de saturación (seguridad)
- **Descripción:** penalizar fracción de saturación en la ventana con perfil convexo, elevando el costo cerca de $\tilde s_t \to 1$.
- **Señales:** $\tilde s_t \in [0,1]$ (fracción de tiempo con $|sat_{err}|>0$); $\phi(\tilde s_t)=\frac{\tilde s_t}{1-\tilde s_t}$.
- **Métricas en intervalo:** 
	- $\bar s=\tfrac{1}{N}\sum \tilde s_t$,
	- $\overline{\phi}=\tfrac{1}{N}\sum \phi(\tilde s_t)$,
	- $\max \tilde s$.
- **Efecto:** evita operar cerca del tope, reduciendo clipping sostenido.
- **Categoría:** asignación (costo).
- **Objetivos:** seguridad del actuador, estabilidad, energía.
- **Temporalidad:** intervalo (intensiva).
- **Justificación:** la convexidad de $\phi$ hace prohibitivos los periodos cercanos a saturación plena.
- **Controversias:** puede penalizar transitorios necesarios; requiere _gating_ para no frenar correcciones críticas.
- **Formas:**
    - **static:** $r_t=-c\cdot \phi(\tilde s_t)$.
    - **decay (type: exp/linear):** $r_t=-c\cdot \phi(\tilde s_t)\cdot \gamma^{k}$ si viene bajando la saturación ($k$ racha sin saturación).
    - **incremental (type: tanh):** $r_t=-c\cdot \tanh(\alpha\,\overline{\phi})$ para encapsular historial de ventana.
    - **conditional (type: tanh/sigmoid):** relajar si $|e_t|$ alto ($r_t\!\gets\! r_t\cdot \sigma(\beta(\epsilon_e-|e_t|))$); endurecer en zona fina.


## Diseño de Recompensa extendido
El diseño se compone de un **término potencial** que mide el desempeño en seguimiento al encapsular la dinámica observable que fenomenológicamente es sensible a los ajustes de $k_p$ (magnitud del error), $k_d$​ (variaciones del error) y $k_i$ (error sostenido). El término **cinético**, por un lado, denota la suavidad con el que el controlador opera, ya que, al penalizar el **esfuerzo de control**, la búsqueda de configuraciones de ganancias se alinean para minimizar las señales de control entre intervalos $W$ consecutivos. Y por otro lado, al regular la **inercia al cambio de ganancias** se impone un costo al movimiento que busca la convergencia hacia ganancias específicas y eliminar la oscilación de acciones repetitivas, lo que estabiliza el entrenamiento (menos ruido en la recompensa) y operación (menos excitación de no linealidades y saturaciones). Adicionalmente, la **barrera de saturación** impide que el aprendizaje busque reducción de error a costa de vivir cerca de los límites del actuador. La convexidad de la saturación hace que el costo marginal crezca de forma pronunciada conforme $s_t$​ se acerca a 1.

Destacar, que el marco propuesto tiene coherencia siempre y cuando que la sensibilidad marginal del retorno a cambios de cada ganancia sea detectable, para esto, es crítico que las variables de desempeño y esfuerzo de control sean contextualizadas al régimen en el que estas operan, así como a su vez, sean normalizadas dentro de un rango común y representativo para su operación, y así evitar que una de ellas domine por escala y distorsione la atribución de crédito dentro del intervalo. El objetivo es eliminar los efectos erróneos debido a escalas físicas heterogéneas, variaciones de longitud del intervalo o regímenes de operación, de modo que los pesos de la Lagrangiana del intervalo y la asignación de crédito funcionen sobre magnitudes con unidades y distribuciones comparables.

### Proceso de transformación de escalas
Sea $\mathbb{x}$ el vector de variables observables, para la transformación de la escala $x \in \mathbb{x}$ se pueden considerar de forma aislada o en simultaneo, una serie de enfoques alternativos que aseguran su comparación y procesamiento. 

Además, cada lazo se organiza en ventanas de decisión $W=(t_{i},t_{f}]\, |\, \text{w} \in W$ sobre las que se obtienen las métricas correspondientes, y sobre las que se pueden calcular los diferentes métodos que a continuación se describen.
#### Transformación instantánea
##### Normalización
El proceso de normalización estándar consiste en ajustar la escala de salida entre valores mínimos y máximos:
$$\hat{x}(t) = \frac{x(t)-x_{min}}{x_{max}-x_{min}}$$
Resultando en una respuesta escalada entre el rango $[0,1]$. Adicionalmente, en la práctica, es común procesar las respuestas de variables mediante sensores, por lo que, a su vez es posible activar la normalización mediante rangos de operación específicos, tal que:
$$\hat{x}(t)_{[-1,1]} = \frac{x(t)-x_{min}}{x_{max}-x_{min}}2-1$$
##### Intensificación Robusta
Para que las magnitudes sean intensivas, es decir, independientes de la duración del intervalo $N_t$, cada métrica del lazo se calcula únicamente con la información que ocurre dentro de la ventana de evaluación $\text{w}$, distribuyendo el valor por la cantidad de pasos dentro del intervalo. 
$$ \hat{x}_{int}{^2}(t) = \frac{1}{N_t}\sum_{t=t_{i}+\Delta t}^{t_{f}} \frac{x(t)^{2}}{\sigma_x^2} $$
Esta estandarización requiere fijar **escalas físicas de referencia** para cada métrica ($\sigma_e$, $\sigma_{\dot e}$, $\sigma_I$, $\sigma_{\Delta u}$) a fin de evitar que los términos de la pérdida dependan de la magnitud del setpoint o del régimen operativo.

#### Transformación al finalizar intervalo
##### Estandarización (Z-score)
Para algunos análisis o modelados, es útil que la escala de la métrica sea ubicada en una distribución con promedio $\mu_{\hat{x}}=0$ y varianza $\sigma_{\hat{x}}^{2}=1$, la que en este caso se obtiene como:
$$\hat{x}_{\text{z}}(t) = \frac{x(t)-\mu_{x}(\text{w})}{\sigma_{x}(\text{w})}$$
##### Winsor
A su vez, se debería poder activar el uso de cuantiles robustos para no contaminarse por valores atípicos:
$$ \hat{x}_{winsor}(t) = \max\{q_{(0.75)}(\hat{x}_{\bullet}(t)),\ x_{\text{ref}}\} $$
donde $x_{\text{ref}}$ es un piso numérico que garantiza estabilidad incluso si el error es pequeño durante largos intervalos.

##### Normalización con MAD
Para evitar que valores extremos produzcan saltos desproporcionados, se estandariza cada métrica mediante una auto-normalización, basada en su mediana y MAD sobre el intervalo $\text{w}$, que corresponde a una ventana deslizante de intervalos homogéneos del mismo lazo. La operación es:
$$ \hat{x}_{MAD}(t) = \frac{x(t) - Mediana(x(\text{w}))}{MAD(x(\text{w})) + \epsilon_{\text{std}}} $$
donde $MAD(x(\text{w}))$ es la desviación absoluta mediana en el horizonte, y $\epsilon_{\text{std}} > 0$ evita degeneración cuando la métrica es muy consistente y la dispersión tiende a cero. El resultado es una métrica centrada de dispersión normalizada.

#### Armonización entre bloques (Normalización)
Una vez que cada bloque está intensificado y estandarizado, todos los términos de la Lagrangiana del intervalo tienen una escala estadísticamente comparable, pero aún se requiere que puedan **ser ponderados correctamente** por los coeficientes de la función de pérdida ($\alpha, \beta, \gamma, \eta, \lambda$). Para garantizar que todos contribuyan de manera equilibrada, los valores se proyectan hacia una banda común mediante recorte por cuantiles seguido de un reescalado lineal.

El recorte por cuantiles elimina valores que no aportan información útil para atribuir crédito:
$$ z_{clip} = clip(z,\ q_{0.05},\ q_{0.95}) $$
manteniendo únicamente la región considerada informativa para aprendizaje y descartando colas estadísticas que no representan comportamiento estructural, sino ruido o eventos puntuales del actuador (saturación momentánea, disturbios transitorios, etc.).

Luego, las métricas recortadas se reescalan a una banda común:
$$ z_{norm} = \frac{z_{clip} - q_{0.05}}{q_{0.95} - q_{0.05} + \epsilon_{\text{norm}}} $$
El denominador asegura consistencia incluso cuando el rango entre cuantiles es pequeño. El resultado es un conjunto de términos intensivos, robustos y armónicos, donde todos tienen el mismo orden de magnitud, de modo que los pesos de la Lagrangiana actúan sobre magnitudes comparables y no sobre escalas físicas heterogéneas.



Con esto, la Lagrangiana del intervalo opera siempre sobre magnitudes comparables, independientemente de la duración del intervalo, del tamaño del salto en la acción de control o del régimen operativo.


### Asignación de crédito
La asignación de crédito se realiza a partir de las mismas métricas transformadas y descritas para el intervalo. Primero se fijan las magnitudes del lazo como términos canónicos del intervalo: 
$$L_e=\overline{e^2}_{t},\quad L_{\dot e}=\overline{\dot e^{\,2}}_{t},\quad L_I=\overline{I^{2}}_{t},\quad L_u=\overline{E^{\Delta u}}_{t},\quad L^{(g)}_{\text{mov}}=J^{(g)}_{t},\quad L_{\tilde{s}}=\phi(\tilde{s}_t)$$
Ahora, la pérdida total del lazo es $\ell_t=\alpha L_e+\beta L_{\dot e}+\gamma L_I+\eta L_u+ \zeta L^{(g)}_{\text{mov}} + \psi L_\tilde{s}$​ y se computa la recompensa del intervalo $r_t=-\ell_t$ una sola vez para el lazo, como antes. Entonces, para la distribución del crédito se derivan tres pérdidas específicas, que se expresan como una combinación lineal con pesos $w$ no negativos:
$$\mathcal{L}_t^{(p)}=\alpha\,w_{e}^{(p)}L_e+\beta\,w_{\dot e}^{(p)}L_{\dot e}+\gamma\,w_{I}^{(p)}L_I+\eta\,w_u^{(p)}L_u+\zeta\,L_{\text{mov}}^{(p)}+\psi\,w_{s}^{(p)}L_\tilde{s}$$$$\mathcal{L}_t^{(i)}=\alpha\,w_{e}^{(i)}L_e+\beta\,w_{\dot e}^{(i)}L_{\dot e}+\gamma\,w_{I}^{(i)}L_I+\eta\,w_u^{(i)}L_u+\zeta\,L_{\text{mov}}^{(i)}+\psi\,w_{s}^{(i)}L_\tilde{s}$$$$\mathcal{L}_t^{(d)}=\alpha\,w_{e}^{(d)}L_e+\beta\,w_{\dot e}^{(d)}L_{\dot e}+\gamma\,w_{I}^{(d)}L_I+\eta\,w_u^{(d)}L_u+\zeta\,L_{\text{mov}}^{(d)}+\psi\,w_{s}^{(d)}L_\tilde{s}$$
y que además, suman 1 por cada métrica correspondiente. Es decir, la partición de unidad de cada una de estas métricas deben cumplir que:
$$w_{e}^{(p)}\!+\!w_{e}^{(i)}\!+\!w_{e}^{(d)}=1,\ \ w_{\dot e}^{(p)}\!+\!w_{\dot e}^{(i)}\!+\!w_{\dot e}^{(d)}=1,\ \ \dots,\ \ w_{\tilde{s}}^{(p)}\!+\!w_{\tilde{s}}^{(i)}\!+\!w_{\tilde{s}}^{(d)}=1$$
Por construcción, se cumple $\mathcal{L}_t=\mathcal{L}_t^{(p)}+\mathcal{L}_t^{(i)}+\mathcal{L}_t^{(d)}$, entonces las recompensas específicas para cada uno de los agentes sería simplemente $r_{t}^{(g)}=-\mathcal{L}_t^{(g)}$

Esta descomposición mantiene la consistencia temporal: todas las cantidades se calculan al cierre del intervalo $(t^{-},t]$ y son intensivas respecto de $N_t$. La elección operativa de los pesos $w_\bullet^{(g)}$ es externa al agente y puede fijarse por régimen en coherencia con la sensibilidad canónica:  prioriza seguimiento ($w_e^{(p)}$ alto) y esfuerzo moderado;  prioriza deriva/offset ($w_I^{(i)}$ alto) y controla riesgo asociado a acumulación;  prioriza amortiguamiento de variaciones ( alto) con atención al esfuerzo por variación de control. El término de saturación se reparte con $w_{\tilde{s}}^{(g)}$ según el rol de cada ganancia frente al uso del canal, manteniendo $w_\tilde{s}^{(p)}+w_\tilde{s}^{(i)}+w_\tilde{s}^{(d)}=1$. En cambio, la inercia al cambio $L^{(g)}_{\text{mov}}$ permanece específica por agente, pues modela directamente la estabilidad de sus decisiones discretas en el espacio de bins; no se comparte ni se particiona.

*Elección operativa de pesos* $w$ (constantes por régimen) coherente con la sensibilidad canónica:
- Proporcional ($k_p$​): $w_{e}^{(p)}$​ alto (seguimiento), $w_{u}^{(p)}$​ moderado (esfuerzo por $\Delta u$), $w_{\dot e}^{(p)}$​ medio-bajo, $w_{I}^{(p)}$ bajo.
- Integral ($k_i$​): $w_{I}^{(i)}$​ alto (offset/deriva), $w_{s}^{(i)}$ medio (por riesgo de saturaciones asociadas a integral), resto bajos.
- Derivativo ($k_d$​): $w_{\dot e}^{(d)}$​ alto (amortiguamiento), $w_{u}^{(d)}$ medio (derivada tiende a rugosidad si se abusa), resto bajos.