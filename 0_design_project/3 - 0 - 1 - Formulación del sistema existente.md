# Marco general del sistema
## 1. Método Global 

Se plantea un esquema de _aprendizaje por refuerzo desacoplado_ con tres agentes independientes, uno por cada ganancia del PID ( $k_p, k_i, k_d$ ). Cada agente dispone de:

- **Estado** $s_t^g$ discretizado únicamente en su propia ganancia (y opcionalmente en otras variables del sistema), mediante un número finito de _bins_.
- **Acciones** $a_t^g \in ({0\ (\downarrow),1\ (\downarrow\\\uparrow),2\ (\uparrow))}$, que ajustan la ganancia en un paso fijo $\Delta k$ igual al tamaño de bin.
- **Decisiones** $\\d_{n}$ = $\Delta t_{dec}$ segundos, lo cual define intervalos uniformes de interacción.

En cada intervalo de decisión $[t_i,t_{n})$:

1. Se aplica la acción conjunta $a_i=(a_i^{p},a_i^{i},a_i^{d})$ al controlador, ajustando simultáneamente $(k_p,k_i,k_d)$.
2. Se acumula la recompensa **global** real
$$ R_{real}(i)  =  \sum_{n:t_i<n Δt≤t_{i+1}}r_{n}$$ ​donde $r_n=R_{\rm inst}(s_{n-1},u_{n-1},s_n)$ es la recompensa instantánea por paso $\Delta t_{n}$ del simulador.

3. Cada agente $g\in({p,i,d})$ observa su estado inicial $s_i^g$, ejecuta su acción $a_i^g$, recibe $R_{\rm learn}^g$ (según estrategias más abajo) y observa su nuevo estado $s_{i+1}^g$.

El **Q-learning off-policy** de cada agente se formula como:
$$Q_g(s_{i}^{g},a_{i}^{g})  ←  Q_g(s_{i}^{g},a_{i}^{g})  +  \alpha [R_{learn}^g+\gamma \max_{a^′} (Q_g(s_{i+1}^{g},a^g_{i+1}))−Q_g(s_{i}^{g},a_{i}^{g})]$$
con tasa de aprendizaje $\alpha$, factor de descuento $\gamma$, y la exploración $\varepsilon$-greedy para selección de acciones.

En el **método global**, se elige la estrategia de recompensa más simple:
$$ R_{learn}  =  R_{real}(i) \ ∀ \ g$$
Es decir, cada agente interpreta la misma señal global sin diferenciar su propia contribución.

---
## 2. Formulaciones de la recompensa instantánea

Los elementos básicos de recompensa $r_n$ en cada paso $\Delta t$ se basan en dos métodos alternativos:
### 2.1. Recompensa Gaussiana

Se asigna un valor a cada variable del estado y a la acción, modulando su magnitud mediante sus respectivas funciones Gaussianas centrada en cero y que luego son ponderadas mediante combinación lineal.

Es decir que, denotando el vector de estado en tiempo $n$ como:
$$x_n = \bigl[x_n^{\rm pos},\,x_n^{\rm vel},\,\theta_n,\,\dot\theta_n\bigr]$$
Y la acción de control $u_n$ , la función de recompensa instantánea se define:
$$r_n = \sum_{k \ \in\{\text{pos,vel},\theta,\dot\theta\}} p_k\exp\Bigl[-\bigl(\tfrac{x_n^k}{\hat{\sigma}_k}\bigr)^2\Bigr] \;+\; p_u\exp\Bigl[-\bigl(\tfrac{u_n}{\hat{\sigma}_u}\bigr)^2\Bigr]$$
donde ${w_i,\hat{\sigma}_i}$ son pesos y escalas que requieren ajuste.

**Ventajas**
- Castiga desvíos grandes de cero con forma suave.
- Parámetros ${w,\sigma}$ permiten priorizar variables.
**Desventajas**
- La señal combinada no discrimina cuál agente provocó el cambio.
- Requiere de ajuste apropiado en las escalas de normalización.
- Función objetivo dispersa y basada en estado del sistema mas no en el rendimiento de control.
**Mejoras posibles**
- Normalizar dinámicamente $\sigma_i$ según histograma de datos??
- Añadir término de entropía para fomentar exploración en regiones críticas??

### 2.2. Recompensa vía _stability_calculator_

Representación de la estabilidad instantánea de cada variable mediante indicador único $w_{inst}$
$$w_{inst}^k = \tfrac{x_n^k}{\hat{\sigma}_k}$$
Luego, dos variantes de cálculo:
#### 2.2.1. Simple Exponential

La estabilidad del sistema como:
$$w_{\rm stab}(x) = \exp\Bigl[-\sum_{k}\lambda_k(w_{inst}^k)^2\Bigr]$$
Y donde la recompensa sería:
$$r_n = w_{\rm stab}(x_n)$$
#### 2.2.2. IRA (Inestabilidad Relativa Acumulada)

Se estandariza cada variable por su media $\mu_i$ y desviación $\sigma_i$ :
$$Z_k​=\frac{x_k−\mu_k}{max(\sigma_{k}​,ε)}$$
Y se calcula la *"energía penalizada de desviación"* $IRA = \sum_{k}p_k\,Z_k^2$ en función al peso de cada variable $p_k$ , luego la estabilidad del sistema:
$$w_{\rm stab}(x) = \exp\Bigl[-\sum_{k}p_k\,Z_k^2\Bigr]$$
Y la recompensa se define como:
$$r_n = \exp\Bigl[-\lambda_{k}*IRA\Bigr]$$
Además, si está habilitado, se adaptan ${\mu_k\ ,\ \sigma_k}$ al final del episodio según min-máxima varianza observada.

**Ventajas**
- Intento de una noción más adaptativa del indicador de “estabilidad”.
- IRA permite adaptar la normalización a la dinámica real obtenida.
**Desventajas**
- Puede sobre-enfatizar estados muy cercanos al punto deseado.
- Cambio brusco entre estados muy lejanos y muy cercanos al punto deseado.
- Requiere de ajuste apropiado en las escalas de normalización.
- Función objetivo dispersa y basada en estado del sistema mas no en el rendimiento de control.
- Adaptatividad de IRA introduce ruido si no cuenta con suficientes datos.
**Mejoras posibles**
- Incorporar _momentum_ al actualizar $(\mu_i,\sigma_i)$ para suavizar cambios??
- Mezclar componentes gaussianos y estabilidad para un balance entre posición y estabilidad??

---

## 3. Estrategias de asignación de recompensa para Q-learning

Tras disponer de $R_{\rm real}^{(n)}$ o de los valores elementales $r_n$ , se deriva $R_{\rm learn}^g$ según la estrategia:
### 3.1. Global Reward Strategy

$$R_{learn}^g =R_{real}^{(n)} \quad\forall\ g$$
**Ventajas**
- Sencilla, baja complejidad.
**Desventajas**
- Severo problema de crédito, dificulta diferenciar impacto individual de cada ganancia.

### 3.2. Echo Baseline Reward Strategy

La idea es aislar la contribución de cada agente mediante simulaciones contrafactuales.

1. Se ejecuta simulación **real** del intervalo y se obtiene la recompensa acumulada $R_{\rm real}$.
2. Luego, para cada ganancia $g$:
    - Simular contrafactual manteniendo $k_g$ antiguo y aplicando las otras ganancias reales, obteniendo $R_{\rm cf}^g$.
    - Y se calcula:
$$R_{diff}^g \;=\;R_{real}-R_{cf}^g$$
3. Entonces para Q-learning de cada $g$:
$$R_{learn}^g \;=\;R_{diff}^g$$

**Ventajas**
- Mejor asignación de créditos: refuerza cada agente según su impacto diferencial.
- Enfoque en sensibilidad local de cada decisión
**Desventajas**
- Requiere 3 simulaciones extra por intervalo ($\sim 15.000$ *odeint* por episodio).
- Baja relación señal-ruido en $R_{diff}^g$ :
	- Los $R_{diff}^g$ suelen ser muy pequeños (fuerte cancelación numérica entre $R_{real}$​ y $R_{cf}^g$)
	- Sumado al exceso de ruido de simulación (etapa de exploración) y truncación (por discretización), de modo que $R_{diff}^g$ tiene alta varianza y su promedio converge muy despacio.
- La dinámica del PID no es separable: cambiar $k_p$ afecta la respuesta conjunta con $k_i\ y\ k_d$ . Por lo que, la aproximación de *“mantener* $g$ *y mudar los otros”* sería un **método de diferencias finitas parciales** poco fiable cuando los parámetros interactúan a mayores no linealidades.
**Mejoras**
- Hibridar con _Shadow_ para reciclar estimaciones de baseline?.

### 3.3. _Shadow Baseline Reward Strategy_

La idea es estimar un _baseline_ $B_g(s)$ del rendimiento promedio en cada estado, y usar la diferencia real-baseline como crédito.

- Definición del baseline en cada estado discreto $s$: tabla $B_g(s)$.
	- *Condición de aislamiento:* solo se actualiza $B_g(s)$ si $g$ **no cambió** y las otras dos ganancias sí cambiaron entre decisiones.
- Para cada intervalo:
    1. Calcular $R_{\rm learn}^g = R_{\rm real}-B_g(s_i^g)$.
    2. Si se cumple la condición de aislamiento, actualizar
$$B_g(s_{i}^g)\;\leftarrow\;B_g(s_i^g)\;+\;\beta\;w_{\rm stab}\;\bigl(R_{\rm real}-B_g(s_i^g)\bigr)$$
**Ventajas**
- Menor sobrecoste computacional que _Echo_.
- Introduce suavizado temporal de baseline que reduce varianza.
**Desventajas**
- Requiere buen muestreo de cada $(s,a)$ para aprender baselines precisos.
- Puede tardar muchos episodios en converger si $\beta$ muy pequeño, pero por el contrario si $\beta$ es muy grande se puede sobre-estimar el promedio de estados en los que una determinada ganancia $g$ se ha mantenido y las otras dos han cambiado.
**Mejoras**
- Ajustar $\beta$ de forma adaptativa según la varianza local de $R_{\rm real}$.
- Extender baseline a dependencias de acción: $B_g(s,a)$ para capturar interacción.
