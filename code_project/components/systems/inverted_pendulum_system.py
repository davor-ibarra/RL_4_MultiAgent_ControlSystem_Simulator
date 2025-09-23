import numpy as np
from scipy.integrate import odeint # type: ignore # odeint puede no tener stubs
from interfaces.dynamic_system import DynamicSystem
from typing import Any, List, Dict # Para type hints
import logging

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class InvertedPendulumSystem(DynamicSystem):
    """
    Implementación del modelo dinámico del péndulo invertido sobre un carro.
    Utiliza scipy.integrate.odeint para resolver las ecuaciones diferenciales.
    """
    def __init__(self, m1: float, m2: float, l: float, g: float, cr: float = 0.0, ca: float = 0.0):
        """
        Inicializa el sistema del péndulo invertido con sus parámetros físicos.

        Args:
            m1: Masa del carro (kg).
            m2: Masa de la lenteja del péndulo (kg).
            l: Longitud de la barra del péndulo (m).
            g: Aceleración debida a la gravedad (m/s^2).
            cr: Coeficiente de amortiguamiento/fricción del carro (opcional, default=0).
            ca: Coeficiente de amortiguamiento/fricción del pivote del péndulo (opcional, default=0).
        """
        logger.info(f"Inicializando InvertedPendulumSystem: m1={m1}, m2={m2}, l={l}, g={g}, cr={cr}, ca={ca}")
        # Validar parámetros
        if not all(isinstance(p, (int, float)) for p in [m1, m2, l, g, cr, ca]):
             raise ValueError("Todos los parámetros físicos deben ser numéricos.")
        if m1 <= 0 or m2 <= 0 or l <= 0 or g <= 0:
             raise ValueError("Las masas, longitud y gravedad deben ser positivas.")
        if cr < 0 or ca < 0:
             logger.warning(f"Coeficientes de amortiguamiento cr({cr}) o ca({ca}) son negativos. Se usarán como están.")
             # raise ValueError("Los coeficientes de amortiguamiento no pueden ser negativos.") # O lanzar error

        # Almacenar parámetros en un diccionario para dynamics
        self.params: Dict[str, float] = {'m1': m1, 'm2': m2, 'l': l, 'g': g, 'cr': cr, 'ca': ca}
        logger.info("InvertedPendulumSystem inicializado.")

    def _dynamics(self, x: np.ndarray, t: float, u: float) -> List[float]:
        """
        Define las ecuaciones diferenciales de primer orden del sistema.
        dx/dt = f(x, t, u)

        Args:
            x (np.ndarray): Vector de estado [x1, x2, x3, x4] = [pos_carro, vel_carro, angulo, vel_angular].
            t (float): Tiempo actual (no usado explícitamente en estas EDOs).
            u (float): Acción de control (fuerza aplicada al carro).

        Returns:
            List[float]: Derivadas del estado [dx1/dt, dx2/dt, dx3/dt, dx4/dt].
        """
        # Extraer parámetros
        m1, m2, l, g, cr, ca = self.params.values()

        # Asegurar que x es un array plano 1D
        x_flat = np.array(x).flatten()
        if x_flat.shape != (4,):
             # Esto no debería ocurrir si el estado se maneja correctamente
             logger.error(f"_dynamics recibió un estado con shape inesperado: {x_flat.shape}. Esperado: (4,).")
             # Devolver ceros o lanzar excepción? Devolver ceros puede ocultar errores.
             raise ValueError(f"Shape de estado inválido en _dynamics: {x_flat.shape}")

        x1, x2, x3, x4 = x_flat

        # Cálculos trigonométricos
        cosx3 = np.cos(x3)
        sinx3 = np.sin(x3)

        # Calcular fuerza neta sobre el carro, incluyendo amortiguamiento
        force_on_cart = u - cr * x2

        # Ecuaciones de movimiento (derivadas del estado)
        # Aceleración del carro (dx2/dt)
        denominator_cart = m1 + m2 * (1 - cosx3**2) # m1 + m2*sin(x3)^2
        if abs(denominator_cart) < 1e-9: # Evitar división por cero si sin(x3) es casi 0
             logger.warning(f"Denominador cercano a cero ({denominator_cart:.2e}) en cálculo de dx2dt. Ángulo={x3:.4f}rad")
             # ¿Qué hacer? Podría indicar estado inestable. Limitar la aceleración?
             # Si el ángulo es 0 o pi, el denominador es m1. Si m1 es 0, hay problema.
             if abs(m1) < 1e-9: raise ValueError("Masa del carro m1 no puede ser cero.")
             denominator_cart = m1 # Usar m1 si sin(x3)^2 es muy pequeño

        dx2dt = (force_on_cart + m2 * l * (x4**2) * sinx3 - m2 * g * sinx3 * cosx3) / denominator_cart

        # Aceleración angular (dx4/dt)
        denominator_pend = l * denominator_cart # l * (m1 + m2*sin(x3)^2)
        if abs(denominator_pend) < 1e-9:
             logger.warning(f"Denominador cercano a cero ({denominator_pend:.2e}) en cálculo de dx4dt. Ángulo={x3:.4f}rad, Longitud={l:.4f}")
             # Si l=0, hay problema. Si el otro denominador es cero, ya se manejó.
             if abs(l) < 1e-9: raise ValueError("Longitud del péndulo l no puede ser cero.")
             # Puede ser necesario devolver 0 o un valor grande si el estado es problemático
             dx4dt = 0.0 # O alguna heurística
        else:
             numerator_pend = (-force_on_cart * cosx3
                               + (m1 + m2) * g * sinx3
                               - m2 * l * (x4**2) * sinx3 * cosx3
                               - ca * x4) # Incluir amortiguamiento angular
             dx4dt = numerator_pend / denominator_pend

        # Derivadas de posición y ángulo son las velocidades
        dx1dt = x2
        dx3dt = x4

        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def apply_action(self, state: Any, action: float, t: float, dt: float) -> Any:
        """
        Aplica la acción y avanza la dinámica del sistema usando odeint.
        Normaliza el ángulo resultante a [-pi, pi].
        """
        # Asegurar que el estado de entrada es un array 1D de numpy
        current_state = np.array(state).flatten()
        if current_state.shape != (4,):
            logger.error(f"apply_action recibió estado con shape inesperado: {current_state.shape}. Esperado: (4,).")
            # Devolver estado actual puede ser lo más seguro
            return current_state

        try:
            # Resolver las EDOs para el intervalo [t, t + dt]
            # odeint devuelve una lista de estados en los tiempos solicitados
            # Queremos el estado en t + dt, que es el segundo elemento ([1])
            next_state = odeint(self._dynamics, current_state, [t, t + dt], args=(float(action),))[1]

            # Normalizar el ángulo (índice 2) al rango [-pi, pi]
            # (x + pi) % (2*pi) - pi mapea cualquier ángulo a este rango
            next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi

            return next_state

        except ValueError as e:
            # Capturar errores específicos de _dynamics (e.g., shape inválido)
            logger.error(f"Error de valor durante integración ODE en apply_action: {e}", exc_info=True)
            return current_state # Devolver estado anterior si la integración falla
        except Exception as e:
            logger.error(f"Error inesperado durante integración ODE (odeint) en apply_action: {e}", exc_info=True)
            # Devolver el estado actual parece lo más seguro
            return current_state


    def reset(self, initial_conditions: Any) -> Any:
        """Resetea el estado del sistema a las condiciones iniciales dadas."""
        # Asegurar que las condiciones iniciales son un array 1D de numpy del tamaño correcto
        try:
            initial_state = np.array(initial_conditions).flatten()
            if initial_state.shape != (4,):
                 raise ValueError(f"Condiciones iniciales tienen shape incorrecto: {initial_state.shape}. Esperado: (4,)")
            # Normalizar ángulo inicial también? Sí, buena práctica.
            initial_state[2] = (initial_state[2] + np.pi) % (2 * np.pi) - np.pi
            logger.debug(f"Sistema reseteado a: {np.round(initial_state, 4)}")
            return initial_state
        except Exception as e:
            logger.error(f"Error formateando condiciones iniciales en reset: {e}")
            # Devolver un estado por defecto o relanzar? Relanzar es mejor.
            raise ValueError(f"Condiciones iniciales inválidas: {initial_conditions}") from e