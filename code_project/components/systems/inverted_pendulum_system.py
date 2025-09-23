import numpy as np
from scipy.integrate import odeint # type: ignore[import-untyped] # odeint puede no tener stubs
from interfaces.dynamic_system import DynamicSystem # Importar Interfaz
from typing import Any, List, Dict
import logging

# 12.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class InvertedPendulumSystem(DynamicSystem): # Implementar Interfaz DynamicSystem
    """
    Modelo dinámico del péndulo invertido. Implementa DynamicSystem.
    Utiliza scipy.integrate.odeint.
    """
    def __init__(self, m1: float, m2: float, l: float, g: float, cr: float = 0.0, ca: float = 0.0):
        """
        Inicializa el sistema del péndulo invertido.

        Args:
            m1: Masa del carro (kg).
            m2: Masa de la lenteja (kg).
            l: Longitud de la barra (m).
            g: Aceleración gravedad (m/s^2).
            cr: Coef. fricción carro (opcional, default=0).
            ca: Coef. fricción pivote (opcional, default=0).

        Raises:
            ValueError: Si m1, m2, l, g no son positivos o params no son numéricos.
        """
        logger.info(f"Inicializando InvertedPendulumSystem: m1={m1}, m2={m2}, l={l}, g={g}, cr={cr}, ca={ca}")
        # 12.2: Validar parámetros (Fail-Fast)
        try:
            params_list = [float(p) for p in [m1, m2, l, g, cr, ca]]
            m1_f, m2_f, l_f, g_f, cr_f, ca_f = params_list
        except (ValueError, TypeError) as e:
             raise ValueError(f"Parámetros físicos deben ser numéricos: {e}") from e

        if not (m1_f > 0 and m2_f > 0 and l_f > 0 and g_f > 0):
            raise ValueError("Masas (m1, m2), longitud (l) y gravedad (g) deben ser positivas.")
        if cr_f < 0 or ca_f < 0:
            logger.warning(f"Coeficientes de amortiguamiento cr({cr_f}) o ca({ca_f}) son negativos.")

        self.params: Dict[str, float] = {'m1': m1_f, 'm2': m2_f, 'l': l_f, 'g': g_f, 'cr': cr_f, 'ca': ca_f}
        logger.info("InvertedPendulumSystem inicializado.")

    def _dynamics(self, t: float, x: np.ndarray, u: float) -> List[float]:
        """Ecuaciones diferenciales del sistema. Llamada por odeint."""
        m1, m2, l, g, cr, ca = self.params.values()
        # Validar forma de x una sola vez al inicio
        if x.shape != (4,):
             # Error crítico si la forma es incorrecta
             raise ValueError(f"_dynamics: Shape de estado inválido {x.shape}. Esperado (4,).")
        x1, x2, x3, x4 = x # Desempaquetar estado [x, x_dot, theta, theta_dot]

        # Precalcular seno y coseno
        cosx3 = np.cos(x3)
        sinx3 = np.sin(x3)

        # Calcular fuerza efectiva en el carro
        force_on_cart = u - cr * x2 # u es la fuerza externa, cr*x2 es fricción

        # Velocidad del carro (dx1/dt)
        dx1dt = x2
        # Calcular aceleración del carro (dx2/dt)
        denominator_cart = m1 + m2 * (sinx3**2) # Equivalente a m1 + m2 * (1 - cosx3**2)
        dx2dt = (force_on_cart + m2 * l * (x4**2) * sinx3 - m2 * g * sinx3 * cosx3) / denominator_cart

        # Velocidad del péndulo (dx3/dt)
        dx3dt = x4
        # Calcular aceleración angular del péndulo (dx4/dt)
        # La fórmula derivada puede variar ligeramente entre fuentes
        # Fuente -> Russ Tedrake Underactuated Robotics:
        #numerator_pend = (g * sinx3 - dx2dt * cosx3 - (ca / (m2 * l)) * x4) # Incluye fricción angular
        #dx4dt = numerator_pend / l # Seguro que l no es cero (validado en init)
        # Fuente -> Steve Brunton:
        dx4dt = (-force_on_cart * cosx3 + (m1 + m2) * g * sinx3 - m2 * l * x4**2 * sinx3 * cosx3 - ca * x4)/(l * (m1 + m2 * (1 - cosx3**2)))

        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def apply_action(self, state: Any, action: float, t: float, dt: float) -> np.ndarray:
        """Aplica acción y avanza la dinámica usando odeint."""
        # 12.4: Validar estado de entrada
        try:
            current_state = np.array(state, dtype=float).flatten()
            if current_state.shape != (4,):
                 raise ValueError(f"Shape de estado inválido: {current_state.shape}")
            if not np.all(np.isfinite(current_state)):
                 raise ValueError(f"Estado contiene NaN/inf: {current_state}")
            action_f = float(action)
            if not np.isfinite(action_f):
                 logger.warning(f"Acción inválida (NaN/inf): {action}. Usando 0.0.")
                 action_f = 0.0
        except (ValueError, TypeError) as e:
            logger.error(f"apply_action: Estado o acción inválida: {e}. Devolviendo estado actual.")
            # Devolver copia del estado de entrada si es inválido
            return np.array(state, dtype=float).flatten() if isinstance(state, (list, tuple, np.ndarray)) and len(state)==4 else np.zeros(4)


        try:
            # Integrar ODEs
            time_span = [t, t + dt]
            # Usar tfirst=True puede ser más estable para algunos sistemas
            next_state = odeint(self._dynamics, current_state, time_span, args=(action_f,), tfirst=True)[-1]

            # Normalizar ángulo theta a [-pi, pi]
            next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi

            # Validar estado resultante (Fail-Fast si odeint falló)
            if not np.all(np.isfinite(next_state)):
                 logger.error(f"Estado resultante de odeint contiene NaN/inf: {next_state}. Estado anterior: {current_state}, Acción: {action_f}")
                 # Devolver estado anterior como fallback, pero indica un problema grave.
                 return current_state
                 # O lanzar error: raise RuntimeError("Fallo de integración ODE: resultado no finito.")

            return next_state

        except ValueError as e: # Capturar error de _dynamics (e.g., shape)
            logger.error(f"Error de valor durante integración ODE (probablemente en _dynamics): {e}", exc_info=True)
            return current_state # Devolver estado anterior
        except Exception as e: # Otros errores de odeint
            logger.error(f"Error inesperado durante integración ODE: {e}", exc_info=True)
            return current_state # Devolver estado anterior


    def reset(self, initial_conditions: Any) -> np.ndarray:
        """Resetea el estado del sistema a las condiciones iniciales."""
        # 12.4: Validar condiciones iniciales (Fail-Fast)
        try:
            initial_state = np.array(initial_conditions, dtype=float).flatten()
            if initial_state.shape != (4,):
                raise ValueError(f"Condiciones iniciales shape incorrecto: {initial_state.shape}. Esperado (4,).")
            if not np.all(np.isfinite(initial_state)):
                raise ValueError(f"Condiciones iniciales contienen NaN/inf: {initial_state}")

            # Normalizar ángulo inicial a [-pi, pi]
            initial_state[2] = (initial_state[2] + np.pi) % (2 * np.pi) - np.pi
            #logger.debug(f"Sistema reseteado a: {np.round(initial_state, 4)}")
            return initial_state

        except (ValueError, TypeError) as e:
            logger.critical(f"Error crítico formateando condiciones iniciales en reset: {e}")
            # Relanzar como error crítico
            raise ValueError(f"Condiciones iniciales inválidas: {initial_conditions}") from e