import numpy as np
from scipy.integrate import odeint # type: ignore # odeint puede no tener stubs
from interfaces.dynamic_system import DynamicSystem # Importar Interfaz
from typing import Any, List, Dict # Para type hints
import logging

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class InvertedPendulumSystem(DynamicSystem): # Implementar Interfaz DynamicSystem
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

        # Almacenar parámetros en un diccionario para dynamics
        self.params: Dict[str, float] = {'m1': m1, 'm2': m2, 'l': l, 'g': g, 'cr': cr, 'ca': ca}
        logger.info("InvertedPendulumSystem inicializado.")

    def _dynamics(self, x: np.ndarray, t: float, u: float) -> List[float]:
        """
        Define las ecuaciones diferenciales de primer orden del sistema.
        dx/dt = f(x, t, u) - Lógica interna sin cambios.
        """
        # ... (código sin cambios) ...
        m1, m2, l, g, cr, ca = self.params.values()
        x_flat = np.array(x).flatten()
        if x_flat.shape != (4,):
             logger.error(f"_dynamics recibió estado con shape inesperado: {x_flat.shape}.")
             raise ValueError(f"Shape de estado inválido en _dynamics: {x_flat.shape}")
        x1, x2, x3, x4 = x_flat
        cosx3 = np.cos(x3); sinx3 = np.sin(x3)
        force_on_cart = u - cr * x2
        denominator_cart = m1 + m2 * (1 - cosx3**2)
        if abs(denominator_cart) < 1e-9:
             if abs(m1) < 1e-9: raise ValueError("Masa del carro m1 no puede ser cero.")
             # logger.warning(f"Denom cart casi cero ({denominator_cart:.2e}). Ángulo={x3:.4f}. Usando m1.")
             denominator_cart = m1
        dx2dt = (force_on_cart + m2 * l * (x4**2) * sinx3 - m2 * g * sinx3 * cosx3) / denominator_cart
        denominator_pend = l * denominator_cart
        if abs(denominator_pend) < 1e-9:
             if abs(l) < 1e-9: raise ValueError("Longitud del péndulo l no puede ser cero.")
             logger.warning(f"Denom pend casi cero ({denominator_pend:.2e}). Ángulo={x3:.4f}, Long={l:.4f}. dx4dt=0.")
             dx4dt = 0.0
        else:
             numerator_pend = (-force_on_cart * cosx3 + (m1 + m2) * g * sinx3
                               - m2 * l * (x4**2) * sinx3 * cosx3 - ca * x4)
             dx4dt = numerator_pend / denominator_pend
        dx1dt = x2; dx3dt = x4
        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def apply_action(self, state: Any, action: float, t: float, dt: float) -> Any:
        """
        Aplica la acción y avanza la dinámica del sistema usando odeint.
        Normaliza el ángulo resultante a [-pi, pi]. Implementa método de interfaz.
        """
        # ... (código sin cambios) ...
        current_state = np.array(state).flatten()
        if current_state.shape != (4,):
            logger.error(f"apply_action recibió estado con shape inesperado: {current_state.shape}. Devolviendo estado actual.")
            return current_state
        try:
            next_state = odeint(self._dynamics, current_state, [t, t + dt], args=(float(action),))[1]
            # Normalizar ángulo [-pi, pi]
            next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
            # Asegurar que el estado no contenga NaN/inf
            if not np.all(np.isfinite(next_state)):
                 logger.error(f"Estado resultante contiene NaN/inf después de odeint: {next_state}. Reemplazando con estado anterior.")
                 return current_state # Devolver estado anterior si el resultado es inválido
            return next_state
        except ValueError as e:
            logger.error(f"Error de valor durante integración ODE: {e}", exc_info=True)
            return current_state
        except Exception as e:
            logger.error(f"Error inesperado durante integración ODE: {e}", exc_info=True)
            return current_state


    def reset(self, initial_conditions: Any) -> Any:
        """Resetea el estado del sistema. Implementa método de interfaz."""
        # ... (código sin cambios) ...
        try:
            initial_state = np.array(initial_conditions).flatten()
            if initial_state.shape != (4,):
                 raise ValueError(f"Condiciones iniciales shape incorrecto: {initial_state.shape}.")
            initial_state[2] = (initial_state[2] + np.pi) % (2 * np.pi) - np.pi
            if not np.all(np.isfinite(initial_state)):
                 raise ValueError(f"Condiciones iniciales contienen NaN/inf: {initial_state}")
            logger.debug(f"Sistema reseteado a: {np.round(initial_state, 4)}")
            return initial_state
        except Exception as e:
            logger.error(f"Error formateando condiciones iniciales en reset: {e}")
            raise ValueError(f"Condiciones iniciales inválidas: {initial_conditions}") from e