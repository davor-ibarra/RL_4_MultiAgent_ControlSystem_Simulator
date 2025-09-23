# components/systems/inverted_pendulum_system.py
import numpy as np
from scipy.integrate import odeint # type: ignore[import-untyped]
from interfaces.dynamic_system import DynamicSystem
from typing import Any, List, Dict
import logging

logger = logging.getLogger(__name__) # Logger específico del módulo

class InvertedPendulumSystem(DynamicSystem):
    def __init__(self, m1: float, m2: float, l: float, g: float, cr: float = 0.0, ca: float = 0.0):
        # Parámetros vienen de config['environment']['system']['params']
        logger.info(f"[InvertedPendulumSystem] Initializing with m1={m1}, m2={m2}, l={l}, g={g}, cr={cr}, ca={ca}")

        try:
            m1_f, m2_f, l_f, g_f, cr_f, ca_f = map(float, [m1, m2, l, g, cr, ca])
        except (ValueError, TypeError) as e:
            msg = f"Physical parameters must be numeric. Error: {e}"
            logger.critical(f"[InvertedPendulumSystem] {msg}"); raise TypeError(msg) from e

        if not (m1_f > 0 and np.isfinite(m1_f)): raise ValueError(f"m1 ({m1_f}) must be positive finite.")
        if not (m2_f > 0 and np.isfinite(m2_f)): raise ValueError(f"m2 ({m2_f}) must be positive finite.")
        if not (l_f > 0 and np.isfinite(l_f)): raise ValueError(f"l ({l_f}) must be positive finite.")
        if not (g_f > 0 and np.isfinite(g_f)): raise ValueError(f"g ({g_f}) must be positive finite.")
        if not np.isfinite(cr_f) or cr_f < 0: raise ValueError(f"cr ({cr_f}) must be non-negative finite.")
        if not np.isfinite(ca_f) or ca_f < 0: raise ValueError(f"ca ({ca_f}) must be non-negative finite.")
        if cr_f < 0: logger.warning(f"[InvertedPendulumSystem] Damping coefficient cr ({cr_f}) is negative.")
        if ca_f < 0: logger.warning(f"[InvertedPendulumSystem] Damping coefficient ca ({ca_f}) is negative.")

        self.params: Dict[str, float] = {'m1': m1_f, 'm2': m2_f, 'l': l_f, 'g': g_f, 'cr': cr_f, 'ca': ca_f}
        logger.info("[InvertedPendulumSystem] Initialization complete.")

    def _dynamics(self, t: float, x: np.ndarray, u: float) -> List[float]:
        # Esta función es llamada por odeint, no directamente por el usuario
        m1, m2, l, g, cr, ca = self.params.values()
        if x.shape != (4,): # Validar forma del estado pasado por odeint
            # Esto es un error de programación interno si ocurre.
            logger.error(f"[InvertedPendulumSystem:_dynamics] CRITICAL: Invalid state shape {x.shape} in ODE solver. Expected (4,).")
            # Devolver derivadas cero para evitar más problemas, pero esto es grave.
            return [0.0, 0.0, 0.0, 0.0]
        x1, x2, x3, x4 = x

        cosx3, sinx3 = np.cos(x3), np.sin(x3)
        force_on_cart = u - cr * x2
        dx1dt = x2
        denominator_cart = m1 + m2 * (sinx3**2)
        if np.isclose(denominator_cart, 0): # Evitar división por cero
            logger.warning(f"[InvertedPendulumSystem:_dynamics] Denominator for cart acceleration close to zero. State: {x}, u: {u}")
            # Podría devolver el estado anterior o una aceleración muy grande/NaN para indicar problema
            # Por ahora, devolvemos 0 para evitar NaN y que la simulación explote inmediatamente.
            dx2dt = 0.0 # O np.sign(numerador) * muy_grande
        else:
            dx2dt = (force_on_cart + m2 * l * (x4**2) * sinx3 - m2 * g * sinx3 * cosx3) / denominator_cart
        dx3dt = x4
        denominator_pend = l * (m1 + m2 * (1 - cosx3**2))
        if np.isclose(denominator_pend, 0):
            logger.warning(f"[InvertedPendulumSystem:_dynamics] Denominator for pendulum angular acceleration close to zero. State: {x}, u: {u}")
            dx4dt = 0.0
        else:
            dx4dt = (-force_on_cart * cosx3 + (m1 + m2) * g * sinx3 - m2 * l * x4**2 * sinx3 * cosx3 - ca * x4) / denominator_pend
        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def apply_action(self, state: Any, action: float, t: float, dt: float) -> np.ndarray:
        try:
            current_state = np.array(state, dtype=float).flatten()
            if current_state.shape != (4,): raise ValueError(f"Invalid state shape: {current_state.shape}.")
            if not np.all(np.isfinite(current_state)): raise ValueError(f"State contains NaN/inf: {current_state}.")
            action_f = float(action)
            if not np.isfinite(action_f):
                 logger.warning(f"[InvertedPendulumSystem:apply_action] Invalid action (NaN/inf): {action}. Using 0.0.")
                 action_f = 0.0
        except (ValueError, TypeError) as e:
            logger.error(f"[InvertedPendulumSystem:apply_action] Invalid input state or action: {e}. Returning previous state or zeros.")
            # Devolver una copia del estado original si es posible, o un estado de error.
            if isinstance(state, (list, tuple, np.ndarray)) and len(state) == 4 and np.all(np.isfinite(np.array(state))):
                return np.array(state, dtype=float).flatten()
            return np.zeros(4, dtype=float) # Fallback muy genérico

        time_span = [t, t + dt] # odeint espera el tiempo actual y el siguiente
        try:
            # odeint llamará a _dynamics(current_state, t, action_f) internamente para el primer paso,
            # y luego _dynamics(y_prev, t_curr, action_f) para los siguientes subpasos.
            # El 't' en _dynamics es el tiempo actual del integrador, no el 't' de inicio del intervalo.
            # tfirst=True hace que el primer argumento de _dynamics sea 't'.
            next_state_ode = odeint(self._dynamics, current_state, time_span, args=(action_f,), tfirst=True)
            next_state = next_state_ode[-1] # El último elemento es el estado en t+dt
            # Normalizar ángulo a [-pi, pi]
            next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
            if not np.all(np.isfinite(next_state)):
                 logger.error(f"[InvertedPendulumSystem:apply_action] odeint returned non-finite state: {next_state}. Prev state: {current_state}, Action: {action_f}. Reverting to prev state.")
                 return current_state # Revertir si la integración falla numéricamente
            return next_state
        except ValueError as ve_ode: # Errores de _dynamics (e.g., shape incorrecto)
            logger.error(f"[InvertedPendulumSystem:apply_action] ValueError during ODE integration (likely from _dynamics): {ve_ode}", exc_info=True)
            return current_state # Revertir a estado anterior
        except Exception as e_ode: # Otros errores de odeint
            logger.error(f"[InvertedPendulumSystem:apply_action] Unexpected error during ODE integration: {e_ode}", exc_info=True)
            return current_state # Revertir a estado anterior

    def reset(self, initial_conditions: Any) -> np.ndarray:
        try:
            initial_state = np.array(initial_conditions, dtype=float).flatten()
            if initial_state.shape != (4,):
                raise ValueError(f"Initial conditions shape incorrect: {initial_state.shape}. Expected (4,).")
            if not np.all(np.isfinite(initial_state)):
                raise ValueError(f"Initial conditions contain NaN/inf: {initial_state}")
            initial_state[2] = (initial_state[2] + np.pi) % (2 * np.pi) - np.pi # Normalizar ángulo
            #logger.debug(f"[InvertedPendulumSystem:reset] System reset to: {np.round(initial_state, 4)}")
            return initial_state
        except (ValueError, TypeError) as e:
            msg = f"Invalid initial conditions for reset: {initial_conditions}. Error: {e}"
            logger.critical(f"[InvertedPendulumSystem:reset] {msg}")
            raise ValueError(msg) from e