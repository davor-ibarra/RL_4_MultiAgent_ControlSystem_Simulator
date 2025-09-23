# components/systems/inverted_pendulum_system.py
import numpy as np
from scipy.integrate import odeint # type: ignore[import-untyped]
from interfaces.dynamic_system import DynamicSystem
from typing import Any, List, Dict # List, Dict
import logging

logger = logging.getLogger(__name__)

class InvertedPendulumSystem(DynamicSystem):
    def __init__(self, 
                 mass_cart_kg: float,         # m1 -> mass_cart_kg
                 mass_pendulum_kg: float,     # m2 -> mass_pendulum_kg
                 l: float,                    # Longitud del péndulo (se mantiene 'l' por ser estándar en física)
                 g: float,                    # Gravedad (se mantiene 'g')
                 cart_friction_coef: float = 0.0, # cr -> cart_friction_coef
                 pivot_friction_coef: float = 0.0 # ca -> air_friction_coef
                ):
        # Parámetros vienen de config['environment']['system']['params']
        logger.info(f"[InvertedPendulumSystem] Initializing with mass_cart_kg={mass_cart_kg}, mass_pendulum_kg={mass_pendulum_kg}, l={l}, g={g}, cart_friction_coef={cart_friction_coef}, pivot_friction_coef={pivot_friction_coef}")

        # Asignación directa, asumiendo validación externa.
        # Usaremos nombres internos m1, m2, etc., para coincidir con las fórmulas.
        self.m1 = float(mass_cart_kg)
        self.m2 = float(mass_pendulum_kg)
        self.l_bar = float(l) # l es una función en Python, usar l_bar o similar
        self.g_accel = float(g) # g también
        self.cr_friction = float(cart_friction_coef)
        self.ca_friction = float(pivot_friction_coef)

        if not (self.m1 > 0 and self.m2 > 0 and self.l_bar > 0 and self.g_accel > 0):
            raise ValueError("Masas, longitud y gravedad deben ser positivas para InvertedPendulumSystem.")
        
        logger.info("[InvertedPendulumSystem] Initialization complete.")

    def _dynamics(self, state_vector_x: np.ndarray, time_t: float, control_input_u: float) -> List[float]: # Nombres más descriptivos 'time_t', 'state_vector_x', 'control_input_u'
        # Desempaquetar parámetros del sistema para las ecuaciones
        m1, m2, l, g, cr, ca = self.m1, self.m2, self.l_bar, self.g_accel, self.cr_friction, self.ca_friction
        
        # Desempaquetar vector de estado
        # x1: posición del carro (x)
        # x2: velocidad del carro (x_dot)
        # x3: ángulo del péndulo (theta, 0 es vertical hacia arriba)
        # x4: velocidad angular del péndulo (theta_dot)
        x1, x2, x3, x4 = state_vector_x[0], state_vector_x[1], state_vector_x[2], state_vector_x[3]

        # Precalcular seno y coseno del ángulo del péndulo
        sin_x3 = np.sin(x3)
        cos_x3 = np.cos(x3)

        # Fuerza neta aplicada al carro, considerando la fricción del carro
        net_force_on_cart = control_input_u - cr * x2
        
        # Ecuaciones de movimiento (derivadas del estado)
        # dx1/dt = x2 (definición de velocidad del carro)
        dx1_dt = x2
        
        # dx2/dt = aceleración del carro (x_ddot)
        
        # Denominador común para x_ddot y theta_ddot
        # Esta es una forma común del denominador: M_total_effective = m1 + m2 * sin_x3**2
        common_denominator_term_A = m1 + m2 * sin_x3**2
        
        # Si common_denominator_term_A es muy pequeño, odeint puede tener problemas.
        dx2_dt = (net_force_on_cart + m2 * l * (x4**2) * sin_x3 - m2 * g * sin_x3 * cos_x3) / common_denominator_term_A
        
        # dx3/dt = x4 (definición de velocidad angular del péndulo)
        dx3_dt = x4
        
        # dx4/dt = aceleración angular del péndulo (theta_ddot)
        numerator_for_dx4_dt = (
            -net_force_on_cart * cos_x3 + (m1 + m2) * g * sin_x3 - # Fuerzas en el Péndulo
            m2 * l * (x4**2) * sin_x3 * cos_x3 -  # Fuerzas centrífugas/coriolis
            ca * x4                               # Fricción del aire
        )
        
        dx4_dt = (numerator_for_dx4_dt) / (l * common_denominator_term_A)

        return [dx1_dt, dx2_dt, dx3_dt, dx4_dt]

    def apply_action(self, current_state_vec: Any, control_action_val: float, current_time: float, dt_val: float) -> np.ndarray:

        time_integration_points = [current_time, current_time + dt_val]

        current_state_np_arr = np.array(current_state_vec, dtype=float).flatten()

        # Llamada a odeint. Si _dynamics o los inputs son malos (ej. NaN), odeint puede fallar o devolver NaNs.
        next_state_integrated_arr = odeint(
            self._dynamics,                 # Función a integrar
            current_state_np_arr,           # Estado inicial y0
            time_integration_points,        # Puntos de tiempo t
            args=(control_action_val,)      # Argumentos extra para _dynamics (después de t)
        )[-1]                               # Tomar el último punto de tiempo (estado final del intervalo dt)

        # Normalizar el ángulo theta (índice 2) a [-pi, pi]
        next_state_integrated_arr[2] = (next_state_integrated_arr[2] + np.pi) % (2 * np.pi) - np.pi
        
        # logger.debug(f"State In: {np.round(current_state_np_arr,3)}, Action: {control_action_val:.3f} -> State Out: {np.round(next_state_integrated_arr,3)}")
        return next_state_integrated_arr

    def reset(self, initial_conditions_state: Any) -> np.ndarray: # 'initial_conditions_state'
        # Asumir que initial_conditions_state es una lista/array de 4 números.
        initial_state_arr = np.array(initial_conditions_state, dtype=float).flatten()
        # Normalizar ángulo inicial
        initial_state_arr[2] = (initial_state_arr[2] + np.pi) % (2 * np.pi) - np.pi
        # logger.debug(f"[InvertedPendulumSystem] Reset to: {np.round(initial_state_arr, 4)}")
        return initial_state_arr