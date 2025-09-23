from interfaces.controller import Controller # Importar Interfaz
from typing import Dict, Any
import logging
import numpy as np # Para NaN

# 5.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class PIDController(Controller): # Implementar Interfaz Controller
    """
    Implementación de un controlador PID estándar.
    Mantiene estado interno (errores) y ganancias configurables.
    """
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float, dt: float):
        """
        Inicializa el controlador PID.

        Args:
            kp: Ganancia Proporcional inicial.
            ki: Ganancia Integral inicial.
            kd: Ganancia Derivativa inicial.
            setpoint: Valor objetivo para la variable controlada.
            dt: Paso de tiempo usado para cálculos de integral y derivada (debe coincidir con simulación).
        Raises:
            ValueError: Si dt no es positivo o ganancias/setpoint no son numéricos.
            TypeError: Si los tipos son incorrectos.
        """
        logger.info(f"Inicializando PIDController: Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint}, dt={dt}")
        # 5.2: Validar parámetros (Fail-Fast)
        try:
            kp_f, ki_f, kd_f, sp_f, dt_f = map(float, [kp, ki, kd, setpoint, dt])
        except (ValueError, TypeError) as e:
            raise TypeError(f"Kp, Ki, Kd, Setpoint y dt deben ser numéricos: {e}") from e

        if dt_f <= 0:
            raise ValueError(f"dt proporcionado a PIDController ({dt_f}) debe ser positivo.")

        # Guardar valores iniciales para reset completo
        self.initial_kp, self.initial_ki, self.initial_kd = kp_f, ki_f, kd_f
        self.setpoint = sp_f
        self._dt = dt_f # Guardar dt interno
        # Ganancias actuales (inicializadas con valores validados)
        self.kp, self.ki, self.kd = kp_f, ki_f, kd_f
        self.prev_kp = kp_f
        self.prev_ki = ki_f
        self.prev_kd = kd_f
        # Estado interno del controlador - inicializar en reset_internal_state
        self.prev_error: float = 0.0
        self.integral_error: float = 0.0
        self.prev_integral_error = 0.0
        self.derivative_error: float = 0.0
        
        # 5.3: Asegurar inicialización limpia de estado interno by reset_internal_state
        logger.debug("PIDController inicializado. Llamando a reset_internal_state inicial.")
        self.reset_internal_state() # Asegurar estado limpio al inicio


    def compute_action(self, state: Any) -> float:
        """
        Calcula la acción de control PID basada en el estado actual.
        Asume que state[2] es la variable a controlar (ángulo del péndulo).

        Args:
            state: Vector de estado del sistema [cart_pos, cart_vel, angle, angular_vel].

        Returns:
            float: Acción de control calculada (fuerza). Devuelve 0.0 si el cálculo falla.
        """
        # 5.4: Validar entrada 'state'
        if not isinstance(state, (np.ndarray, list)) or len(state) < 3:
            logger.warning(f"PID compute_action: Estado inválido o incompleto: {state}. Devolviendo 0.0.")
            return 0.0
        try:
            current_measurement = float(state[2]) # Ángulo
            if not np.isfinite(current_measurement):
                 logger.warning(f"PID compute_action: Medición inválida (NaN/inf): {current_measurement}. Devolviendo 0.0.")
                 return 0.0
            error = current_measurement - self.setpoint

            # --- Término Integral ---
            # Normal -> I(t) = Sumatoria(error(t))
            #self.integral_error += error * self._dt
            # Anti-WindUp -> I(t) = (Kp(t-1)*error(t-1) + Ki(t-1)*I(t-1)*dt - Kp(t)*error(t)) / Ki(t)
            self.integral_error += ((self.prev_kp * self.prev_error) + (self.prev_ki * self.prev_integral_error * self._dt) - (self.kp * error))/self.ki if self.ki!=0 else 0

            # --- Término Derivativo ---
            # D(t) = (error(t) - error(t-1)) / dt
            # _dt ya validado como > 0 en init
            derivative = (error - self.prev_error) / self._dt
            self.derivative_error = derivative # Guardar para loggeo

            # --- Cálculo de la Acción de Control ---
            # u(t) = Kp*error(t) + Ki*Integral(t) + Kd*Derivada(t)
            proportional_term = self.kp * error
            integral_term = self.ki * self.integral_error
            derivative_term = self.kd * derivative

            control_action = proportional_term + integral_term + derivative_term

            logger.debug(f"Controller -> compute_action() -> dt={self._dt}, Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
            logger.debug(f"Controller -> compute_action() -> prev_err={self.prev_error:.8f}, prev_I_error={self.prev_integral_error:.8f}")
            logger.debug(f"Controller -> compute_action() -> err={error:.8f}, I_error={self.integral_error:.8f}, D_error={self.derivative_error:.8f}")
            logger.debug(f"Controller -> compute_action() -> PID Compute: P={proportional_term:.8f}, I={integral_term:.8f}, D={derivative_term:.8f} -> u={control_action:.8f}")

            # --- Actualizar estado interno para el siguiente paso ---
            self.prev_error = error
            self.prev_integral_error = self.integral_error

            # Devolver como float estándar
            return float(control_action) if np.isfinite(control_action) else 0.0 # Devolver 0 si es NaN/inf

        except IndexError as e:
             logger.error(f"PIDController: Error de índice en compute_action: {e}. Estado: {state}")
             return 0.0
        except Exception as e:
            logger.error(f"PIDController: Error inesperado en compute_action: {e}", exc_info=True)
            return 0.0


    def update_params(self, kp: float, ki: float, kd: float):
        """Actualiza las ganancias del controlador."""
        # 5.5: Guardar ganancias ANTERIORES antes de actualizar
        self.prev_kp = self.kp
        self.prev_ki = self.ki
        self.prev_kd = self.kd

        # Validar y actualizar nuevas ganancias
        self.kp = float(kp) if np.isfinite(kp) else self.prev_kp # Revertir si es inválido
        self.ki = float(ki) if np.isfinite(ki) else self.prev_ki
        self.kd = float(kd) if np.isfinite(kd) else self.prev_kd
        # Loguear si se revirtió alguna ganancia?
        if not np.isfinite(kp) or not np.isfinite(ki) or not np.isfinite(kd):
             logger.warning(f"update_params recibió NaN/inf. Kp={kp}, Ki={ki}, Kd={kd}. Se revirtieron las inválidas.")
        #logger.debug(f"PID Gains Updated: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f} (Prev: Kp={self.prev_kp:.2f}, Ki={self.prev_ki:.2f}, Kd={self.prev_kd:.2f})")

    def get_params(self) -> Dict[str, float]:
        """Devuelve las ganancias actuales."""
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}

    def reset(self):
        """Resetea las ganancias a sus valores iniciales y limpia el estado interno."""
        logger.debug("PIDController: Resetting gains to initial values and clearing internal state.")
        # Restaurar ganancias iniciales
        self.kp, self.ki, self.kd = self.initial_kp, self.initial_ki, self.initial_kd
        # Resetear estado interno también
        self.reset_internal_state()

    def reset_internal_state(self):
        """Resetea solo el estado interno (errores, integral) y prev_gains"""
        # logger.debug("PIDController: Resetting internal state (errors, integral, prev_gains).")
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_integral_error = 0.0
        self.derivative_error = 0.0
        # 5.6: Resetear prev_gains a las ganancias *actuales* después del reset (o iniciales)
        self.prev_kp = self.kp
        self.prev_ki = self.ki
        self.prev_kd = self.kd
        