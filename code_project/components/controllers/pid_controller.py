from interfaces.controller import Controller # Importar Interfaz
from typing import Dict, Any
import logging
import numpy as np # Para NaN

# Obtener logger específico para este módulo
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
            setpoint: Valor objetivo para la variable controlada (e.g., ángulo 0).
            dt: Paso de tiempo usado para cálculos de integral y derivada.
                Este dt se usa internamente y debe ser el mismo que el de la simulación.
        """
        logger.info(f"Inicializando PIDController: Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint}, dt={dt}")
        # Validar parámetros
        if not all(isinstance(p, (int, float)) for p in [kp, ki, kd, setpoint, dt]):
             raise ValueError("Kp, Ki, Kd, Setpoint y dt deben ser numéricos.")
        if dt <= 0:
            raise ValueError(f"dt proporcionado a PIDController ({dt}) debe ser positivo.")

        # Guardar valores iniciales para reset completo
        self.initial_kp, self.initial_ki, self.initial_kd = kp, ki, kd
        self.setpoint = setpoint
        self._dt = dt # Guardar dt interno

        # Ganancias
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_kp = kp
        self.prev_ki = ki
        self.prev_kd = kd

        # Estado interno del controlador - inicializar en reset_internal_state
        self.prev_error: float = 0.0
        self.integral_error: float = 0.0
        self.prev_integral_error = 0.0
        self.derivative_error: float = 0.0
        # Asegurar inicialización limpia
        self.reset_internal_state()

        logger.info("PIDController inicializado.")


    def compute_action(self, state: Any) -> float:
        """
        Calcula la acción de control PID basada en el estado actual.
        Asume que state[2] es la variable a controlar (ángulo del péndulo).

        Args:
            state: Vector de estado del sistema [cart_pos, cart_vel, angle, angular_vel].

        Returns:
            float: Acción de control calculada (fuerza).
        """
        try:
            # Asegurar que state es indexable y tiene longitud suficiente
            if not isinstance(state, (np.ndarray, list)) or len(state) < 3:
                 raise IndexError(f"Estado inválido o incompleto para PIDController: {state}")

            current_measurement = state[2] # Ángulo
            error = current_measurement - self.setpoint

            # --- Término Integral ---
            #self.integral_error += error * self._dt
            self.integral_error += ((self.prev_kp * self.prev_error) + (self.prev_ki * self.prev_integral_error * self._dt) - (self.kp * error))/self.ki if self.ki!=0 else 0

            # --- Término Derivativo ---
            # D(t) = (error(t) - error(t-1)) / dt
            # Usar self._dt asegurando que no es cero (validado en init)
            derivative_error = (error - self.prev_error) / self._dt
            self.derivative_error = derivative_error # Guardar para posible loggeo

            # --- Cálculo de la Acción de Control ---
            # u(t) = Kp*error(t) + Ki*Integral(t) + Kd*Derivada(t)
            proportional_term = self.kp * error
            integral_term = self.ki * self.integral_error
            derivative_term = self.kd * derivative_error

            u = proportional_term + integral_term + derivative_term

            # --- Actualizar estado interno para el siguiente paso ---
            self.prev_kp = self.kp
            self.prev_ki = self.ki
            self.prev_kd = self.kd
            self.prev_error = error
            self.prev_integral_error = self.integral_error

            # logger.debug(f"PID Compute: Err={error:.3f}, P={proportional_term:.3f}, I={integral_term:.3f}, D={derivative_term:.3f} -> u={u:.3f}")
            # Devolver como float estándar
            return float(u) if np.isfinite(u) else 0.0 # Devolver 0 si es NaN/inf

        except IndexError as e:
             logger.error(f"PIDController: Error de índice en compute_action: {e}. Estado: {state}")
             return 0.0
        except Exception as e:
            logger.error(f"PIDController: Error inesperado en compute_action: {e}", exc_info=True)
            return 0.0


    def update_params(self, kp: float, ki: float, kd: float):
        """Actualiza las ganancias del controlador."""
        # logger.debug(f"PID Gains Update: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f} (Prev: Kp={self.kp:.2f}, Ki={self.ki:.2f}, Kd={self.kd:.2f})")
        # Validar que sean números finitos?
        self.kp = kp if np.isfinite(kp) else self.kp # Mantener anterior si es inválido
        self.ki = ki if np.isfinite(ki) else self.ki
        self.kd = kd if np.isfinite(kd) else self.kd


    def get_params(self) -> Dict[str, float]:
        """Devuelve las ganancias actuales."""
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}


    def reset(self):
        """Resetea las ganancias a sus valores iniciales y limpia el estado interno."""
        logger.debug("PIDController: Resetting gains to initial values and clearing internal state.")
        self.kp, self.ki, self.kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.reset_internal_state() # Llamar a reset interno para limpiar errores


    def reset_internal_state(self):
        """Resetea solo el estado interno (errores, integral) sin tocar las ganancias."""
        # logger.debug("PIDController: Resetting internal state (errors, integral).")
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_integral_error = 0.0
        self.derivative_error = 0.0
        self.prev_kp = self.initial_kp
        self.prev_ki = self.initial_ki
        self.prev_kd = self.initial_kd
        