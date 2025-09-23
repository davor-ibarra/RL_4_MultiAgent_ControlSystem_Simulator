from interfaces.controller import Controller
from typing import Dict, Any
import logging

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PIDController(Controller):
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
                Nota: Este dt es el inicial, el environment puede usar uno diferente
                      en la simulación real. El cálculo interno aquí podría necesitar
                      ser actualizado si dt cambia dinámicamente, o el dt real debería
                      pasarse a compute_action. Por ahora, asumimos dt constante aquí.
        """
        logger.info(f"Inicializando PIDController: Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint}, dt={dt}")
        # Guardar valores iniciales para reset completo
        self.initial_kp, self.initial_ki, self.initial_kd = kp, ki, kd
        self.setpoint = setpoint
        self._dt = dt # Guardar dt interno (puede o no ser usado dependiendo de la implementación)

        # Ganancias actuales (modificables por el agente)
        self.kp, self.ki, self.kd = kp, ki, kd

        # Estado interno del controlador
        self.prev_error: float = 0.0
        self.integral_error: float = 0.0
        self.derivative_error: float = 0.0 # Guardar para posible loggeo

        # Necesitamos estado adicional para cálculo correcto del integral discreto
        # si las ganancias cambian entre pasos (como en este caso)
        self.prev_ki = ki # Ki usado en el paso anterior
        self.prev_integral_error: float = 0.0 # Integral acumulado hasta el paso anterior

        # Validar dt inicial
        if not isinstance(dt, (float, int)) or dt <= 0:
            logger.warning(f"dt proporcionado a PIDController ({dt}) no es positivo. Usando 0.01 por defecto.")
            self._dt = 0.01


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
            # Calcular error actual (Setpoint - Medición) o (Medición - Setpoint)
            # Usaremos error = Medición - Setpoint (consistente con fórmula original)
            # Asumiendo que state[2] es el ángulo actual
            current_measurement = state[2]
            error = current_measurement - self.setpoint

            # --- Término Integral (con corrección por cambio de Ki) ---
            # Aproximación discreta: Integral(t) = Integral(t-dt) + error(t)*dt
            # Corrección si Ki cambia: I(t) = I(t-dt) * (Ki_prev / Ki_actual) + error(t)*dt
            # O alternativa (como en código original): recalcular I(t) para compensar cambio.
            # Fórmula original: integral_error += ((prev_kp * prev_error) + (prev_ki * prev_integral_error * dt) - (kp * error))/ki if ki!=0 else 0
            # Esta fórmula parece intentar compensar cambios en Kp Y Ki en el término integral, lo cual es complejo.
            # Usemos una aproximación más estándar: I(t) = I(t-dt) + error * dt
            # Si Ki es muy pequeño o cero, el término integral puede explotar.
            if abs(self.ki) > 1e-9: # Evitar división por cero o número muy pequeño
                 # Actualización simple del término integral
                 self.integral_error = self.prev_integral_error + error * self._dt
            else:
                 self.integral_error = 0.0 # Resetear si Ki es (casi) cero


            # --- Término Derivativo ---
            # D(t) = (error(t) - error(t-dt)) / dt
            if self._dt > 1e-9:
                 derivative_error = (error - self.prev_error) / self._dt
            else:
                 derivative_error = 0.0
            # Guardar para posible loggeo
            self.derivative_error = derivative_error

            # --- Cálculo de la Acción de Control ---
            # u(t) = Kp*error(t) + Ki*Integral(t) + Kd*Derivada(t)
            proportional_term = self.kp * error
            integral_term = self.ki * self.integral_error
            derivative_term = self.kd * derivative_error

            u = proportional_term + integral_term + derivative_term

            # --- Actualizar estado interno para el siguiente paso ---
            self.prev_error = error
            self.prev_integral_error = self.integral_error # Guardar integral *antes* de la nueva acumulación
            self.prev_ki = self.ki # Guardar Ki usado en este cálculo

            # logger.debug(f"PID Compute: Err={error:.3f}, P={proportional_term:.3f}, I={integral_term:.3f}, D={derivative_term:.3f} -> u={u:.3f}")
            return float(u)

        except IndexError:
             logger.error(f"PIDController: Estado recibido ({state}) no tiene índice 2 para calcular error.")
             return 0.0 # Devolver acción neutral en caso de error
        except Exception as e:
            logger.error(f"PIDController: Error inesperado en compute_action: {e}", exc_info=True)
            return 0.0

    def update_params(self, kp: float, ki: float, kd: float):
        """Actualiza las ganancias del controlador."""
        # logger.debug(f"PID Gains Update: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f} (Prev: Kp={self.kp:.2f}, Ki={self.ki:.2f}, Kd={self.kd:.2f})")
        # Validar que las ganancias no sean negativas? O permitirlo? Por ahora permitir.
        self.kp, self.ki, self.kd = kp, ki, kd

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
        self.derivative_error = 0.0
        # Resetear también Ki previo y integral previo
        self.prev_ki = self.ki # Asumir que el Ki actual es el "previo" para el siguiente ciclo tras reset
        self.prev_integral_error = 0.0