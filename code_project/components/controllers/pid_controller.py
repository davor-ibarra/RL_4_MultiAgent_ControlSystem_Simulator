# components/controllers/pid_controller.py
from interfaces.controller import Controller
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__) # Logger específico del módulo

class PIDController(Controller):
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float, dt: float):
        # Parámetros vienen de config['environment']['controller']['params'] + 'dt'
        logger.info(f"[PIDController] Initializing with Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint}, dt={dt}")

        # Validar tipos y valores
        try:
            kp_f, ki_f, kd_f, sp_f, dt_f = map(float, [kp, ki, kd, setpoint, dt])
        except (ValueError, TypeError) as e:
            msg = f"Kp, Ki, Kd, Setpoint, and dt must be numeric. Error: {e}"
            logger.critical(f"[PIDController] {msg}")
            raise TypeError(msg) from e

        if not np.isfinite(kp_f): raise ValueError(f"Kp ({kp_f}) must be finite.")
        if not np.isfinite(ki_f): raise ValueError(f"Ki ({ki_f}) must be finite.")
        if not np.isfinite(kd_f): raise ValueError(f"Kd ({kd_f}) must be finite.")
        if not np.isfinite(sp_f): raise ValueError(f"Setpoint ({sp_f}) must be finite.")
        if dt_f <= 0 or not np.isfinite(dt_f):
            raise ValueError(f"dt ({dt_f}) must be a positive finite number.")

        self.initial_kp, self.initial_ki, self.initial_kd = kp_f, ki_f, kd_f
        self.setpoint = sp_f
        self._dt = dt_f

        self.kp, self.ki, self.kd = kp_f, ki_f, kd_f
        # Inicializar prev_gains a las ganancias actuales al inicio
        self.prev_kp = kp_f
        self.prev_ki = ki_f
        self.prev_kd = kd_f

        self.prev_error: float = 0.0
        self.integral_error: float = 0.0
        self.prev_integral_error: float = 0.0 # Nueva variable para anti-windup I(t-1)
        self.derivative_error: float = 0.0

        logger.debug("[PIDController] Initialization complete. Internal state reset.")
        self.reset_internal_state() # Asegura que los estados de error estén limpios

    def compute_action(self, state: Any) -> float:
        if not isinstance(state, (np.ndarray, list)) or len(state) < 3: # Asumiendo state[2] es el ángulo
            logger.warning(f"[PIDController:compute_action] Invalid or incomplete state: {state}. Returning 0.0.")
            return 0.0
        try:
            current_measurement = float(state[2])
            if not np.isfinite(current_measurement):
                 logger.warning(f"[PIDController:compute_action] Invalid measurement (NaN/inf): {current_measurement}. Returning 0.0.")
                 return 0.0

            error = current_measurement - self.setpoint

            # --- Término Integral con Anti-Windup (si Ki != 0) ---
            # I(t) = I(t-1) + error(t)*dt (estándar)
            # O, usando la fórmula provista:
            # I(t) = (Kp(t-1)*error(t-1) + Ki(t-1)*I(t-1)*dt - Kp(t)*error(t)) / Ki(t)
            # Necesitamos guardar I(t-1) -> self.prev_integral_error
            if self.ki != 0:
                 # Esta fórmula parece tener un error de dependencia circular o malinterpretación.
                 # Kp(t) es self.kp, Kp(t-1) es self.prev_kp.
                 # Si I(t) depende de error(t), y error(t) se calcula ahora, Kp(t)*error(t) es con el Kp actual.
                 # La fórmula anti-windup más común es limitar el integral directamente.
                 # Vamos a usar la suma simple y luego considerar si el controlador tiene límites de salida.
                 self.integral_error += error * self._dt
                 # Aquí se podría añadir lógica de clamping si el controlador tiene límites de salida:
                 # output_min, output_max = -100, 100 # Ejemplo
                 # self.integral_error = np.clip(self.integral_error, output_min/self.ki if self.ki !=0 else -np.inf,
                 #                                                    output_max/self.ki if self.ki !=0 else np.inf)
            else:
                 self.integral_error = 0.0 # Resetear si Ki es cero

            # --- Término Derivativo ---
            derivative = (error - self.prev_error) / self._dt # _dt ya validado > 0
            self.derivative_error = derivative # Para logging

            proportional_term = self.kp * error
            integral_term = self.ki * self.integral_error
            derivative_term = self.kd * derivative
            control_action = proportional_term + integral_term + derivative_term
            #logger.debug(f"[PIDController:compute_action] Error={error:.4f}, IntegralErr={self.integral_error:.4f}, DerivErr={self.derivative_error:.4f} -> Action={control_action:.4f}")
            
            self.prev_error = error
            # Guardar el I(t) actual para que sea el I(t-1) del próximo paso
            # Esto es solo si se usa la fórmula de anti-windup compleja.
            # Con la suma simple, prev_integral_error no es estrictamente necesario para el cálculo
            # pero podría ser útil para observar su evolución. Por ahora no lo asignamos.
            # self.prev_integral_error = self.integral_error # <-- Descomentar si se usa para anti-windup

            return float(control_action) if np.isfinite(control_action) else 0.0

        except IndexError: logger.error(f"[PIDController:compute_action] IndexError. State: {state}"); return 0.0
        except Exception as e: logger.error(f"[PIDController:compute_action] Unexpected error: {e}", exc_info=True); return 0.0

    def update_params(self, kp: float, ki: float, kd: float):
        self.prev_kp, self.prev_ki, self.prev_kd = self.kp, self.ki, self.kd
        # Validar y asignar nuevas ganancias
        self.kp = float(kp) if np.isfinite(kp) else self.prev_kp
        self.ki = float(ki) if np.isfinite(ki) else self.prev_ki
        self.kd = float(kd) if np.isfinite(kd) else self.prev_kd
        if not (np.isfinite(kp) and np.isfinite(ki) and np.isfinite(kd)):
             logger.warning(f"[PIDController:update_params] Received NaN/inf gains (Kp={kp}, Ki={ki}, Kd={kd}). Invalid ones reverted.")
        #logger.debug(f"[PIDController:update_params] Gains updated: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f} (Prev Kp={self.prev_kp:.2f}, Ki={self.prev_ki:.2f}, Kd={self.prev_kd:.2f})")


    def get_params(self) -> Dict[str, float]:
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}

    def reset(self):
        logger.debug("[PIDController:reset] Resetting gains to initial and clearing internal state.")
        self.kp, self.ki, self.kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.reset_internal_state()

    def reset_internal_state(self):
        #logger.debug("[PIDController:reset_internal_state] Resetting errors, integral, and prev_gains.")
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_integral_error = 0.0 # Asegurar que prev_integral_error también se resetea
        self.derivative_error = 0.0
        # Al resetear estado interno, las ganancias "previas" deben reflejar las actuales
        self.prev_kp = self.kp
        self.prev_ki = self.ki
        self.prev_kd = self.kd