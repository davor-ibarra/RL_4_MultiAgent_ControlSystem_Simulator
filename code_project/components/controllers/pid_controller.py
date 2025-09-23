# components/controllers/pid_controller.py
from interfaces.controller import Controller
from typing import Dict, Any
import logging
import numpy as np
from typing import Dict, Any, Optional # Optional añadido

logger = logging.getLogger(__name__)

class PIDController(Controller):
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float, 
                 setpoint: float, 
                 dt_sec: float, 
                 anti_windup: Optional[Dict[str, Any]] = None
                 ):
        # Parámetros vienen de config['environment']['controller']['params'] + 'dt_sec'
        # anti_windup vendrá de config['environment']['controller']['params']['anti_windup']
        logger.info(f"[PIDController] Initializing with Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint}, dt_sec={dt_sec}, , AntiWindupConfig={anti_windup}")

        try:
            kp_float, ki_float, kd_float, setpoint_float, dt_sec_float = map(float, [kp, ki, kd, setpoint, dt_sec]) # '..._float'
        except (ValueError, TypeError) as e_map_float: # 'e_map_float'
            error_msg_numeric = f"Kp, Ki, Kd, Setpoint, and dt_sec must be numeric. Error: {e_map_float}" # 'error_msg_numeric'
            logger.critical(f"[PIDController] {error_msg_numeric}")
            raise TypeError(error_msg_numeric) from e_map_float

        if not np.isfinite(kp_float): raise ValueError(f"Kp ({kp_float}) must be finite.")
        if not np.isfinite(ki_float): raise ValueError(f"Ki ({ki_float}) must be finite.")
        if not np.isfinite(kd_float): raise ValueError(f"Kd ({kd_float}) must be finite.")
        if not np.isfinite(setpoint_float): raise ValueError(f"Setpoint ({setpoint_float}) must be finite.")
        if dt_sec_float <= 0 or not np.isfinite(dt_sec_float):
            raise ValueError(f"dt_sec ({dt_sec_float}) must be a positive finite number.")

        self.initial_kp_config, self.initial_ki_config, self.initial_kd_config = kp_float, ki_float, kd_float
        self.target_setpoint = setpoint_float
        self._time_step_duration = dt_sec_float

        self.current_kp, self.current_ki, self.current_kd = kp_float, ki_float, kd_float
        self.previous_kp, self.previous_ki, self.previous_kd = kp_float, ki_float, kd_float

        self.previous_error_value: float = 0.0
        self.accumulated_integral_error: float = 0.0
        self.previous_accumulated_integral_error: float = 0.0
        self.current_derivative_error: float = 0.0

        self.last_proportional_term: float = 0.0
        self.last_integral_term: float = 0.0
        self.last_derivative_term: float = 0.0

        # Anti-windup parameters
        self.aw_enabled = anti_windup.get('enabled', False)
        if self.aw_enabled:
            self.aw_use_back_calculation_formula = bool(anti_windup.get('use_back_calculation_formula', False))
            if self.aw_use_back_calculation_formula:
                logger.info("[PIDController] Anti-windup enabled using specific back-calculation formula for integral term.")
            else:
                logger.info("[PIDController] Anti-windup not enabled in configuration.")

        logger.debug("[PIDController] Initialization complete. Internal state reset.")
        self.reset_internal_state()

    def compute_action(self, current_system_state: Any) -> float: # 'current_system_state'
        try:
            measured_process_variable = float(current_system_state[2]) # 'measured_process_variable' (ángulo del péndulo)
            current_error_signal = measured_process_variable - self.target_setpoint # 'current_error_signal'

            # --- Proportonial Term ---
            self.last_proportional_term = self.current_kp * current_error_signal

            # --- Integral Term ---
            if self.current_ki != 0: # Acumular integral solo si Ki no es cero
                if self.aw_enabled and self.aw_use_back_calculation_formula: # aw_enabled ahora implica aw_use_back_calculation_formula
                    # I(t) = (Kp(t-1)*error(t-1) + Ki(t-1)*I(t-1)*dt - Kp(t)*error(t)) / Ki(t)
                    numerator = (self.previous_kp * self.previous_error_value) + \
                                (self.previous_ki * self.previous_accumulated_integral_error * self._time_step_duration) - \
                                (self.current_kp * current_error_signal)
                    self.accumulated_integral_error = numerator / self.current_ki
                    # logger.debug(f"AW Formula Integral: num={numerator}, ki={self.current_ki}, I_acc={self.accumulated_integral_error}")
                else:
                    # Acumulación estándar
                    self.accumulated_integral_error += current_error_signal * self._time_step_duration
            else:
                 self.accumulated_integral_error = 0.0 # Resetear integral si Ki es cero

            self.last_integral_term = self.current_ki * self.accumulated_integral_error

            # --- Derivative Term ---
            self.current_derivative_error = (current_error_signal - self.previous_error_value) / self._time_step_duration
            self.last_derivative_term = self.current_kd * self.current_derivative_error
            
            # --- Control Action PID ---
            final_control_action = self.last_proportional_term + self.last_integral_term + self.last_derivative_term
            
            # Save previous error for next time
            self.previous_error_value = current_error_signal
            self.previous_accumulated_integral_error = self.accumulated_integral_error

            return float(final_control_action)

        except IndexError: logger.error(f"[PIDController:compute_action] IndexError processing state: {current_system_state}"); return 0.0
        except Exception as e_compute: logger.error(f"[PIDController:compute_action] Unexpected error: {e_compute}", exc_info=True); return 0.0 # 'e_compute'

    def update_params(self, new_kp: float, new_ki: float, new_kd: float): # 'new_kp' etc.
        # Guardar ganancias actuales como "previas" antes de actualizar
        self.previous_kp, self.previous_ki, self.previous_kd = self.current_kp, self.current_ki, self.current_kd
        
        # Validar y asignar nuevas ganancias
        self.current_kp = float(new_kp) if np.isfinite(new_kp) else self.previous_kp
        self.current_ki = float(new_ki) if np.isfinite(new_ki) else self.previous_ki
        self.current_kd = float(new_kd) if np.isfinite(new_kd) else self.previous_kd
        
        if not (np.isfinite(new_kp) and np.isfinite(new_ki) and np.isfinite(new_kd)):
             logger.warning(f"[PIDController:update_params] Received NaN/inf gains (Kp={new_kp}, Ki={new_ki}, Kd={new_kd}). Invalid ones reverted to previous values.")
        #logger.debug(f"[PIDController:update_params] Gains updated: Kp={self.current_kp:.3f}, Ki={self.current_ki:.3f}, Kd={self.current_kd:.3f} (Prev Kp={self.previous_kp:.2f}, Ki={self.previous_ki:.2f}, Kd={self.previous_kd:.2f})")

    def get_params(self) -> Dict[str, float]:
        return {'kp': self.current_kp, 'ki': self.current_ki, 'kd': self.current_kd}

    def get_target(self) -> Any: # Implementación de la interfaz
        return self.target_setpoint

    def reset(self): # Implementación de la interfaz (reset completo)
        logger.debug("[PIDController:reset] FULL RESET: Resetting gains to initial config and clearing internal state.")
        self.current_kp, self.current_ki, self.current_kd = self.initial_kp_config, self.initial_ki_config, self.initial_kd_config
        self.reset_internal_state() # También resetea estados de error y prev_gains a actuales

    def reset_internal_state(self): # Implementación de la interfaz (solo estado interno)
        #logger.debug("[PIDController:reset_internal_state] INTERNAL STATE RESET: Resetting errors, integral. Prev_gains set to current gains.")
        self.previous_error_value = 0.0
        self.accumulated_integral_error = 0.0
        self.previous_accumulated_integral_error = 0.0
        self.current_derivative_error = 0.0
        self.last_proportional_term = 0.0
        self.last_integral_term = 0.0
        self.last_derivative_term = 0.0
        self.previous_kp = self.current_kp
        self.previous_ki = self.current_ki
        self.previous_kd = self.current_kd
        
    def reset_policy(self, reset_level_policy: str): # 'reset_level_policy' (Implementación de la interfaz)
        logger.debug(f"[PIDController:reset_policy] Received reset_policy request with level: '{reset_level_policy}'")
        if reset_level_policy == 'full_params_and_state':
            self.reset()
        elif reset_level_policy == 'internal_state_only':
            self.reset_internal_state()
        # Se podrían añadir más niveles, e.g., 'adaptive_components_only' si el PID tuviera auto-tuning.
        else:
            logger.warning(f"[PIDController:reset_policy] Unknown reset_level '{reset_level_policy}'. Defaulting to 'internal_state_only'.")
            self.reset_internal_state()