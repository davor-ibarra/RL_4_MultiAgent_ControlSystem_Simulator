# components/controllers/pid_controller.py
from interfaces.controller import Controller
from typing import Dict, List, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PIDController(Controller):
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float, 
                 setpoint: float, 
                 dt_sec: float,
                 name_objective_var: str,
                 actuator_limits: List,
                 clipping_output: bool = False,
                 normalize_output: bool = False,
                 error_is_setpoint_minus_pv: bool = False,
                 anti_windup: Optional[Dict[str, Any]] = None
                 ):
        # Parámetros vienen de config['environment']['controller']['params'] + 'dt_sec'
        # anti_windup vendrá de config['environment']['controller']['params']['anti_windup']
        logger.info(f"[PIDController] Initializing with Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint}, name_obj_var={name_objective_var}, dt_sec={dt_sec}, AntiWindupConfig={anti_windup}")

        params = list(map(float, [kp, ki, kd, setpoint, dt_sec]))
        self.name_objective_var = str(name_objective_var)

        if not all(np.isfinite(p) for p in params):
            raise ValueError("Kp, Ki, Kd, Setpoint, and dt_sec must be finite numbers.")
        if params[4] <= 0:
            raise ValueError("dt_sec must be a positive number.")

        self.initial_kp, self.initial_ki, self.initial_kd = params[0], params[1], params[2]
        self.target_setpoint = params[3]
        self._dt = params[4]
        self.actuator_limits = list(actuator_limits)
        self.clipping_output = bool(clipping_output)
        self.normalize_output = bool(normalize_output)

        self.current_kp, self.current_ki, self.current_kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.previous_kp, self.previous_ki, self.previous_kd = self.current_kp, self.current_ki, self.current_kd

        self.current_error: float = 0.0
        self.previous_error: float = 0.0
        self.current_accumulated_integral_error: float = 0.0
        self.previous_accumulated_integral_error: float = 0.0
        self.current_derivative_error: float = 0.0

        self.last_proportional_term: float = 0.0
        self.last_integral_term: float = 0.0
        self.last_derivative_term: float = 0.0

        self.control_action: float = 0.0
        self.raw_control_action: float = 0.0
        # Atributos para la acción localmente limitada y normalizada
        self.limited_control_action: float = 0.0
        self.normalize_control_action: float = 0.0
        # Almacena la acción efectiva del paso anterior (informada por el Environment)
        self.last_effective_control_action: float = 0.0
        # Almacena el error de saturación real del paso anterior (calculado en track_actuator_output)
        self.last_saturation_error: float = 0.0

        # Actuator
        self.actuator_min = float(self.actuator_limits[0])
        self.actuator_max = float(self.actuator_limits[1])
        if self.actuator_max <= self.actuator_min:
            raise ValueError("Actuator 'output_max' must be greater than 'output_min' in config.")
        self.actuator_range = self.actuator_max - self.actuator_min
        logger.info(
                    f"[PIDController] Actuator range [{self.actuator_min}, {self.actuator_max}], "
                    f"clipping_output={self.clipping_output}, normalize_output={self.normalize_output}"
                )

        self.error_is_setpoint_minus_pv = bool(error_is_setpoint_minus_pv)

        # Anti-windup parameters
        self.aw_enabled = bool(anti_windup.get('enabled', False))
        self.aw_method  = str(anti_windup.get('method', 'none')).lower()        
        self.aw_beta    = float(anti_windup.get('back_calculation_betha', 0.0))                   # solo back-calculation

        self.reset_internal_state()
        logger.debug("[PIDController] Initialization complete. Internal state reset.")

    def compute_action(self, state_dict: Dict[str, Any]) -> float: # 'current_system_state'
        try:
            process_variable = float(state_dict[self.name_objective_var])
            
            # Según sistema de referencia
            if self.error_is_setpoint_minus_pv:
                self.current_error = self.target_setpoint - process_variable
            else:
                self.current_error = process_variable - self.target_setpoint

            # --- Proportonial Term ---
            self.last_proportional_term = self.current_kp * self.current_error

            # --- Integral Term ---
            self._update_integral_term()
            #self.current_accumulated_integral_error += self.current_error * self._dt
            self.last_integral_term = self.current_ki * self.current_accumulated_integral_error

            # --- Derivative Term ---
            self.current_derivative_error = (self.current_error - self.previous_error) / self._dt if self._dt > 0 else 0.0
            self.last_derivative_term = self.current_kd * self.current_derivative_error
            
            # --- Control Action PID ---
            self.raw_control_action = self.last_proportional_term + self.last_integral_term + self.last_derivative_term
            self.control_action = self.raw_control_action

            # --- Lógica de Limitación Local ---
            # Se aplican los límites propios del controlador si están activados.
            if self.clipping_output:
                self.limited_control_action = np.clip(self.raw_control_action, self.actuator_min, self.actuator_max)
                self.control_action = self.limited_control_action
            if self.normalize_output:
                control_action = np.clip(self.control_action, self.actuator_min, self.actuator_max)
                # La normalización se aplica sobre la acción ya clipeada
                self.normalize_control_action = 2.0 * (control_action - self.actuator_min) / self.actuator_range - 1.0
                self.control_action = self.normalize_control_action
            
            # Actualiza los estados de error para el siguiente ciclo.
            self.previous_error = self.current_error
            self.previous_accumulated_integral_error = self.current_accumulated_integral_error

            # El método devuelve la acción procesada localmente.
            return float(self.control_action)
        
        except Exception as e_compute:
            logger.error("[PIDController:compute_action] Unexpected error: %s", e_compute, exc_info=True)
            self.reset_internal_state()           # evita arrastre del integral
            return 0.0

    def _update_integral_term(self):
        """
        Actualiza el término integral, incorporando la lógica de anti-windup
        basada en el estado del ciclo anterior.
        """
        if self.current_ki == 0:
            self.current_accumulated_integral_error = 0.0
            self.last_integral_term = 0.0
            return
        
        is_in_windup_state = False
        # Para el método 'conditional', se evalúa si se debe acumular
        if self.aw_enabled and self.aw_method == 'conditional':
            sat = self.last_saturation_error
            is_in_windup_state = (sat > 0 and self.current_error > 0) or (sat < 0 and self.current_error < 0)
        
        # Acumular el error solo si no estamos en una condición de windup
        if not is_in_windup_state:
            self.current_accumulated_integral_error += self.current_error * self._dt

        # Para el método 'back_calculation', se aplica una corrección
        if self.aw_enabled and self.aw_method == 'back_calculation':
            correction = (self.aw_beta * self.last_saturation_error) / max(abs(self.current_ki), 1e-6)
            self.current_accumulated_integral_error -= correction
        # Lógica para 'last_reference_calculation'
        elif self.aw_enabled and self.aw_method == 'last_reference_calculation' and self.current_ki != 0:
            numerator = (self.previous_kp * self.previous_error) + \
                        (self.previous_ki * self.previous_accumulated_integral_error * self._dt) - \
                        (self.current_kp * self.current_error)
            self.current_accumulated_integral_error = numerator / self.current_ki
    
    def track_actuator_output(self, effective_actuator_value: float):
        """
        El coordinador (Environment) informa al PID cuál fue su contribución final.
        El PID usa esta información para calcular el error de saturación real y actualizar su estado.
        """
        # El error de saturación es la diferencia entre la acción ideal (raw) y la contribución efectiva.
        self.last_saturation_error = self.raw_control_action - effective_actuator_value
        # Almacena la contribución efectiva para la lógica 'conditional' del siguiente ciclo.
        self.last_effective_control_action = effective_actuator_value


    def update_params(self, new_gains_dict: Dict[str, float]):
        self.previous_kp, self.previous_ki, self.previous_kd = self.current_kp, self.current_ki, self.current_kd
        
        kp_name = f'kp_{self.name_objective_var}'
        ki_name = f'ki_{self.name_objective_var}'
        kd_name = f'kd_{self.name_objective_var}'

        new_kp = new_gains_dict.get(kp_name, self.current_kp)
        new_ki = new_gains_dict.get(ki_name, self.current_ki)
        new_kd = new_gains_dict.get(kd_name, self.current_kd)

        self.current_kp = float(new_kp) if np.isfinite(new_kp) else self.previous_kp
        self.current_ki = float(new_ki) if np.isfinite(new_ki) else self.previous_ki
        self.current_kd = float(new_kd) if np.isfinite(new_kd) else self.previous_kd

    def get_params(self) -> Dict[str, float]:
        return {
            f'kp_{self.name_objective_var}': self.current_kp,
            f'ki_{self.name_objective_var}': self.current_ki,
            f'kd_{self.name_objective_var}': self.current_kd
        }

    def get_target(self) -> Any: # Implementación de la interfaz
        return self.target_setpoint
    
    def set_target(self, new_target: Any):
        """Dynamically updates the controller's setpoint."""
        try:
            self.target_setpoint = float(new_target)
        except (ValueError, TypeError):
            logger.warning(f"[PIDController] Could not set new target; invalid value: {new_target}")

    def reset(self): # Implementación de la interfaz (reset completo)
        logger.debug("[PIDController:reset] FULL RESET: Resetting gains to initial config and clearing internal state.")
        self.current_kp, self.current_ki, self.current_kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.reset_internal_state() # También resetea estados de error y prev_gains a actuales

    def reset_internal_state(self): # Implementación de la interfaz (solo estado interno)
        #logger.debug("[PIDController:reset_internal_state] INTERNAL STATE RESET: Resetting errors, integral. Prev_gains set to current gains.")
        self.previous_error = 0.0
        self.current_accumulated_integral_error = 0.0
        self.previous_accumulated_integral_error = 0.0
        self.current_derivative_error = 0.0
        self.last_proportional_term = 0.0
        self.last_integral_term = 0.0
        self.last_derivative_term = 0.0
        self.previous_kp = self.current_kp
        self.previous_ki = self.current_ki
        self.previous_kd = self.current_kd
        self.control_action = 0.0
        self.raw_control_action = 0.0
        self.limited_control_action = 0.0
        self.normalize_control_action = 0.0
        self.last_effective_control_action = 0.0
        self.last_saturation_error = 0.0
        
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

    def get_params_log(self) -> Dict[str, Any]:
        """Implements the loggable parameters interface for the PID controller."""
        obj_name = self.name_objective_var
        return {
            f'kp_{obj_name}': self.current_kp,
            f'ki_{obj_name}': self.current_ki,
            f'kd_{obj_name}': self.current_kd,
            f'error_{obj_name}': self.current_error,
            f'integral_error_{obj_name}': self.current_accumulated_integral_error,
            f'p_term_{obj_name}': self.last_proportional_term,
            f'i_term_{obj_name}': self.last_integral_term,
            f'd_term_{obj_name}': self.last_derivative_term,
            f'control_action_{obj_name}': self.control_action,
            f'raw_action_{obj_name}': self.raw_control_action, # Acción ideal
            f'limited_action_{obj_name}': self.limited_control_action,
            f'normalized_action_{obj_name}': self.normalize_control_action,
            f'effective_action_{obj_name}': self.last_effective_control_action, # Contribución real
            f'saturation_error_{obj_name}': self.last_saturation_error, # Error de saturación real
        }