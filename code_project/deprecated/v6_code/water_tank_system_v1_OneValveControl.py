# components/systems/water_tank_system_v1_OneValveControl.py
import numpy as np
from scipy.integrate import odeint
from interfaces.dynamic_system import DynamicSystem
from typing import Any, List, Dict
import logging

logger = logging.getLogger(__name__)

class WaterTankSystem(DynamicSystem):
    """
    Implements the dynamics of a single water tank with one controllable inlet valve
    and one fixed outlet valve, based on the physical model.
    The state is the water level `h`. The action `u` is the inlet opening [0, 1].
    """
    def __init__(self,
                 area_m2: float,
                 inflow_coeff_Cd: float,
                 outflow_coeff_Cd: float,
                 outflow_opening_u: float,
                 supply_pressure_pa: float,
                 fluid_density_kg_m3: float,
                 g_accel: float,
                 max_level_m: float
                 ):
        logger.info("[WaterTankSystem] Initializing with physical parameters...")

        if not all(p > 0 for p in [area_m2, inflow_coeff_Cd, outflow_coeff_Cd, supply_pressure_pa, fluid_density_kg_m3, g_accel, max_level_m]):
            raise ValueError("All physical parameters for WaterTankSystem must be positive.")
        if not (0 <= outflow_opening_u <= 1):
            raise ValueError(f"outflow_opening_u ({outflow_opening_u}) must be between 0 and 1.")

        self.P_sup = float(supply_pressure_pa)
        self.A = float(area_m2)
        self.rho = float(fluid_density_kg_m3)
        self.g = float(g_accel)
        self.h_max = float(max_level_m)
        self.last_mass_in_rate: float = 0.0
        self.last_mass_out_rate: float = 0.0
        
        # Pre-calculate constant terms for optimization
        self.in_term_const = inflow_coeff_Cd * np.sqrt(2 * self.rho)
        self.out_term_const = outflow_coeff_Cd * outflow_opening_u * np.sqrt(2 * self.rho * self.g)
        self.inv_rho_A = 1.0 / (self.rho * self.A)

        logger.info("[WaterTankSystem] Initialization complete.")

    def _dynamics(self, state_vector: np.ndarray, t: float, control_input_u: float) -> List[float]:
        """
        Defines the differential equation dh/dt based on mass flow balance.
        dh/dt = (mass_in_rate - mass_out_rate) / (rho * Area)
        """
        h = state_vector[0]

        # Mass flow rate IN (kg/s)
        pressure_diff_term = max(0.0, (self.P_sup - self.rho * self.g * h))
        self.last_mass_in_rate = control_input_u * self.in_term_const * np.sqrt(pressure_diff_term)

        # Mass flow rate OUT (kg/s)
        self.last_mass_out_rate = self.out_term_const * np.sqrt(max(0.0, h))

        # Differential Equation
        dh_dt = (self.last_mass_in_rate - self.last_mass_out_rate) * self.inv_rho_A

        return [dh_dt]

    def apply_action(self, current_state: Any, control_action: float, current_time: float, dt: float) -> np.ndarray:
        """
        Integrates the system dynamics over one time step `dt`.
        """
        time_integration_points = [current_time, current_time + dt]
        current_state_np = np.array(current_state, dtype=float).flatten()

        next_state = odeint(
            self._dynamics,
            current_state_np,
            time_integration_points,
            args=(control_action,)
        )[-1]

        return next_state

    def reset(self, initial_conditions: Dict[str, float]) -> np.ndarray:
        """
        Resets the system to a given initial water level from a dictionary.
        """
        initial_h = float(initial_conditions.get('level', 0.0))
        self.last_mass_in_rate = 0.0
        self.last_mass_out_rate = 0.0
        
        initial_state = np.array([initial_h], dtype=float)
        logger.debug(f"[WaterTankSystem] Reset to initial state: {initial_state}")
        return initial_state
    
    def get_log_system_params(self) -> Dict[str, float]:
        """Devuelve los últimos flujos másicos calculados."""
        return {
            'mass_in_rate': self.last_mass_in_rate,
            'mass_out_rate': self.last_mass_out_rate,
        }