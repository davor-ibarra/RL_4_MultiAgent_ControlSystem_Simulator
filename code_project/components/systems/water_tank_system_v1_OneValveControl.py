# components/systems/water_tank_system_v1_OneValveControl.py
import numpy as np
from scipy.integrate import odeint
from interfaces.dynamic_system import DynamicSystem
from typing import Any, List, Dict
import logging

logger = logging.getLogger(__name__)

class WaterTankSystem(DynamicSystem):
    """
    Implements the dynamics of a common industrial single water tank,
    with a controllable valves and pumps of inlet and outlet,
    The action 'u' is the valve and actutator opening [0, 1].
    """
    def __init__(self,
                area_m2: float,                         # Área transversal del tanque (cte)
                inflow_pump_P_max_pa: float,            # Presión máxima que entrega la bomba de entrada (cte)
                inflow_pump_opening_u: float,           # Apertura normalizada de la bomba de entrada (fija)
                inflow_pump_k: float,                   # Resistencia interna de la bomba de entrada (cte)
                inflow_coeff_Cd: float,                 # Coeficiente de descarga de la válvula de entrada (cte)
                inflow_ao_m2: float,                    # Área de sección de la válvula de entrada (cte)
                inflow_opening_u: float,                # Apertura normalizada de la válvula de entrada (variable de control)
                outflow_pump_P_max_pa: float,           # Presión máxima que entrega la bomba de salida (cte)
                outflow_pump_opening_u: float,          # Apertura normalizada de la bomba de salida (fija)
                outflow_pump_k: float,                  # Resistencia interna de la bomba de salida (cte)
                outflow_coeff_Cd: float,                # Coeficiente de descarga de la válvula de salida (cte)
                outflow_ao_m2: float,                   # Área de sección de la válvula de salida (cte)
                outflow_opening_u: float,               # Apertura normalizada de la válvula de salida (fija)
                fluid_density_kg_m3: float,             # Densidad del fluido (cte)
                g_accel: float,                         # Aceleración de la gravedad (cte)
                max_level_m: float                      # Altura máxima del tanque (cte)
                ):
        logger.info("[WaterTankSystem] Initializing with physical parameters...")

        # Assign all parameters from config
        self.A = float(area_m2)
        self.rho = float(fluid_density_kg_m3)
        self.g = float(g_accel)
        self.h_max = float(max_level_m)
        
        # Inflow parameters
        self.in_P_pi_max = float(inflow_pump_P_max_pa)
        self.in_K_pi = float(inflow_pump_k)
        self.in_Cd = float(inflow_coeff_Cd)
        self.in_Ao = float(inflow_ao_m2)
        self.init_u_valve_in = float(inflow_opening_u)
        self.init_u_pump_in = float(inflow_pump_opening_u)
        self.u_valve_in = float(inflow_opening_u)
        self.u_pump_in = float(inflow_pump_opening_u)
        
        # Outflow parameters
        self.out_P_po_max = float(outflow_pump_P_max_pa)
        self.out_K_po = float(outflow_pump_k)
        self.out_Cd = float(outflow_coeff_Cd)
        self.out_Ao = float(outflow_ao_m2)
        self.u_valve_out = float(outflow_opening_u)
        self.u_pump_out = float(outflow_pump_opening_u)
        self.init_u_valve_out = float(outflow_opening_u)
        self.init_u_pump_out = float(outflow_pump_opening_u)

        # Pre-calculate constant parts of resistance for performance
        self.in_valve_denom_const = 2 * (self.in_Cd * self.in_Ao)**2
        self.out_valve_denom_const = 2 * (self.out_Cd * self.out_Ao * self.u_valve_out)**2

        # Last calculated flow rates for logging
        self.Q_in_total: float = 0.0
        self.Q_out_total: float = 0.0

        logger.info("[WaterTankSystem] Initialization complete.")

    def _calculate_inflow_q(self, u_valve: float, u_pump: float) -> float:
        if u_valve <= 1e-6 or self.in_valve_denom_const <= 1e-9:
            return 0.0
        valve_resistance = self.rho / (self.in_valve_denom_const * u_valve**2)
        numerator = self.in_P_pi_max * u_pump
        denominator = self.in_K_pi + valve_resistance
        return np.sqrt(numerator / denominator)

    def _calculate_outflow_q(self, h: float) -> float:
        if self.out_valve_denom_const <= 1e-9:
            return 0.0
        valve_resistance = self.rho / self.out_valve_denom_const
        numerator = self.rho * self.g * h + self.out_P_po_max * self.u_pump_out
        denominator = self.out_K_po + valve_resistance
        return np.sqrt(numerator / denominator)

    def _dynamics(self, state_vector: np.ndarray, t: float, u_valve_in_control: float, u_pump_in_control: float) -> List[float]:
        """
        Defines the differential equation dh/dt based on volumetric flow balance.
        dh/dt = (Q_in - Q_out) / (Area)
        """
        h = state_vector[0]

        # --- Inflows (m^3/s) ---
        self.Q_in_total = self._calculate_inflow_q(u_valve_in_control, u_pump_in_control)
        # --- Outflows (m^3/s) ---
        self.Q_out_total = self._calculate_outflow_q(h)
        
        # --- Volumetric Balance Equation ---
        dh_dt = (1.0 / self.A) * (self.Q_in_total - self.Q_out_total)

        return [dh_dt]

    def apply_action(self, current_state: Any, control_action: Any, current_time: float, dt: float) -> np.ndarray:
        """
        Integrates the system dynamics over one time step `dt`.
        """
        time_integration_points = [current_time, current_time + dt]
        current_state_np = np.array(current_state, dtype=float).flatten()
        self.u_valve_in = control_action            # Agregar variables de control que se requieran

        next_state = odeint(
            self._dynamics,
            current_state_np,
            time_integration_points,
            args=(self.u_valve_in, self.u_pump_in)
        )[-1]

        return next_state

    def reset(self, initial_conditions: Dict[str, float]) -> np.ndarray:
        """
        Resets the system to a given initial water level from a dictionary.
        """
        initial_h = float(initial_conditions.get('level', 0.0))
        self.Q_in_total, self.Q_out_total = 0.0, 0.0
        self.u_valve_in = self.init_u_valve_in
        self.u_pump_in = self.init_u_pump_in
        self.u_valve_out = self.init_u_valve_out
        self.u_pump_out = self.init_u_pump_out
        
        initial_state = np.array([initial_h], dtype=float)
        logger.debug(f"[WaterTankSystem] Reset to initial state: {initial_state}")
        return initial_state
    
    def get_log_system_params(self) -> Dict[str, float]:
        """Devuelve los últimos flujos másicos calculados."""
        return {
            'Q_in': self.Q_in_total,
            'Q_out': self.Q_out_total,
        }