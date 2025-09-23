import numpy as np
import logging
from typing import Any, Dict
import copy

from interfaces.virtual_simulator import VirtualSimulator
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller
from interfaces.reward_function import RewardFunction

class PendulumVirtualSimulator(VirtualSimulator):
    """
    Runs virtual simulations for a Pendulum environment interval.
    Uses the *real* controller instance but temporarily modifies its gains
    and resets its internal state (error, integral) for the simulation duration.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller,
                 reward_function: RewardFunction,
                 dt: float):
        self.system = system
        self.controller_template = controller
        self.reward_function = reward_function
        self.dt = dt

        # --- VALIDACIONES (Check attributes on the template) ---
        if not hasattr(self.controller_template, 'reset_internal_state'):
            logging.error("VirtualSimulator CRITICAL: Provided controller template MUST have 'reset_internal_state' method.")
            raise AttributeError("VirtualSimulator: Provided controller template MUST have 'reset_internal_state' method.")
        if not hasattr(self.controller_template, 'update_params'):
            logging.error("VirtualSimulator CRITICAL: Provided controller template MUST have 'update_params' method.")
            raise AttributeError("VirtualSimulator: Provided controller template MUST have 'update_params' method.")
        if not hasattr(self.controller_template, 'compute_action'):
            logging.error("VirtualSimulator CRITICAL: Provided controller template MUST have 'compute_action' method.")
            raise AttributeError("VirtualSimulator: Provided controller template MUST have 'compute_action' method.")
        if not all(hasattr(self.controller_template, attr) for attr in ['kp', 'ki', 'kd']):
            logging.error("VirtualSimulator CRITICAL: Provided controller template MUST have 'kp', 'ki', 'kd' attributes.")
            raise AttributeError("VirtualSimulator: Provided controller template MUST have 'kp', 'ki', 'kd' attributes.")
        logging.info("PendulumVirtualSimulator initialized (using controller template).")

    # @formula: R_cf = Sum[ R(s_virt(t), a_virt(t), s_virt(t+dt), t) ] over interval
    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        
        virtual_state = np.array(initial_state_vector).flatten()
        virtual_time = start_time
        accumulated_reward = 0.0
        num_steps = max(1, int(round(duration / self.dt)))

        # --- Create an independent copy of the controller for this virtual run ---
        try:
            virtual_controller = copy.deepcopy(self.controller_template)
            logging.debug(f"VirtualSim: Created deepcopy of controller template. ID: {id(virtual_controller)}")
        except Exception as e:
            logging.error(f"VirtualSimulator CRITICAL: Failed to deepcopy controller template: {e}", exc_info=True)
            return -1000.0 # Return very low reward on critical failure

        try:
            # Reset the *virtual* controller's internal error/integral state
            virtual_controller.reset_internal_state()

            # --- Set virtual gains ---
            kp_virt = controller_gains_dict['kp']
            ki_virt = controller_gains_dict['ki']
            kd_virt = controller_gains_dict['kd']
            virtual_controller.update_params(kp_virt, ki_virt, kd_virt)
            logging.debug(f"VirtualSim ID {id(virtual_controller)}: Set gains to {kp_virt:.4f}, {ki_virt:.4f}, {kd_virt:.4f}")

            # --- Run virtual steps ---
            for step_num in range(num_steps):
                # 1. Compute virtual force using the *temporarily set* gains and *virtual state*
                try:
                    # Use the virtual controller instance
                    force = virtual_controller.compute_action(virtual_state)
                except Exception as e:
                    logging.error(f"VirtualSim ID {id(virtual_controller)} Error computing virtual action step {step_num} t={virtual_time:.4f}: {e}", exc_info=True)
                    force = 0.0

                # 2. Apply virtual force to system dynamics (system is assumed stateless or reset elsewhere)
                try:
                    next_virtual_state = self.system.apply_action(virtual_state, force, virtual_time, self.dt)
                except Exception as e:
                    logging.error(f"VirtualSim ID {id(virtual_controller)} Error applying virtual action step {step_num} t={virtual_time:.4f}: {e}", exc_info=True)
                    next_virtual_state = virtual_state # Keep state if dynamics fail

                # 3. Calculate instantaneous reward for this virtual step
                try:
                    # Use the configured reward function
                    inst_reward, _ = self.reward_function.calculate(virtual_state, force, next_virtual_state, virtual_time)
                    accumulated_reward += inst_reward
                except Exception as e:
                    logging.error(f"VirtualSim ID {id(virtual_controller)} Error calculating virtual reward step {step_num} t={virtual_time:.4f}: {e}", exc_info=True)
                    # Don't add reward if calculation fails

                # 4. Update virtual state and time for the next virtual step
                virtual_state = next_virtual_state
                virtual_time += self.dt

        except KeyError as e:
            logging.error(f"VirtualSimulator: Missing gain key in controller_gains_dict: {e}. Dict: {controller_gains_dict}")
            accumulated_reward = -1000.0 # Penalize heavily if gains missing
        except Exception as e:
            logging.error(f"VirtualSimulator: Unexpected error during virtual simulation ID {id(virtual_controller)}: {e}", exc_info=True)
            accumulated_reward = -1000.0 # Penalize heavily on other errors
        # --- NO finally block needed to restore state, the original controller was untouched ---

        logging.debug(f"VirtualSim ID {id(virtual_controller)}: Finished. Accumulated Reward = {accumulated_reward:.4f}")
        return accumulated_reward