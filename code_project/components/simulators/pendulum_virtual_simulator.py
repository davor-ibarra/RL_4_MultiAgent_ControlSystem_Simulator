import numpy as np
import logging
from typing import Any, Dict
import copy # Needed for deep copying controller state if necessary

from interfaces.virtual_simulator import VirtualSimulator
from interfaces.dynamic_system import DynamicSystem
from interfaces.controller import Controller # Import Controller interface
from interfaces.reward_function import RewardFunction # Import RewardFunction interface

class PendulumVirtualSimulator(VirtualSimulator):
    """
    Runs virtual simulations for the Pendulum environment interval.
    Uses copies or careful state management to avoid altering real components.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller, # Use Controller interface type hint
                 reward_function: RewardFunction, # Use RewardFunction interface
                 dt: float):
        """
        Initializes the virtual simulator with references to the core components.

        Args:
            system: The dynamic system model (e.g., InvertedPendulumSystem).
            controller: The controller instance (e.g., PIDController). MUST support
                        methods like update_params, compute_action, reset_episode,
                        and have attributes kp, ki, kd.
            reward_function: The reward function instance.
            dt: The simulation time step.
        """
        self.system = system
        # Store the original controller to reset state later if needed,
        # but we'll mostly work by setting parameters directly.
        self.original_controller = controller
        self.reward_function = reward_function
        self.dt = dt
        if not hasattr(controller, 'reset_episode'):
             logging.warning("VirtualSimulator: Provided controller might lack 'reset_episode' method, which is recommended for resetting internal state.")
        logging.info("PendulumVirtualSimulator initialized.")

    def run_interval_simulation(self,
                                initial_state_vector: Any,
                                start_time: float,
                                duration: float,
                                controller_gains_dict: Dict[str, float]) -> float:
        """
        Runs the virtual simulation interval.
        """
        virtual_state = np.array(initial_state_vector).flatten() # Ensure it's a flat numpy array
        virtual_time = start_time
        accumulated_reward = 0.0
        num_steps = max(1, int(round(duration / self.dt))) # Ensure at least one step

        # --- Store original controller state ---
        # It's safer to store and restore gains rather than deep copying
        # the entire controller, especially if it has complex internal states.
        original_gains = {
            'kp': self.original_controller.kp,
            'ki': self.original_controller.ki,
            'kd': self.original_controller.kd
        }
        # Store other relevant state if necessary (e.g., integral term)
        # For PID, resetting the episode error/integral might be sufficient.
        # If the controller had more complex state, deepcopy might be needed,
        # but let's try resetting first for efficiency.
        try:
             # Reset controller's internal error/integral state for the virtual run
             if hasattr(self.original_controller, 'reset_episode'):
                 self.original_controller.reset_episode()
        except Exception as e:
             logging.error(f"VirtualSimulator: Error resetting controller episode state: {e}", exc_info=True)
             # Continue simulation but be aware controller state might be incorrect

        try:
            # --- Set virtual gains ---
            kp_virt = controller_gains_dict['kp']
            ki_virt = controller_gains_dict['ki']
            kd_virt = controller_gains_dict['kd']
            self.original_controller.update_params(kp_virt, ki_virt, kd_virt)

            # --- Run virtual steps ---
            for _ in range(num_steps):
                # 1. Compute virtual force using the temporarily set gains
                try:
                    # Pass the *virtual* state to compute_action
                    force = self.original_controller.compute_action(virtual_state)
                except Exception as e:
                    logging.error(f"VirtualSimulator: Error computing virtual action at t={virtual_time}: {e}")
                    force = 0.0

                # 2. Apply virtual force to system dynamics
                try:
                    next_virtual_state = self.system.apply_action(virtual_state, force, virtual_time, self.dt)
                except Exception as e:
                    logging.error(f"VirtualSimulator: Error applying virtual action at t={virtual_time}: {e}")
                    next_virtual_state = virtual_state # Keep state if dynamics fail

                # 3. Calculate instantaneous reward for this virtual step
                # The reward function itself shouldn't rely on the real agent's state
                try:
                    # calculate returns (reward, stability_score) - we only need reward here
                    inst_reward, _ = self.reward_function.calculate(virtual_state, force, next_virtual_state, virtual_time)
                    accumulated_reward += inst_reward
                except Exception as e:
                    logging.error(f"VirtualSimulator: Error calculating virtual reward at t={virtual_time}: {e}")
                    # Don't add reward if calculation fails

                # 4. Update virtual state and time for the next virtual step
                virtual_state = next_virtual_state
                virtual_time += self.dt

        except KeyError as e:
             logging.error(f"VirtualSimulator: Missing gain key in controller_gains_dict: {e}. Dict: {controller_gains_dict}")
             accumulated_reward = 0.0 # Or a penalty value
        except Exception as e:
            logging.error(f"VirtualSimulator: Unexpected error during virtual simulation: {e}", exc_info=True)
            accumulated_reward = 0.0 # Or a penalty value
        finally:
            # --- IMPORTANT: Restore original controller gains ---
            try:
                 self.original_controller.update_params(original_gains['kp'], original_gains['ki'], original_gains['kd'])
                 # Also reset episode state again after virtual run to be safe
                 if hasattr(self.original_controller, 'reset_episode'):
                       self.original_controller.reset_episode()
            except Exception as e:
                 logging.critical(f"VirtualSimulator: FAILED to restore original controller state after virtual run! Error: {e}. Subsequent steps may be incorrect.")

        return accumulated_reward