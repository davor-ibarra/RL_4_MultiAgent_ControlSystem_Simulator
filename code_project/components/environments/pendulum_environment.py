from interfaces.environment import Environment
import numpy as np
import logging # Import logging
from typing import Tuple, Dict, Any

class PendulumEnvironment(Environment):
    # Add config to __init__ if needed for reward/termination access later
    def __init__(self, system, controller, agent, reward_function, dt, gain_step, variable_step, reset_gains, config): # Added config
        self.system = system
        self.controller = controller
        self.agent = agent
        self.reward_function = reward_function
        self.dt = dt
        self.gain_step = gain_step
        self.variable_step = variable_step
        self.reset_gains = reset_gains
        self.config = config # Store config
        self.state = None
        self.t = 0.0
        logging.info("PendulumEnvironment initialized.")

    def step(self, actions: Dict[str, Any]) -> Tuple[Any, Tuple[float, float], Any]:
        """Applies actions, steps the system, calculates reward and stability score."""
        if self.state is None:
            logging.error("Environment step called before reset.")
            if 'initial_conditions' in self.config and 'x0' in self.config['initial_conditions']:
                 self.reset(self.config['initial_conditions']['x0'])
                 logging.warning("Environment was auto-reset in step() using initial_conditions from config.")
            else:
                 raise ValueError("Environment not reset and no initial conditions found to auto-reset.")

        # 1. Apply Agent Actions (Update Controller Gains)
        kp, ki, kd = self.controller.kp, self.controller.ki, self.controller.kd
        gain_cfg = self.agent.state_config # Get gain limits/config from agent

        try:
            # --- Gain Update Logic (unchanged) ---
            if not self.variable_step:
                # Fixed step logic
                step = self.gain_step
                if actions['kp'] == 0: kp -= step
                elif actions['kp'] == 2: kp += step
                if actions['ki'] == 0: ki -= step
                elif actions['ki'] == 2: ki += step
                if actions['kd'] == 0: kd -= step
                elif actions['kd'] == 2: kd += step
            else:
                # Variable step logic
                if not isinstance(self.gain_step, dict):
                     logging.warning("Variable step is True but gain_step is not a dict. Using gain_step value for all.")
                     step_kp = step_ki = step_kd = self.gain_step
                else:
                     step_kp = self.gain_step.get('kp', 0)
                     step_ki = self.gain_step.get('ki', 0)
                     step_kd = self.gain_step.get('kd', 0)

                if actions['kp'] == 0: kp -= step_kp
                elif actions['kp'] == 2: kp += step_kp
                if actions['ki'] == 0: ki -= step_ki
                elif actions['ki'] == 2: ki += step_ki
                if actions['kd'] == 0: kd -= step_kd
                elif actions['kd'] == 2: kd += step_kd

            # Clip gains using boundaries from agent's state config
            # Ensure gain names in gain_cfg match 'kp', 'ki', 'kd'
            kp = np.clip(kp, gain_cfg['kp']['min'], gain_cfg['kp']['max']) if 'kp' in gain_cfg else kp
            ki = np.clip(ki, gain_cfg['ki']['min'], gain_cfg['ki']['max']) if 'ki' in gain_cfg else ki
            kd = np.clip(kd, gain_cfg['kd']['min'], gain_cfg['kd']['max']) if 'kd' in gain_cfg else kd

            self.controller.update_params(kp, ki, kd)

        except KeyError as e:
             logging.error(f"Invalid action key received in environment step: {e}. Actions: {actions}")
             # Proceed with potentially unchanged gains.

        # 2. Compute Control Force (using current state and updated gains)
        try:
            force = self.controller.compute_action(self.state)
        except Exception as e:
             logging.error(f"Error computing control action: {e}")
             force = 0.0 # Apply zero force if controller fails

        # 3. Apply Force to System Dynamics
        try:
            next_state = self.system.apply_action(self.state, force, self.t, self.dt)
        except Exception as e:
             logging.error(f"Error applying action to system dynamics: {e}")
             next_state = self.state # Keep current state if dynamics fail

        # 4. Calculate Reward and Stability Score
        reward = 0.0
        stability_score = 1.0 # Default
        try:
            # Pass necessary arguments to reward function
            # calculate now returns a tuple: (reward, stability_score)
            reward, stability_score = self.reward_function.calculate(self.state, force, next_state, self.t)
        except Exception as e:
             logging.error(f"Error calculating reward/stability: {e}", exc_info=True)
             # Assign defaults if calculation fails

        # 5. Update Internal State and Time
        self.state = next_state
        self.t += self.dt

        # Return: next state, (reward, stability_score) tuple, and force applied
        return next_state, (reward, stability_score), force

    # --- reset method remains the same ---
    def reset(self, initial_conditions):
        """Resets the environment state, controller, and agent parameters."""
        logging.debug(f"Resetting environment with initial conditions: {initial_conditions}")
        try:
            self.state = self.system.reset(initial_conditions)
            self.t = 0.0
            if self.reset_gains:
                self.controller.reset()
                logging.debug("Controller gains reset.")
            else:
                 # Reset only error/integral terms if gains are kept across episodes
                 self.controller.reset_episode() # Assumes this method exists in PIDController
            # Agent reset (epsilon/LR decay) should happen here or before episode start
            self.agent.reset_agent() # Use reset_agent for clarity
            logging.debug(f"Agent parameters updated: epsilon={self.agent.epsilon:.4f}, LR={self.agent.learning_rate:.4f}")
            return self.state
        except Exception as e:
            logging.error(f"Error during environment reset: {e}")
            raise # Reraise exception as reset failure is critical

    # --- check_termination method remains the same ---
    def check_termination(self, config):
        """Checks angle, cart limits and stabilization criteria."""
        # Check if state is valid before accessing elements
        if self.state is None or len(self.state) < 4:
             logging.warning("check_termination called with invalid state. Returning False.")
             return False, False, False

        # Use .get for safer access to config dictionary
        env_config = config.get('environment', {}) # General environment settings
        sim_config = config.get('simulation', {}) # Simulation specific limits
        stab_config = config.get('stabilization_criteria', {}) # Stabilization criteria
        ctrl_config = env_config.get('controller', {}).get('params', {}) # Controller params for setpoint

        # Angle Limit Check
        angle_limit = sim_config.get('angle_limit', np.pi) # Default if missing
        use_angle_limit = sim_config.get('use_angle_limit', True)
        angle_exceeded = use_angle_limit and (abs(self.state[2]) > angle_limit)

        # Cart Limit Check
        cart_limit = sim_config.get('cart_limit', 5.0) # Default if missing
        use_cart_limit = sim_config.get('use_cart_limit', True)
        cart_exceeded = use_cart_limit and (abs(self.state[0]) > cart_limit)

        # Stabilization Check
        # Use controller setpoint for stabilization check (get from config)
        setpoint = ctrl_config.get('setpoint', 0.0) # Default setpoint if not found
        angle_threshold = stab_config.get('angle_threshold', 0.01) # Default if missing
        velocity_threshold = stab_config.get('velocity_threshold', 0.01) # Default if missing
        stabilized = (
            abs(self.state[2] - setpoint) < angle_threshold and
            abs(self.state[3]) < velocity_threshold
        )
        return angle_exceeded, cart_exceeded, stabilized
    
    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """
        Triggers the update of statistics within the reward function's components
        (like the stability calculator).
        """
        if hasattr(self.reward_function, 'update_calculator_stats'):
            try:
                 self.reward_function.update_calculator_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                 logging.error(f"Error calling update_calculator_stats on reward function: {e}", exc_info=True)
        else:
            logging.debug("Reward function does not have 'update_calculator_stats' method.")