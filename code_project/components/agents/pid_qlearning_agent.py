import numpy as np
import logging
from interfaces.rl_agent import RLAgent
from interfaces.reward_strategy import RewardStrategy # Import Reward Strategy Interface
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union

class PIDQLearningAgent(RLAgent):
    def __init__(self, state_config, num_actions, gain_step, variable_step,
                 reward_strategy: RewardStrategy,
                 discount_factor=0.98, epsilon=1.0, epsilon_min=0.1,
                 epsilon_decay=0.99954, learning_rate=1.0, learning_rate_min=0.01,
                 learning_rate_decay=0.999425, use_epsilon_decay=True, use_learning_rate_decay=True,
                 shadow_baseline_params: Optional[Dict] = None, # Recibe params desde world_initializer
                 q_init_value=0.0, visit_init_value=0):

        logging.info("Initializing PIDQLearningAgent...")
        self.state_config = self._validate_and_prepare_state_config(state_config)
        self.num_actions = num_actions
        self.q_init_value = q_init_value
        self.visit_init_value = visit_init_value
        self.float_precision_for_saved_keys = 6 # For get_agent_state_for_saving

        self.gain_step = gain_step
        self.variable_step = variable_step
        self.discount_factor = discount_factor
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.learning_rate, self.learning_rate_min, self.learning_rate_decay = learning_rate, learning_rate_min, learning_rate_decay
        self.use_epsilon_decay, self.use_learning_rate_decay = use_epsilon_decay, use_learning_rate_decay

        # --- Reward Strategy ---
        self.reward_strategy = reward_strategy
        logging.info(f"Agent using Reward Strategy: {type(self.reward_strategy).__name__}")

        # --- Shadow Baseline Initialization (if strategy requires it) ---
        # Check if the provided strategy is ShadowBaseline (or similar)
        # This relies on the class name, consider a better check if needed (e.g., isinstance or attribute check)
        self.is_shadow_mode = "ShadowBaseline" in type(self.reward_strategy).__name__
        if self.is_shadow_mode:
             shadow_params = shadow_baseline_params if shadow_baseline_params else {}
             # Beta now lives inside the Shadow strategy, but we need baseline init value
             self.baseline_init_value = shadow_params.get('baseline_init_value', 0.0)
             logging.info(f"Shadow Baseline mode detected. Baseline init value: {self.baseline_init_value}")
        else:
             self.baseline_init_value = 0.0 # Default if not shadow mode

        self.ordered_state_vars_for_gain = {}
        self.gain_variables = ['kp', 'ki', 'kd']

        self.q_tables_np = {}
        self.visit_counts_np = {}
        self.baseline_tables_np = {} # Always create, but only fill if shadow mode

        # Store last calculated TD errors for logging
        self._last_td_errors: Dict[str, float] = {}

        # --- Initialize NumPy Tables ---
        for gain in self.gain_variables:
            if self.state_config.get(gain, {}).get('enabled', False):
                logging.info(f"Pre-initializing NumPy tables for gain '{gain}'...")
                ordered_vars_list = self._get_ordered_vars_for_gain(gain)
                self.ordered_state_vars_for_gain[gain] = ordered_vars_list
                try:
                    state_dims = [self.state_config[var]['bins'] for var in ordered_vars_list]
                    q_visit_shape = tuple(state_dims + [self.num_actions])
                    baseline_shape = tuple(state_dims) # Shape for B(s)
                    logging.debug(f"  - State variable order for '{gain}': {ordered_vars_list}")
                    logging.debug(f"  - Q/Visit Array shape for '{gain}': {q_visit_shape}")
                    logging.debug(f"  - Baseline Array shape for '{gain}': {baseline_shape}")

                    self.q_tables_np[gain] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
                    self.visit_counts_np[gain] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)
                    # Initialize baseline table, even if not strictly "shadow" mode, doesn't hurt much
                    self.baseline_tables_np[gain] = np.full(baseline_shape, self.baseline_init_value, dtype=np.float32)

                except KeyError as e:
                    logging.error(f"Error getting bin count for variable {e} while creating array for gain '{gain}'. Check state_config. Skipping initialization for '{gain}'.")
                    if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
                    if gain in self.q_tables_np: del self.q_tables_np[gain]
                    if gain in self.visit_counts_np: del self.visit_counts_np[gain]
                    if gain in self.baseline_tables_np: del self.baseline_tables_np[gain]
                    continue # Skip to next gain
            else:
                logging.debug(f"Skipping NumPy array creation for disabled gain: {gain}")
        logging.info("PIDQLearningAgent initialized successfully.")

    # --- Helper methods (_get_ordered_vars_for_gain, _get_representative_value_from_index, _discretize_value, _validate_and_prepare_state_config, get_discrete_state_indices_tuple) ---
    # --- (unchanged from Step 1 description) ---
    def _get_ordered_vars_for_gain(self, gain):
        # ... (implementation as before) ...
        ordered_vars = OrderedDict()
        # Include non-gain enabled state variables first
        for var, cfg in self.state_config.items():
            if cfg['enabled'] and var not in self.gain_variables:
                ordered_vars[var] = True
        # Add the current gain variable if enabled
        if self.state_config.get(gain, {}).get('enabled', False):
            ordered_vars[gain] = True
        return list(ordered_vars.keys())

    def _get_representative_value_from_index(self, index, var_name):
        # ... (implementation as before) ...
        if var_name not in self.state_config or not self.state_config[var_name]['enabled']:
             # logging.warning(f"Attempted to get representative value for disabled/missing var: {var_name}")
             return 0.0 # Or raise error?
        config = self.state_config[var_name]
        min_val, max_val, bins = config.get('min', 0.0), config.get('max', 0.0), config.get('bins', 1)
        if not isinstance(bins, int) or bins <= 0: return min_val # Or avg?
        if min_val >= max_val: return min_val
        # Ensure index is within bounds
        safe_index = np.clip(index, 0, bins - 1)
        if bins == 1:
             value = (min_val + max_val) / 2.0
        else:
             # Calculate step size, handle potential division by zero if bins=1 (though handled above)
             step = (max_val - min_val) / max(1, bins - 1)
             value = min_val + safe_index * step
        # Clip value to ensure it's within the defined min/max range
        return float(np.clip(value, min_val, max_val))

    def _discretize_value(self, value, config):
        # ... (implementation as before) ...
        bins = config.get('bins', 1)
        min_val, max_val = config.get('min', 0.0), config.get('max', 0.0)
        if not isinstance(bins, int) or bins <= 0: return 0
        if max_val <= min_val: return 0 # Cannot discretize if range is zero or negative
        # Calculate bin size, handle division by zero if bins=1
        bin_size = (max_val - min_val) / max(1, bins) # Divide by bins, not bins-1 for floor logic
        if bin_size <= 1e-9: return 0 # Bin size too small
        # Clip value to the defined range *before* calculating index
        clipped_value = np.clip(value, min_val, max_val)
        # Handle edge case where value is exactly max_val
        if clipped_value >= max_val:
            idx = bins - 1
        else:
            # Calculate index using floor
            idx = int(np.floor((clipped_value - min_val) / bin_size))
        # Final clip to ensure index is within [0, bins-1]
        return np.clip(idx, 0, bins - 1)

    def _validate_and_prepare_state_config(self, config):
        # ... (implementation as before) ...
        validated_config = {}
        required_keys = ['enabled', 'min', 'max', 'bins']
        if not isinstance(config, dict):
             raise ValueError("state_config must be a dictionary.")
        for var, cfg in config.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"State config for '{var}' must be a dictionary.")
            if 'enabled' not in cfg:
                # Default to disabled if 'enabled' key is missing? Or raise error? Let's raise.
                raise ValueError(f"State config for '{var}' missing 'enabled' key.")

            validated_config[var] = cfg # Store config regardless of enabled status

            if cfg['enabled']:
                for key in required_keys:
                    if key not in cfg:
                        raise ValueError(f"Enabled state config for '{var}' missing required key: '{key}'.")
                # Validate types and values for enabled states
                if not isinstance(cfg['min'], (int, float)) or not isinstance(cfg['max'], (int, float)):
                    raise ValueError(f"State config '{var}': 'min' and 'max' must be numeric.")
                if cfg['min'] >= cfg['max']:
                    # Allow min == max only if bins == 1? Let's disallow for now.
                    raise ValueError(f"State config '{var}': 'min' ({cfg['min']}) must be strictly less than 'max' ({cfg['max']}).")
                if not isinstance(cfg['bins'], int) or cfg['bins'] <= 0:
                    raise ValueError(f"State config '{var}': 'bins' must be a positive integer.")
        return validated_config

    def get_discrete_state_indices_tuple(self, agent_state_dict: Dict, gain_variable: str) -> Optional[tuple]:
        # ... (implementation as before) ...
         if gain_variable not in self.ordered_state_vars_for_gain:
             logging.error(f"Gain variable '{gain_variable}' not found in ordered state variables map.")
             return None

         ordered_vars = self.ordered_state_vars_for_gain[gain_variable]
         indices = []
         try:
             for var in ordered_vars:
                 if var not in self.state_config:
                      raise KeyError(f"Configuration for state variable '{var}' (needed for gain '{gain_variable}') not found in agent's state_config.")
                 config = self.state_config[var]
                 if not config['enabled']: # Should not happen if _get_ordered_vars_for_gain is correct
                      logging.warning(f"Trying to discretize disabled variable '{var}' for gain '{gain_variable}'. Skipping.")
                      continue # Or should this be an error?

                 if var not in agent_state_dict:
                     # Provide more context in error message
                     raise KeyError(f"Variable '{var}' required for state discretization of gain '{gain_variable}' not found in provided agent_state_dict. Dict keys: {list(agent_state_dict.keys())}")

                 value = agent_state_dict[var]
                 index = self._discretize_value(value, config)
                 indices.append(index)
             return tuple(indices)
         except KeyError as e:
             logging.error(f"Error discretizing state for gain '{gain_variable}': {e}")
             return None
         except Exception as e:
             logging.error(f"Unexpected error discretizing state for '{gain_variable}': {e}", exc_info=True)
             return None


    # --- select_action (unchanged from Step 1 description) ---
    def select_action(self, agent_state_dict: Dict) -> Dict[str, int]:
        # ... (implementation as before) ...
        actions = {}
        for gain in self.gain_variables:
            action_index = 1 # Default 'maintain'
            if gain in self.q_tables_np: # Check if Q-table exists for this gain
                try:
                    state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                    if state_indices is not None:
                        if np.random.rand() < self.epsilon:
                            # Explore: choose a random action
                            action_index = np.random.randint(self.num_actions)
                        else:
                            # Exploit: choose the best action from Q-table
                            q_values_for_state = self.q_tables_np[gain][state_indices]
                            # Find the index of the maximum Q-value. Handles ties by taking the first max.
                            action_index = np.argmax(q_values_for_state).item() # .item() converts numpy int to python int
                    else:
                        logging.warning(f"Could not get state indices for gain '{gain}' in select_action. Using default action (1). State dict keys: {list(agent_state_dict.keys())}")
                        action_index = 1
                except IndexError as e:
                    # Provide more context if possible
                    q_shape = self.q_tables_np[gain].shape if gain in self.q_tables_np else 'N/A'
                    logging.error(f"IndexError accessing Q-table '{gain}' for indices {state_indices}. Shape: {q_shape}. State dict keys: {list(agent_state_dict.keys())}. Error: {e}.")
                    action_index = 1 # Fallback to default action
                except Exception as e:
                    logging.error(f"Unexpected error selecting action for gain '{gain}': {e}. Using default action (1).", exc_info=True)
                    action_index = 1
            else:
                # If Q-table doesn't exist (e.g., gain disabled), use default action
                # logging.debug(f"Q-table for gain '{gain}' not found (likely disabled). Using default action (1).")
                action_index = 1

            actions[gain] = action_index
        return actions

    # --- learn (MODIFIED to use RewardStrategy) ---
    def learn(self, current_agent_state_dict: Dict, actions_dict: Dict,
              # Reward info can be float (global), tuple (shadow), or dict (echo)
              reward_info: Union[float, Tuple[float, float], Dict[str, float]],
              next_agent_state_dict: Dict, done: bool):
        """Updates Q-tables and visit counts using the injected RewardStrategy."""

        # 1. Parse reward_info to extract components needed by strategies
        interval_reward = 0.0
        avg_w_stab = 1.0 # Default stability
        reward_dict = {} # For echo

        if isinstance(reward_info, dict): # Echo Baseline
            reward_dict = reward_info
            interval_reward = np.mean(list(reward_dict.values())) if reward_dict else 0.0
        elif isinstance(reward_info, tuple): # Shadow Baseline
            interval_reward, avg_w_stab = reward_info
            # Ensure avg_w_stab is valid, default to 1.0 if NaN or invalid
            if not isinstance(avg_w_stab, (float, int, np.number)) or np.isnan(avg_w_stab):
                 avg_w_stab = 1.0
        elif isinstance(reward_info, (float, int, np.number)): # Global Reward
            interval_reward = float(reward_info)
        else:
            logging.error(f"Learn received reward_info of unexpected type: {type(reward_info)}. Skipping learn step.")
            return

        # Clear previous TD errors before new calculation
        self._last_td_errors = {}

        # 2. Iterate through each gain and update its Q-table
        for gain in self.gain_variables:
            if gain not in self.q_tables_np: continue # Skip disabled/uninitialized gains

            try:
                # 3. Get discretized states S and S'
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)

                if current_state_indices is None or next_state_indices is None:
                    logging.warning(f"Could not generate state indices for gain '{gain}' during learn. Skipping update.")
                    continue

                # 4. Get action taken A for this gain
                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not isinstance(action_taken_idx, (int, np.integer)) or not (0 <= action_taken_idx < self.num_actions):
                    logging.warning(f"Invalid action index '{action_taken_idx}' for gain '{gain}' in learn(). Skipping.")
                    continue

                # 5. Calculate reward_for_q_update using the strategy
                #    The strategy might also update baseline tables internally.
                reward_for_q_update = self.reward_strategy.compute_reward_for_learning(
                    gain=gain,
                    interval_reward=interval_reward,
                    avg_w_stab=avg_w_stab,
                    reward_dict=reward_dict,
                    agent_state_dict=current_agent_state_dict, # Pass S dictionary
                    agent=self, # Pass agent instance
                    action_taken_idx=action_taken_idx,
                    current_state_indices=current_state_indices, # Pass S indices
                    actions_dict=actions_dict
                )

                # 6. Q-Learning Update Calculation
                # @formula: Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
                # @formula (if done): Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]

                if done:
                    td_target = reward_for_q_update
                else:
                    next_q_values = self.q_tables_np[gain][next_state_indices]
                    max_next_q = np.max(next_q_values)
                    if np.isnan(max_next_q):
                        logging.warning(f"Max next Q-value is NaN for gain '{gain}', state {next_state_indices}. Using 0 for target calculation.")
                        max_next_q = 0.0
                    td_target = reward_for_q_update + self.discount_factor * max_next_q

                td_error = td_target - current_q
                new_q_value = current_q + self.learning_rate * td_error

                # 7. Update Q-Table and Visit Count
                self.q_tables_np[gain][full_index_current] = new_q_value
                self.visit_counts_np[gain][full_index_current] += 1

                # 8. Store TD error for logging
                self._last_td_errors[gain] = td_error.item() if not np.isnan(td_error) else np.nan # Guardar NaN si td_error es NaN

            except IndexError as e:
                q_shape = self.q_tables_np[gain].shape if gain in self.q_tables_np else 'N/A'
                logging.error(f"IndexError updating NumPy Q/Visit table '{gain}'. Indices: {current_state_indices}, Action: {action_taken_idx}. Shape: {q_shape}. Error: {e}.")
            except KeyError as e:
                logging.error(f"KeyError during learn step for gain '{gain}': {e}. Check state variables or strategy logic.")
            except Exception as e:
                logging.error(f"Unexpected error during learn step for gain '{gain}': {e}.", exc_info=True)

    # --- reset_agent (unchanged) ---
    def reset_agent(self):
        # Update epsilon and learning rate based on decay settings
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.use_learning_rate_decay:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay)
        # Reset last TD errors
        self._last_td_errors = {}
        # logging.debug(f"Agent reset: Epsilon={self.epsilon:.4f}, LR={self.learning_rate:.4f}")


    # --- build_agent_state (mostly unchanged, uses state_config passed in) ---
    def build_agent_state(self, state_vector, controller, state_config_for_build: Dict):
        """ Builds the agent's state representation using enabled variables. """
        # This method now receives the relevant state_config section directly.
        # This avoids relying on self.state_config if a different config scope is needed later.
        agent_state = {}
        state_mapping = {} # Maps standard names to state_vector indices

        # Define mapping based on expected state vector structure
        # Use .get on state_config_for_build to be safer
        if state_config_for_build.get('cart_position', {}).get('enabled'): state_mapping['cart_position'] = 0
        if state_config_for_build.get('cart_velocity', {}).get('enabled'): state_mapping['cart_velocity'] = 1
        if state_config_for_build.get('angle', {}).get('enabled'): state_mapping['angle'] = 2
        if state_config_for_build.get('angular_velocity', {}).get('enabled'): state_mapping['angular_velocity'] = 3

        for var, config in state_config_for_build.items():
            if config.get('enabled', False):
                if var in state_mapping:
                    idx = state_mapping[var]
                    if idx < len(state_vector):
                        agent_state[var] = state_vector[idx]
                    else:
                        raise IndexError(f"State vector index {idx} for '{var}' out of bounds (len={len(state_vector)})")
                elif var == 'kp':
                    agent_state['kp'] = getattr(controller, 'kp', np.nan)
                elif var == 'ki':
                    agent_state['ki'] = getattr(controller, 'ki', np.nan)
                elif var == 'kd':
                    agent_state['kd'] = getattr(controller, 'kd', np.nan)
                else:
                    # This case should ideally not be reached if config is validated
                    logging.warning(f"Enabled state variable '{var}' could not be mapped to state_vector or controller gains.")
                    agent_state[var] = np.nan # Assign NaN if mapping fails

        return agent_state

    # --- get_agent_state_for_saving (unchanged from Step 1 description) ---
    def get_agent_state_for_saving(self) -> Dict:
        # ... (implementation as before - ensures baseline tables are included) ...
        structured_q_tables = {}; structured_visit_counts = {}; structured_baseline_tables = {}
        logging.info("Structuring agent state for JSON saving (Pandas-friendly format)...")

        processed_gains = 0
        for gain in self.gain_variables:
            # Check if Q-table exists and state variables are defined for this gain
            if gain in self.q_tables_np and gain in self.ordered_state_vars_for_gain:
                logging.debug(f"Structuring data for gain '{gain}'...")
                q_table_list = []; visit_count_list = []; baseline_list = []
                ordered_vars = self.ordered_state_vars_for_gain[gain]

                # Get the corresponding NumPy arrays
                np_q_table = self.q_tables_np[gain]
                np_visits = self.visit_counts_np[gain]
                # Baseline table might exist even if not shadow mode (initialized anyway)
                np_baseline = self.baseline_tables_np.get(gain) # Use .get for safety

                state_shape = np_q_table.shape[:-1] # Shape excluding the action dimension
                action_dim = np_q_table.shape[-1]

                # Iterate through all possible discrete state combinations
                for state_indices_tuple in np.ndindex(state_shape):
                    # Create dictionary representing the state {var_name: representative_value}
                    state_dict = {}
                    try:
                        for i, var_name in enumerate(ordered_vars):
                            state_dict[var_name] = self._get_representative_value_from_index(state_indices_tuple[i], var_name)
                    except Exception as e:
                         logging.error(f"Error getting representative value for state {state_indices_tuple}, var index {i} ('{var_name}'): {e}. Skipping state.")
                         continue

                    # --- Q-Table Row ---
                    q_values = np_q_table[state_indices_tuple]
                    q_row = state_dict.copy()
                    for action_idx in range(action_dim):
                        q_row[str(action_idx)] = q_values[action_idx].item() # Use .item() for python float
                    q_table_list.append(q_row)

                    # --- Visit Count Row ---
                    visit_counts = np_visits[state_indices_tuple]
                    visit_row = state_dict.copy()
                    for action_idx in range(action_dim):
                        visit_row[str(action_idx)] = visit_counts[action_idx].item() # Use .item() for python int
                    visit_count_list.append(visit_row)

                    # --- Baseline Table Row (if table exists) ---
                    if np_baseline is not None:
                        baseline_value = np_baseline[state_indices_tuple]
                        baseline_row = state_dict.copy()
                        baseline_row['baseline_value'] = baseline_value.item() # Use .item() for python float
                        baseline_list.append(baseline_row)

                # Store the list of rows for this gain
                structured_q_tables[gain] = q_table_list
                structured_visit_counts[gain] = visit_count_list
                if np_baseline is not None: # Only add baseline if it existed
                    structured_baseline_tables[gain] = baseline_list

                logging.debug(f"Structuring for gain '{gain}' complete. Processed {len(q_table_list)} states.")
                processed_gains += 1
            else:
                # Log if a gain is skipped (e.g., disabled or error during init)
                if gain in self.gain_variables: # Check if it was expected
                     logging.debug(f"Skipping structuring for gain '{gain}'. Q-table or state vars likely not initialized.")

        if processed_gains == 0:
             logging.warning("No gains were processed during agent state structuring. Check agent initialization and config.")

        logging.info("Agent state structuring complete.")
        # Always return all three keys, even if baseline is empty
        return {
            "q_tables": structured_q_tables,
            "visit_counts": structured_visit_counts,
            "baseline_tables": structured_baseline_tables
        }


    # --- NEW Helper methods for logging ---
    def get_q_values_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
         """ Gets Q-values for all actions for the given state for enabled gains. """
         q_values = {}
         for gain in self.gain_variables:
             if gain in self.q_tables_np:
                 state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                 if state_indices is not None:
                     try:
                         q_values[gain] = self.q_tables_np[gain][state_indices]
                     except IndexError:
                          q_values[gain] = np.full(self.num_actions, np.nan) # Return NaNs on error
                 else:
                     q_values[gain] = np.full(self.num_actions, np.nan)
         return q_values

    def get_visit_counts_for_state(self, agent_state_dict: Dict) -> Dict[str, np.ndarray]:
        """ Gets visit counts for all actions for the given state for enabled gains. """
        visit_counts = {}
        for gain in self.gain_variables:
            if gain in self.visit_counts_np:
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        visit_counts[gain] = self.visit_counts_np[gain][state_indices]
                    except IndexError:
                        visit_counts[gain] = np.full(self.num_actions, -1) # Indicate error
                else:
                    visit_counts[gain] = np.full(self.num_actions, -1)
        return visit_counts

    def get_baseline_value_for_state(self, agent_state_dict: Dict) -> Dict[str, float]:
        """ Gets baseline value B(s) for the given state for enabled gains. """
        baselines = {}
        for gain in self.gain_variables:
            if gain in self.baseline_tables_np: # Check if baseline table exists
                state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                if state_indices is not None:
                    try:
                        baselines[gain] = self.baseline_tables_np[gain][state_indices].item()
                    except IndexError:
                        baselines[gain] = np.nan # Return NaN on error
                else:
                    baselines[gain] = np.nan
            else:
                baselines[gain] = np.nan # Return NaN if table doesn't exist
        return baselines

    def get_last_td_errors(self) -> Dict[str, float]:
         """ Returns the TD errors calculated in the most recent learn step. """
         # Return a copy to prevent external modification
         return self._last_td_errors.copy()