import numpy as np
import logging
from interfaces.rl_agent import RLAgent
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Union # Added Union

class PIDQLearningAgent(RLAgent):
    # --- __init__ (unchanged from Step 2) ---
    def __init__(self, state_config, num_actions, gain_step, variable_step,
                 discount_factor=0.98, epsilon=1.0, epsilon_min=0.1,
                 epsilon_decay=0.99954, learning_rate=1.0, learning_rate_min=0.01,
                 learning_rate_decay=0.999425, use_epsilon_decay=True, use_learning_rate_decay=True,
                 reward_mode='global', shadow_baseline_params: Optional[Dict] = None,
                 q_init_value=0.0, visit_init_value=0):

        logging.info("Initializing PIDQLearningAgent...")
        self.state_config = self._validate_and_prepare_state_config(state_config)
        self.num_actions = num_actions
        self.q_init_value = q_init_value
        self.visit_init_value = visit_init_value
        self.float_precision_for_saved_keys = 6

        self.gain_step = gain_step
        self.variable_step = variable_step
        self.discount_factor = discount_factor
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.learning_rate, self.learning_rate_min, self.learning_rate_decay = learning_rate, learning_rate_min, learning_rate_decay
        self.use_epsilon_decay, self.use_learning_rate_decay = use_epsilon_decay, use_learning_rate_decay

        self.reward_mode = reward_mode
        self.shadow_params = shadow_baseline_params if shadow_baseline_params else {}
        self.beta = self.shadow_params.get('beta', 0.1)
        self.baseline_init_value = self.shadow_params.get('baseline_init_value', 0.0)
        logging.info(f"Agent reward_mode: {self.reward_mode}")
        if self.reward_mode == 'shadow-baseline':
            logging.info(f"Shadow Baseline params: beta={self.beta}, init_value={self.baseline_init_value}")

        self.ordered_state_vars_for_gain = {}
        self.gain_variables = ['kp', 'ki', 'kd']

        self.q_tables_np = {}
        self.visit_counts_np = {}
        self.baseline_tables_np = {}

        for gain in self.gain_variables:
            if self.state_config[gain]['enabled']:
                logging.info(f"Pre-initializing NumPy tables for gain '{gain}'...")
                ordered_vars_list = self._get_ordered_vars_for_gain(gain)
                self.ordered_state_vars_for_gain[gain] = ordered_vars_list
                try:
                    state_dims = [self.state_config[var]['bins'] for var in ordered_vars_list]
                    q_visit_shape = tuple(state_dims + [self.num_actions])
                    baseline_shape = tuple(state_dims)
                    logging.debug(f"  - State variable order for '{gain}': {ordered_vars_list}")
                    logging.debug(f"  - Q/Visit Array shape for '{gain}': {q_visit_shape}")
                    logging.debug(f"  - Baseline Array shape for '{gain}': {baseline_shape}")
                    self.q_tables_np[gain] = np.full(q_visit_shape, self.q_init_value, dtype=np.float32)
                    self.visit_counts_np[gain] = np.full(q_visit_shape, self.visit_init_value, dtype=np.int32)
                    if self.reward_mode == 'shadow-baseline':
                        self.baseline_tables_np[gain] = np.full(baseline_shape, self.baseline_init_value, dtype=np.float32)
                except KeyError as e:
                     logging.error(f"Error getting bin count for variable {e} while creating array for gain '{gain}'. Skipping initialization.")
                     # Clean up partial initialization
                     if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
                     if gain in self.q_tables_np: del self.q_tables_np[gain]
                     if gain in self.visit_counts_np: del self.visit_counts_np[gain]
                     if gain in self.baseline_tables_np: del self.baseline_tables_np[gain]
                     continue
            else:
                 logging.debug(f"Skipping NumPy array creation for disabled gain: {gain}")
        logging.info("PIDQLearningAgent initialized successfully.")

    # --- Helper methods (_get_ordered_vars_for_gain, _get_representative_value_from_index, _discretize_value, _validate_and_prepare_state_config, get_discrete_state_indices_tuple) ---
    # --- (unchanged from Step 2) ---
    def _get_ordered_vars_for_gain(self, gain):
        ordered_vars = OrderedDict()
        for var, cfg in self.state_config.items():
            if cfg['enabled'] and var not in self.gain_variables:
                ordered_vars[var] = True
        if self.state_config[gain]['enabled']:
             ordered_vars[gain] = True
        return list(ordered_vars.keys())
    def _get_representative_value_from_index(self, index, var_name):
        if var_name not in self.state_config or not self.state_config[var_name]['enabled']: return 0.0
        config = self.state_config[var_name]; min_val, max_val, bins = config['min'], config['max'], config['bins']
        if not isinstance(bins, int) or bins <= 0: return min_val
        if min_val >= max_val: return min_val
        if index < 0 or index >= bins: return min_val
        if bins == 1: value = (min_val + max_val) / 2.0
        else: step = (max_val - min_val) / (bins - 1); value = min_val + index * step
        return float(np.clip(value, min_val, max_val))
    def _discretize_value(self, value, config):
        bins = config['bins']; min_val, max_val = config['min'], config['max']
        if max_val <= min_val or bins <= 0: return 0
        bin_size = (max_val - min_val) / bins
        if bin_size <= 1e-9: return 0
        clipped_value = np.clip(value, min_val, max_val)
        if clipped_value >= max_val: idx = bins - 1
        else: idx = int(np.floor((clipped_value - min_val) / bin_size))
        return np.clip(idx, 0, bins - 1)
    def _validate_and_prepare_state_config(self, config):
        validated_config = {}; required_keys = ['enabled', 'min', 'max', 'bins']
        for var, cfg in config.items():
            if not isinstance(cfg, dict): raise ValueError(f"State config for '{var}' must be a dictionary.")
            if 'enabled' not in cfg: raise ValueError(f"State config for '{var}' missing 'enabled' key.")
            if cfg['enabled']:
                 for key in required_keys:
                     if key not in cfg: raise ValueError(f"Enabled state config for '{var}' missing '{key}' key.")
                 if cfg['min'] >= cfg['max']: raise ValueError(f"State config '{var}': 'min' >= 'max'.")
                 if not isinstance(cfg['bins'], int) or cfg['bins'] <= 0: raise ValueError(f"State config '{var}': 'bins' must be positive int.")
            validated_config[var] = cfg
        return validated_config
    def get_discrete_state_indices_tuple(self, agent_state_dict, gain_variable):
        if gain_variable not in self.ordered_state_vars_for_gain: return None
        ordered_vars = self.ordered_state_vars_for_gain[gain_variable]; indices = []
        try:
            for var in ordered_vars:
                config = self.state_config[var]
                if var not in agent_state_dict: raise KeyError(f"Variable '{var}' not in agent state: {agent_state_dict}")
                value = agent_state_dict[var]; index = self._discretize_value(value, config); indices.append(index)
            return tuple(indices)
        except KeyError as e: logging.error(f"Error discretizing state for gain '{gain_variable}': {e}"); return None
        except Exception as e: logging.error(f"Unexpected error discretizing state for '{gain_variable}': {e}", exc_info=True); return None


    # --- select_action (unchanged from Step 2) ---
    def select_action(self, agent_state_dict):
        actions = {}
        for gain in self.gain_variables:
            action_index = 1 # Default 'maintain'
            if gain in self.q_tables_np:
                try:
                    state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)
                    if state_indices is not None:
                        if np.random.rand() < self.epsilon: action_index = np.random.randint(self.num_actions)
                        else: q_values_for_state = self.q_tables_np[gain][state_indices]; action_index = np.argmax(q_values_for_state).item()
                    else: logging.warning(f"Could not get state indices for '{gain}' in select_action. Using default.")
                except IndexError as e: logging.error(f"IndexError accessing Q-table '{gain}' indices {state_indices}. Shape: {self.q_tables_np[gain].shape}. Error: {e}.")
                except Exception as e: logging.error(f"Unexpected error selecting action for '{gain}': {e}.", exc_info=True)
            actions[gain] = action_index
        return actions

    # --- learn (MODIFIED for Echo Baseline dictionary reward) ---
    def learn(self, current_agent_state_dict: Dict, actions_dict: Dict,
              # Reward can be float (global), tuple (shadow), or dict (echo)
              reward: Union[float, Tuple[float, float], Dict[str, float]],
              next_agent_state_dict: Dict, done: bool):
        """Updates Q-tables, visit counts, and optionally baseline tables based on reward mode."""

        # Determine reward mode based on the type of 'reward' received
        local_reward_mode = 'global' # Default
        interval_reward = 0.0
        avg_w_stab = 1.0
        reward_dict = {}

        if isinstance(reward, dict):
            local_reward_mode = 'echo-baseline'
            reward_dict = reward
            # For logging/consistency, we might estimate a single interval reward if needed,
            # e.g., by averaging the dict values, but it's not used for learning.
            # interval_reward = np.mean(list(reward_dict.values())) if reward_dict else 0.0
        elif isinstance(reward, tuple):
            local_reward_mode = 'shadow-baseline'
            interval_reward, avg_w_stab = reward
        elif isinstance(reward, (float, int, np.number)): # Handles numpy floats etc.
            local_reward_mode = 'global'
            interval_reward = float(reward)
        else:
            logging.error(f"Learn received reward of unexpected type: {type(reward)}. Skipping learn step.")
            return

        # Verify consistency if agent has its own reward_mode setting
        if hasattr(self, 'reward_mode') and self.reward_mode != local_reward_mode:
             logging.warning(f"Agent reward_mode '{self.reward_mode}' mismatches detected mode '{local_reward_mode}' based on reward type. Proceeding with detected mode.")


        for gain in self.gain_variables:
            if gain not in self.q_tables_np: continue # Skip disabled/uninitialized gains

            is_shadow_mode_and_valid = (local_reward_mode == 'shadow-baseline' and gain in self.baseline_tables_np)

            try:
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)

                if current_state_indices is None or next_state_indices is None:
                     logging.warning(f"Could not generate state indices for gain '{gain}' during learn. Skipping update.")
                     continue

                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not isinstance(action_taken_idx, (int, np.integer)) or not (0 <= action_taken_idx < self.num_actions):
                     logging.warning(f"Invalid action index '{action_taken_idx}' for gain '{gain}' in learn(). Skipping.")
                     continue

                # --- Determine Reward for Q-Update & Update Baseline (if Shadow) ---
                reward_for_q_update = 0.0 # Initialize

                if local_reward_mode == 'global':
                    reward_for_q_update = interval_reward
                elif local_reward_mode == 'echo-baseline':
                    reward_for_q_update = reward_dict.get(gain, 0.0) # Get specific reward for this gain
                    if gain not in reward_dict:
                         logging.warning(f"Gain '{gain}' not found in echo-baseline reward dictionary: {reward_dict}. Using 0 reward.")
                elif local_reward_mode == 'shadow-baseline':
                    # Shadow logic remains the same as Step 2
                    baseline_value = self.baseline_tables_np[gain][current_state_indices]
                    if action_taken_idx == 1: # Action 'maintain'
                        delta_B = self.beta * avg_w_stab * (interval_reward - baseline_value)
                        self.baseline_tables_np[gain][current_state_indices] = baseline_value + delta_B
                        reward_for_q_update = interval_reward
                    else: # Action 'decrease' or 'increase'
                        reward_for_q_update = interval_reward - baseline_value

                # --- Q-Learning Update (uses reward_for_q_update determined above) ---
                if done:
                    td_target = reward_for_q_update
                else:
                    next_q_values = self.q_tables_np[gain][next_state_indices]
                    max_next_q = np.max(next_q_values)
                    td_target = reward_for_q_update + self.discount_factor * max_next_q

                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]
                td_error = td_target - current_q
                new_q_value = current_q + self.learning_rate * td_error
                self.q_tables_np[gain][full_index_current] = new_q_value

                # --- Visit Count Update ---
                self.visit_counts_np[gain][full_index_current] += 1

            except IndexError as e:
                 table_name = "Q/Visit"
                 if is_shadow_mode_and_valid and 'baseline_value' in locals(): table_name="Baseline"
                 logging.error(f"IndexError updating NumPy {table_name} table '{gain}'. Indices: {current_state_indices}, Action: {action_taken_idx}. Error: {e}.")
            except KeyError as e:
                 logging.error(f"KeyError during learn step for gain '{gain}': {e}. This might indicate missing state variables or reward dict keys.")
            except Exception as e:
                 logging.error(f"Unexpected error during learn step for gain '{gain}': {e}.", exc_info=True)

    # --- reset_agent (unchanged) ---
    def reset_agent(self):
        if self.use_epsilon_decay: self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.use_learning_rate_decay: self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay)

    # --- build_agent_state (unchanged) ---
    def build_agent_state(self, state_vector, controller, state_config_unused):
        # (Code unchanged from Step 2)
        agent_state = {}; state_mapping = {}
        config_keys = list(self.state_config.keys())
        cart_pos_key=next((k for k in config_keys if 'cart_position' in k),None); cart_vel_key=next((k for k in config_keys if 'cart_velocity' in k),None); angle_key=next((k for k in config_keys if 'angle' in k),None); ang_vel_key=next((k for k in config_keys if 'velocity' in k and ('angle' in k or 'pendulum' in k)),None)
        if cart_pos_key: 
            state_mapping[cart_pos_key]=0
        if cart_vel_key: 
            state_mapping[cart_vel_key]=1
        if angle_key: 
            state_mapping[angle_key]=2
        if ang_vel_key: 
            state_mapping[ang_vel_key]=3
        for var, config in self.state_config.items():
            if config['enabled']:
                if var in state_mapping:
                    idx=state_mapping[var];
                    if idx < len(state_vector): agent_state[var] = state_vector[idx]
                    else: raise IndexError(f"State vector index {idx} for '{var}' out of bounds (len={len(state_vector)})")
                elif var == 'kp': agent_state['kp'] = controller.kp
                elif var == 'ki': agent_state['ki'] = controller.ki
                elif var == 'kd': agent_state['kd'] = controller.kd
                else: raise KeyError(f"Enabled state variable '{var}' could not be mapped.")
        return agent_state


    # --- get_agent_state_for_saving (unchanged from Step 2) ---
    def get_agent_state_for_saving(self):
        # (Code unchanged from Step 2 - already saves Q, Visit, Baseline)
        structured_q_tables = {}; structured_visit_counts = {}; structured_baseline_tables = {}
        logging.info("Structuring agent state for JSON saving (Pandas-friendly format)...")
        for gain in self.gain_variables:
            if gain in self.q_tables_np and gain in self.ordered_state_vars_for_gain:
                logging.debug(f"Structuring data for gain '{gain}'...")
                q_table_list = []; visit_count_list = []; baseline_list = []
                ordered_vars = self.ordered_state_vars_for_gain[gain]
                np_q_table = self.q_tables_np[gain]; np_visits = self.visit_counts_np[gain]
                np_baseline = self.baseline_tables_np.get(gain)
                state_shape = np_q_table.shape[:-1]; action_dim = np_q_table.shape[-1]
                for state_indices_tuple in np.ndindex(state_shape):
                    state_dict = {}
                    for i, var_name in enumerate(ordered_vars): state_dict[var_name] = self._get_representative_value_from_index(state_indices_tuple[i], var_name)
                    q_values = np_q_table[state_indices_tuple]; q_row = state_dict.copy()
                    for action_idx in range(action_dim): q_row[str(action_idx)] = q_values[action_idx].item()
                    q_table_list.append(q_row)
                    visit_counts = np_visits[state_indices_tuple]; visit_row = state_dict.copy()
                    for action_idx in range(action_dim): visit_row[str(action_idx)] = visit_counts[action_idx].item()
                    visit_count_list.append(visit_row)
                    if np_baseline is not None:
                        baseline_value = np_baseline[state_indices_tuple]; baseline_row = state_dict.copy()
                        baseline_row['baseline_value'] = baseline_value.item(); baseline_list.append(baseline_row)
                structured_q_tables[gain] = q_table_list; structured_visit_counts[gain] = visit_count_list
                if np_baseline is not None: structured_baseline_tables[gain] = baseline_list
                logging.debug(f"Structuring for gain '{gain}' complete.")
            else: logging.debug(f"Skipping structuring for gain '{gain}'.")
        logging.info("Agent state structuring complete.")
        return {"q_tables": structured_q_tables, "visit_counts": structured_visit_counts, "baseline_tables": structured_baseline_tables}