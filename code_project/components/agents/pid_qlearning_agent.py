import numpy as np
import logging
from interfaces.rl_agent import RLAgent
from collections import OrderedDict

class PIDQLearningAgent(RLAgent):
    def __init__(self, state_config, num_actions, gain_step, variable_step, discount_factor=0.98, epsilon=1.0, epsilon_min=0.1,
                 epsilon_decay=0.99954, learning_rate=1.0, learning_rate_min=0.01,
                 learning_rate_decay=0.999425, use_epsilon_decay=True, use_learning_rate_decay=True,
                 q_init_value=0.0, visit_init_value=0): # Initial values for NumPy arrays

        logging.info("Initializing PIDQLearningAgent with NumPy Array internal structure...")
        self.state_config = self._validate_and_prepare_state_config(state_config)
        self.num_actions = num_actions
        self.q_init_value = q_init_value
        self.visit_init_value = visit_init_value
        # Define float precision for string keys used ONLY during saving
        self.float_precision_for_saved_keys = 6

        self.gain_step = gain_step
        self.variable_step = variable_step
        self.discount_factor = discount_factor
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.learning_rate, self.learning_rate_min, self.learning_rate_decay = learning_rate, learning_rate_min, learning_rate_decay
        self.use_epsilon_decay, self.use_learning_rate_decay = use_epsilon_decay, use_learning_rate_decay

        # Define the exact order of state variables for array dimensions
        self.ordered_state_vars_for_gain = {} # Dict mapping gain -> list of ordered state vars
        self.gain_variables = ['kp', 'ki', 'kd']

        # Initialize Q-tables and Visit Counts as NumPy arrays
        self.q_tables_np = {}       # Stores NumPy arrays for Q-values
        self.visit_counts_np = {}   # Stores NumPy arrays for visit counts

        # --- Pre-Initialization Logic with NumPy Arrays ---
        for gain in self.gain_variables:
            if self.state_config[gain]['enabled']:
                logging.info(f"Pre-initializing NumPy Q-table and Visit Count arrays for gain '{gain}'...")

                ordered_vars_list = self._get_ordered_vars_for_gain(gain)
                self.ordered_state_vars_for_gain[gain] = ordered_vars_list

                # Determine the shape of the NumPy arrays
                # Shape = (bins_var1, bins_var2, ..., bins_varN, num_actions)
                try:
                    state_dims = [self.state_config[var]['bins'] for var in ordered_vars_list]
                    array_shape = tuple(state_dims + [self.num_actions])
                    logging.debug(f"  - State variable order for '{gain}': {ordered_vars_list}")
                    logging.debug(f"  - Array shape for '{gain}': {array_shape}")

                    # Initialize arrays
                    self.q_tables_np[gain] = np.full(array_shape, self.q_init_value, dtype=np.float32)
                    self.visit_counts_np[gain] = np.full(array_shape, self.visit_init_value, dtype=np.int32) # Use int for counts

                except KeyError as e:
                     logging.error(f"Error getting bin count for variable {e} while creating array for gain '{gain}'. Skipping initialization for this gain.")
                     # Remove potentially partially created entries
                     if gain in self.ordered_state_vars_for_gain: del self.ordered_state_vars_for_gain[gain]
                     if gain in self.q_tables_np: del self.q_tables_np[gain]
                     if gain in self.visit_counts_np: del self.visit_counts_np[gain]
                     continue # Skip to next gain
            else:
                 logging.debug(f"Skipping NumPy array creation for disabled gain: {gain}")

        logging.info("PIDQLearningAgent initialized successfully with NumPy Array tables.")

    # --- Helper: Get Ordered Variables (Unchanged from previous NumPy approach) ---
    def _get_ordered_vars_for_gain(self, gain):
        """Determines the ordered list of state variables for a given gain's table dimensions."""
        ordered_vars = OrderedDict()
        # 1. Add non-gain enabled variables first
        for var, cfg in self.state_config.items():
            if cfg['enabled'] and var not in self.gain_variables:
                ordered_vars[var] = True # Just need the var name in order
        # 2. Add the gain variable itself
        if self.state_config[gain]['enabled']:
             ordered_vars[gain] = True
        return list(ordered_vars.keys())

    # --- Helper: Get Representative Value from Index (Needed ONLY for Saving Transformation) ---
    def _get_representative_value_from_index(self, index, var_name):
        """
        Calculates a representative value for a given bin index.
        For bins > 1, the values span from min_val to max_val inclusive.
        For bins = 1, it returns the midpoint.
        """
        if var_name not in self.state_config or not self.state_config[var_name]['enabled']:
            logging.error(f"Configuración no encontrada o no habilitada para la variable '{var_name}' al calcular valor representativo.")
            return 0.0 # Fallback

        config = self.state_config[var_name]
        min_val, max_val, bins = config['min'], config['max'], config['bins']

        # Validaciones básicas
        if not isinstance(bins, int) or bins <= 0:
             logging.error(f"Número de bins inválido ({bins}) para la variable '{var_name}'.")
             return min_val # Fallback
        if min_val >= max_val:
            logging.warning(f"min_val ({min_val}) >= max_val ({max_val}) for variable '{var_name}'. Returning min_val.")
            return min_val
        if index < 0 or index >= bins:
             logging.error(f"Índice {index} fuera de rango [0, {bins-1}] para variable '{var_name}'.")
             # Podríamos retornar min o max dependiendo del lado, pero min es más seguro.
             return min_val # Fallback

        if bins == 1:
            # Si solo hay un bin -> punto medio.
            value = (min_val + max_val) / 2.0
        else:
            # Para bins > 1, calculamos el paso para que los puntos cubran el rango [min, max]
            # Hay 'bins - 1' intervalos entre 'bins' puntos.
            step = (max_val - min_val) / (bins - 1)
            value = min_val + index * step

        # Redondeo opcional para limpieza en JSON (ajusta decimales si es necesario)
        # value = round(value, self.float_precision_for_saved_keys) # Puedes descomentar y ajustar precisión

        # Asegurar tipo float estándar de Python para JSON y aplicar clip final por seguridad
        # (aunque con el cálculo correcto y el índice válido, no debería salirse)
        return float(np.clip(value, min_val, max_val))

    # --- Helper: Discretize Value to Index (Crucial for NumPy access) ---
    def _discretize_value(self, value, config):
        """Discretizes a single value based on its configuration. Returns the bin index (int)."""
        # (Implementation unchanged from previous versions - ensure it returns int)
        bins = config['bins']
        min_val, max_val = config['min'], config['max']

        if max_val <= min_val or bins <= 0: return 0
        bin_size = (max_val - min_val) / bins
        if bin_size <= 1e-9: return 0 # Avoid division by zero for very small ranges or bins=1

        clipped_value = np.clip(value, min_val, max_val)

        # Handle the edge case where value is exactly max_val
        if clipped_value >= max_val:
            # Ensure floating point comparison is robust if needed, but direct >= might suffice
             idx = bins - 1
        else:
             # Use floor division after scaling
             idx = int(np.floor((clipped_value - min_val) / bin_size))

        # Final clip to ensure index is within [0, bins-1]
        return np.clip(idx, 0, bins - 1) # Return as standard Python int


    # --- Helper: Validate State Config ---
    def _validate_and_prepare_state_config(self, config):
        validated_config = {}
        required_keys = ['enabled', 'min', 'max', 'bins']
        for var, cfg in config.items():
            if not isinstance(cfg, dict): raise ValueError(f"State config for '{var}' must be a dictionary.")
            if 'enabled' not in cfg: raise ValueError(f"State config for '{var}' missing 'enabled' key.")
            if cfg['enabled']:
                 for key in required_keys:
                     if key not in cfg: raise ValueError(f"Enabled state config for '{var}' missing '{key}' key.")
                 if cfg['min'] >= cfg['max']: raise ValueError(f"State config for '{var}': 'min' ({cfg['min']}) must be less than 'max' ({cfg['max']}).")
                 if not isinstance(cfg['bins'], int) or cfg['bins'] <= 0: raise ValueError(f"State config for '{var}': 'bins' ({cfg['bins']}) must be a positive integer.")
            validated_config[var] = cfg
        return validated_config

    # --- NEW Helper: Get State Index Tuple (Integer Indices for NumPy) ---
    def get_discrete_state_indices_tuple(self, agent_state_dict, gain_variable):
        """
        Generates the TUPLE of integer indices representing the discretized state,
        in the specific order required for NumPy array indexing for the given gain.
        Returns None if the gain is disabled or an error occurs.
        """
        if gain_variable not in self.ordered_state_vars_for_gain:
             if gain_variable in self.state_config and not self.state_config[gain_variable]['enabled']:
                 # This is expected for disabled gains, do nothing specific here, caller should handle
                 pass
             else:
                  logging.error(f"Order of state variables not defined for gain '{gain_variable}'.")
             return None # Indicate failure or disabled gain

        ordered_vars = self.ordered_state_vars_for_gain[gain_variable]
        indices = []
        try:
            for var in ordered_vars:
                config = self.state_config[var] # Assumes var is always in state_config if in ordered_vars
                if var not in agent_state_dict:
                     # This is a critical error if the variable should be present
                     raise KeyError(f"Variable '{var}' required for discretization not found in agent state dictionary: {agent_state_dict}")

                value = agent_state_dict[var]
                # Discretize the value to get the integer index
                index = self._discretize_value(value, config)
                indices.append(index)
            return tuple(indices) # Return tuple of integer indices
        except KeyError as e:
             logging.error(f"Error generating state indices tuple for gain '{gain_variable}': {e}")
             return None
        except Exception as e:
             logging.error(f"Unexpected error generating state indices tuple for gain '{gain_variable}': {e}", exc_info=True)
             return None


    # --- Core Agent Methods using NumPy Arrays ---

    def select_action(self, agent_state_dict):
        """Selects actions using the NumPy Q-tables."""
        actions = {} # Stores chosen action INDEX {gain: action_idx}
        for gain in self.gain_variables:
            # Default action if gain disabled or error occurs
            action_index = 1 # Default to 'maintain' action index

            if gain in self.q_tables_np: # Check if gain is enabled and initialized
                try:
                    # Get the tuple of integer state indices
                    state_indices = self.get_discrete_state_indices_tuple(agent_state_dict, gain)

                    if state_indices is not None: # If indices obtained successfully
                        if np.random.rand() < self.epsilon:
                            # Explore: choose random action index
                            action_index = np.random.randint(self.num_actions)
                        else:
                            # Exploit: Get Q-values for this state from NumPy array
                            q_values_for_state = self.q_tables_np[gain][state_indices]
                            # Find the action index with the maximum Q-value
                            action_index = np.argmax(q_values_for_state).item() # .item() converts numpy int to python int
                            # logging.debug(f"Gain {gain}: State Indices {state_indices}, Q-vals {q_values_for_state}, Chosen Action Index {action_index}")
                    else:
                         # Error getting indices, keep default action
                         logging.warning(f"Could not get state indices for gain '{gain}' in select_action. Using default action {action_index}.")

                except IndexError as e:
                     logging.error(f"IndexError accessing NumPy Q-table for gain '{gain}' with indices {state_indices}. Shape: {self.q_tables_np[gain].shape}. Error: {e}. Using default action.")
                     # Keep default action_index = 1
                except Exception as e:
                     logging.error(f"Unexpected error during action selection for gain '{gain}': {e}. Using default action.", exc_info=True)
                     # Keep default action_index = 1

            actions[gain] = action_index # Store the chosen action *index*

        return actions # Return dictionary mapping gain -> action *index*

    def learn(self, current_agent_state_dict, actions_dict, reward, next_agent_state_dict, done):
        """Updates NumPy Q-tables and visit counts."""
        for gain in self.gain_variables:
            if gain not in self.q_tables_np: continue # Skip disabled/uninitialized gains

            try:
                # Get integer index tuples for current and next states
                current_state_indices = self.get_discrete_state_indices_tuple(current_agent_state_dict, gain)
                next_state_indices = self.get_discrete_state_indices_tuple(next_agent_state_dict, gain)

                if current_state_indices is None or next_state_indices is None:
                     logging.warning(f"Could not generate state indices for gain '{gain}' during learn. Skipping update.")
                     continue

                # Get the action *index* taken for this gain
                action_taken_idx = actions_dict.get(gain)
                if action_taken_idx is None or not isinstance(action_taken_idx, (int, np.integer)) or not (0 <= action_taken_idx < self.num_actions):
                     logging.warning(f"Invalid or missing action index '{action_taken_idx}' for gain '{gain}' during learn(). Skipping update.")
                     continue

                # --- Q-Learning Update using NumPy indexing ---

                # Target value calculation
                if done:
                    td_target = reward
                else:
                    # Max Q-value over actions in the next state
                    next_q_values = self.q_tables_np[gain][next_state_indices] # Slice yields 1D array of action Q-values
                    max_next_q = np.max(next_q_values)
                    td_target = reward + self.discount_factor * max_next_q

                # Get the current Q-value for the state-action pair taken
                # Combine state indices tuple with action index for full coordinate
                full_index_current = current_state_indices + (action_taken_idx,)
                current_q = self.q_tables_np[gain][full_index_current]

                # TD Error and Q-value update
                td_error = td_target - current_q
                new_q_value = current_q + self.learning_rate * td_error

                # Update the Q-table
                self.q_tables_np[gain][full_index_current] = new_q_value

                # --- Visit Count Update ---
                self.visit_counts_np[gain][full_index_current] += 1
                # --- End Updates ---

            except IndexError as e:
                 logging.error(f"IndexError updating NumPy tables for gain '{gain}'. Current indices: {current_state_indices}, Action: {action_taken_idx}, Next indices: {next_state_indices}. Error: {e}. Skipping update.")
            except Exception as e:
                 logging.error(f"Unexpected error during learn step for gain '{gain}': {e}. Skipping update.", exc_info=True)


    def reset_agent(self):
        """Called at the start of each episode to update exploration/learning rates."""
        # (No changes needed)
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.use_learning_rate_decay:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay)


    # --- Recursive Helper for Saving Transformation ---
    def _build_nested_dict_from_np(self, np_array_slice, ordered_vars, current_depth=0):
        """
        Recursively builds a nested dictionary from a slice of the NumPy Q-table or visit count array.
        Uses representative value strings as keys.
        """
        num_state_vars = len(ordered_vars)

        # Base case: reached the leaf node representing actions for a specific state
        if current_depth == num_state_vars:
            # np_array_slice should now be a 1D array of size num_actions
            if np_array_slice.ndim == 1 and len(np_array_slice) == self.num_actions:
                 # Create the action dictionary mapping action *strings* to values
                 # Convert numpy types (e.g., np.float32) to standard python types for JSON
                return {str(action_idx): value.item() for action_idx, value in enumerate(np_array_slice)}
            else:
                 # This shouldn't happen if indexing is correct
                 logging.error(f"Reached base case in _build_nested_dict_from_np, but array slice has unexpected shape: {np_array_slice.shape}. Expected ({self.num_actions},).")
                 return {} # Return empty dict on error


        # Recursive step: iterate through bins of the current state variable
        var_name = ordered_vars[current_depth]
        try:
            num_bins = self.state_config[var_name]['bins']
        except KeyError:
             logging.error(f"Variable '{var_name}' not found in state_config during saving transformation.")
             return {} # Cannot proceed without bin info

        current_level_dict = {}
        for bin_index in range(num_bins):
            # Calculate the representative value string key for this bin
            repr_value = self._get_representative_value_from_index(bin_index, var_name)
            string_key = f"{repr_value:.{self.float_precision_for_saved_keys}f}"

            # Recursively call for the next depth, passing the corresponding slice of the array
            try:
                # Slice the array along the current dimension (current_depth)
                next_slice = np_array_slice[bin_index]
                current_level_dict[string_key] = self._build_nested_dict_from_np(
                    next_slice, ordered_vars, current_depth + 1
                )
            except IndexError as e:
                 logging.error(f"IndexError during recursive slicing in _build_nested_dict_from_np. Depth: {current_depth}, Var: {var_name}, Index: {bin_index}. Slice shape: {np_array_slice.shape}. Error: {e}")
                 current_level_dict[string_key] = {} # Add empty dict for this key on error
            except Exception as e:
                 logging.error(f"Unexpected error during recursive call in _build_nested_dict_from_np. Depth: {current_depth}, Var: {var_name}, Index: {bin_index}. Error: {e}", exc_info=True)
                 current_level_dict[string_key] = {}

        return current_level_dict

    # --- Build Agent State (Crucial Input for Indexing) ---
    def build_agent_state(self, state_vector, controller, state_config_unused):
        """
        Constructs the dictionary representation of the agent's current state
        containing the continuous values needed for discretization.
        """
        # El argumento state_config_unused no se usa directamente aquí,
        # pero se mantiene por coherencia si la interfaz lo espera.
        # Usamos self.state_config internamente.
        agent_state = {}
        # Map state vector indices to names used in state_config
        state_mapping = {}
        config_keys = list(self.state_config.keys())
        # Determinar nombres correctos usados en config dinámicamente
        # NOTA: Ajusta estas búsquedas si los nombres en tu config son diferentes
        #       (e.g., si usas 'angle' en lugar de 'pendulum_angle')
        cart_pos_key = next((k for k in config_keys if 'cart_position' in k), None)
        cart_vel_key = next((k for k in config_keys if 'cart_velocity' in k), None)
        angle_key = next((k for k in config_keys if 'angle' in k), None) # Coincide con 'pendulum_angle' o 'angle'
        ang_vel_key = next((k for k in config_keys if 'velocity' in k and ('angle' in k or 'pendulum' in k)), None) # Coincide con 'pendulum_velocity' o 'angular_velocity'

        # Mapeo basado en índices del vector de estado estándar del CartPole
        if cart_pos_key: state_mapping[cart_pos_key] = 0
        if cart_vel_key: state_mapping[cart_vel_key] = 1
        if angle_key: state_mapping[angle_key] = 2
        if ang_vel_key: state_mapping[ang_vel_key] = 3

        for var, config in self.state_config.items():
             if config['enabled']:
                 if var in state_mapping:
                     idx = state_mapping[var]
                     if idx < len(state_vector):
                          agent_state[var] = state_vector[idx]
                     else:
                          # Error crítico: el índice esperado no existe en el vector de estado
                          logging.error(f"State vector index {idx} for '{var}' out of bounds (len={len(state_vector)}). Cannot build state for this variable.")
                          raise IndexError(f"State vector index {idx} for '{var}' out of bounds (len={len(state_vector)})")
                 elif var == 'kp': agent_state['kp'] = controller.kp
                 elif var == 'ki': agent_state['ki'] = controller.ki
                 elif var == 'kd': agent_state['kd'] = controller.kd
                 else:
                      # Error crítico: Variable habilitada pero no mapeada
                      logging.error(f"Enabled state variable '{var}' from config is not found in state_vector mapping or controller gains. Agent state cannot be fully constructed.")
                      raise KeyError(f"Enabled state variable '{var}' could not be mapped to a value.")


        # Verificación final opcional (puede ayudar en debugging)
        # Asegura que todas las variables necesarias para las ganancias activas estén presentes
        # for gain, ordered_vars in self.ordered_state_vars_for_gain.items():
        #      if gain in self.q_tables_np: # Solo verificar para ganancias activas
        #           for var in ordered_vars:
        #               if var not in agent_state:
        #                    logging.critical(f"CRITICAL: Constructed agent_state is missing required variable '{var}' for active gain '{gain}'. State: {agent_state}")
        #                    raise RuntimeError(f"Constructed agent_state is missing required variable '{var}' for active gain '{gain}'.")


        return agent_state
    
    # --- Saving Method (Performs Transformation) ---
    def get_agent_state_for_saving(self):
        """
        Transforms the internal NumPy arrays into a structure suitable for
        saving to JSON and easy loading into Pandas (list of dictionaries).

        Returns:
            dict: Contains 'q_tables' and 'visit_counts'. Each is a dict
                  mapping gain name to a list of dictionaries. Each inner
                  dictionary represents a state and its corresponding action values.
                  Example: {'q_tables': {'kp': [ {'state_var1': val, 'state_var2': val, '0': q0, '1': q1, '2': q2}, ... ]}}
        """
        structured_q_tables = {}
        structured_visit_counts = {}
        logging.info("Structuring agent state for JSON saving (Pandas-friendly format)...")

        for gain in self.gain_variables:
            if gain in self.q_tables_np and gain in self.ordered_state_vars_for_gain:
                logging.debug(f"Structuring data for gain '{gain}'...")
                q_table_list = []
                visit_count_list = []
                ordered_vars = self.ordered_state_vars_for_gain[gain]
                np_q_table = self.q_tables_np[gain]
                np_visits = self.visit_counts_np[gain]

                # Iterar a través de todas las combinaciones de índices de estado
                # np.ndindex itera sobre los índices de un array N-dimensional
                state_shape = np_q_table.shape[:-1] # Dimensiones de los estados
                action_dim = np_q_table.shape[-1]   # Dimensión de las acciones

                for state_indices_tuple in np.ndindex(state_shape):
                    state_dict = {}
                    # Obtener los valores representativos para este estado
                    for i, var_name in enumerate(ordered_vars):
                        state_dict[var_name] = self._get_representative_value_from_index(
                            state_indices_tuple[i], var_name
                        )

                    # Obtener los Q-values y visit counts para este estado
                    q_values = np_q_table[state_indices_tuple] # Es un array 1D de acciones
                    visit_counts = np_visits[state_indices_tuple]

                    # Crear diccionario para Q-table y añadirlo a la lista
                    q_row = state_dict.copy()
                    for action_idx in range(action_dim):
                        # Usar strings para las claves de acción, como en el JSON original
                        # Convertir valores numpy a tipos nativos de Python para JSON
                        q_row[str(action_idx)] = q_values[action_idx].item()
                    q_table_list.append(q_row)

                    # Crear diccionario para Visit Counts y añadirlo a la lista
                    visit_row = state_dict.copy()
                    for action_idx in range(action_dim):
                        # Usar strings para las claves de acción
                        # Convertir valores numpy a tipos nativos de Python para JSON
                        visit_row[str(action_idx)] = visit_counts[action_idx].item()
                    visit_count_list.append(visit_row)

                structured_q_tables[gain] = q_table_list
                structured_visit_counts[gain] = visit_count_list
                logging.debug(f"Structuring for gain '{gain}' complete.")
            else:
                 logging.debug(f"Skipping data structuring for gain '{gain}' (disabled or not initialized).")

        logging.info("Agent state structuring complete.")

        # Devolver el nuevo formato estructurado
        return {
            "q_tables": structured_q_tables,
            "visit_counts": structured_visit_counts,
        }