import numpy as np
import math
from interfaces import RLAgent, RewardFunction

def discretize_state(state_values, state_config, variables_to_include):
    state_indices = []
    for var in variables_to_include:
        cfg = state_config.get(var)
        if cfg and cfg.get('enabled'):
            min_val = cfg['min']
            max_val = cfg['max']
            bins = cfg['bins']
            value = state_values.get(var, min_val)
            step_size = (max_val - min_val) / (bins - 1)
            if value <= min_val:
                index = 0
            elif value >= max_val:
                index = bins - 1
            else:
                index = int(round((value - min_val) / step_size))
            state_indices.append(index)
    return tuple(state_indices)

def calculate_reward(pendulum_angle, pendulum_velocity, cart_position, cart_velocity, force, time, reward_weights, reward_scales):
    angle_norm = pendulum_angle / reward_scales['angle']
    velocity_norm = pendulum_velocity / reward_scales['angular_velocity']
    force_norm = force / reward_scales['force']
    k1 = reward_weights.get('stability', 1.0)
    k2 = reward_weights.get('force', 1.0)
    R_stability = k1 * math.exp(-angle_norm**2) * math.exp(-velocity_norm**2)
    R_force = k2 * math.exp(-force_norm**2)
    return R_stability - R_force

class GaussianRewardFunction(RewardFunction):
    def __init__(self, reward_weights, reward_scales):
        self.reward_weights = reward_weights
        self.reward_scales = reward_scales

    def compute_reward(self, state, action, next_state, **kwargs):
        # Asumimos que el estado es una lista/array con: [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
        pendulum_angle = next_state[2]
        pendulum_velocity = next_state[3]
        cart_position = next_state[0]
        cart_velocity = next_state[1]
        # Si se requiere, se pueden incluir parámetros adicionales de kwargs, por ejemplo 'force' o 'time'
        force = kwargs.get('force', 0)
        time_val = kwargs.get('time', 0)
        # Se reutiliza la función calculate_reward definida en este módulo
        return calculate_reward(pendulum_angle, pendulum_velocity, cart_position, cart_velocity, force, time_val,
                                  self.reward_weights, self.reward_scales)

class QLearning:
    def __init__(self, state_config, state_dims, num_actions, discount_factor,
                 epsilon=0.5, epsilon_decay=0.959, epsilon_min=0.01, use_epsilon_decay=False,
                 learning_rate=0.1, learning_rate_decay=0.959, learning_rate_min=0.01, use_learning_rate_decay=False,
                 consider_done=True):
        self.state_config = state_config
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_epsilon_decay = use_epsilon_decay
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.use_learning_rate_decay = use_learning_rate_decay
        self.consider_done = consider_done

        self.q_min = -1e5
        self.q_max = 1e5
        self.q_table = np.zeros(tuple(self.state_dims) + (self.num_actions,))
        self.episode_record = []

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return int(np.random.choice(self.num_actions))
        else:
            return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done=False):
        old_value = self.q_table[state + (action,)]
        next_max = 0 if (self.consider_done and done) else np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state + (action,)] = max(self.q_min, min(self.q_max, new_value))

    def update_epsilon(self):
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_learning_rate(self):
        if self.use_learning_rate_decay:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay)

    def reset(self):
        self.q_table.fill(0)

    def get_qtable(self):
        qtable_list = []
        for state_indices in np.ndindex(self.q_table.shape[:-1]):
            state_values = self._get_state_values(state_indices)
            row = {**state_values}
            for a in range(self.num_actions):
                row[f'action_{a}'] = self.q_table[state_indices + (a,)]
            qtable_list.append(row)
        return qtable_list

    def _get_state_values(self, state_indices):
        state_values = {}
        enabled_vars = [var for var, cfg in self.state_config.items() if cfg.get('enabled')]
        for var, index in zip(enabled_vars, state_indices):
            cfg = self.state_config[var]
            min_val = cfg['min']
            max_val = cfg['max']
            bins = cfg['bins']
            step_size = (max_val - min_val) / (bins - 1)
            state_values[var] = min_val + index * step_size
        return state_values

    def reset_episode_record(self):
        self.episode_record = []

    def register_episode_decision(self, state, action):
        self.episode_record.append((state, action))

    def apply_expansion_factor(self, factor):
        for state, action in self.episode_record:
            self.q_table[state + (action,)] *= factor

# PIDQLearning implementa los métodos de RLAgent para cumplir con la interfaz.
class PIDQLearning(RLAgent):
    def __init__(self, state_config, num_actions, discount_factor, 
                 epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.01, use_epsilon_decay=False, 
                 learning_rate=0.1, learning_rate_decay=0.01, learning_rate_min=0.01, use_learning_rate_decay=False,
                 consider_done=True):
        self.state_config = state_config
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_epsilon_decay = use_epsilon_decay
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        self.use_learning_rate_decay = use_learning_rate_decay
        self.consider_done = consider_done

        self.common_state_vars = ['angle', 'angular_velocity']
        self.state_vars_kp = self.common_state_vars + ['kp']
        self.state_vars_ki = self.common_state_vars + ['ki']
        self.state_vars_kd = self.common_state_vars + ['kd']

        self.state_dims_kp = [self.state_config[var]['bins'] for var in self.state_vars_kp if self.state_config[var].get('enabled')]
        self.state_dims_ki = [self.state_config[var]['bins'] for var in self.state_vars_ki if self.state_config[var].get('enabled')]
        self.state_dims_kd = [self.state_config[var]['bins'] for var in self.state_vars_kd if self.state_config[var].get('enabled')]

        self.kp_qlearning = QLearning(self.state_config, self.state_dims_kp, num_actions, discount_factor,
                                      epsilon, epsilon_decay, epsilon_min, use_epsilon_decay,
                                      learning_rate, learning_rate_decay, learning_rate_min, use_learning_rate_decay,
                                      consider_done)
        self.ki_qlearning = QLearning(self.state_config, self.state_dims_ki, num_actions, discount_factor,
                                      epsilon, epsilon_decay, epsilon_min, use_epsilon_decay,
                                      learning_rate, learning_rate_decay, learning_rate_min, use_learning_rate_decay,
                                      consider_done)
        self.kd_qlearning = QLearning(self.state_config, self.state_dims_kd, num_actions, discount_factor,
                                      epsilon, epsilon_decay, epsilon_min, use_epsilon_decay,
                                      learning_rate, learning_rate_decay, learning_rate_min, use_learning_rate_decay,
                                      consider_done)

    def choose_actions(self, state_values):
        state_kp = discretize_state(state_values, self.state_config, self.state_vars_kp)
        state_ki = discretize_state(state_values, self.state_config, self.state_vars_ki)
        state_kd = discretize_state(state_values, self.state_config, self.state_vars_kd)
        actions = {
            'kp': self.kp_qlearning.choose_action(state_kp),
            'ki': self.ki_qlearning.choose_action(state_ki),
            'kd': self.kd_qlearning.choose_action(state_kd)
        }
        return actions

    def learn(self, state_values, actions, reward, next_state_values, done=False):
        state_kp = discretize_state(state_values, self.state_config, self.state_vars_kp)
        next_state_kp = discretize_state(next_state_values, self.state_config, self.state_vars_kp)
        self.kp_qlearning.learn(state_kp, actions['kp'], reward, next_state_kp, done=done)
        state_ki = discretize_state(state_values, self.state_config, self.state_vars_ki)
        next_state_ki = discretize_state(next_state_values, self.state_config, self.state_vars_ki)
        self.ki_qlearning.learn(state_ki, actions['ki'], reward, next_state_ki, done=done)
        state_kd = discretize_state(state_values, self.state_config, self.state_vars_kd)
        next_state_kd = discretize_state(next_state_values, self.state_config, self.state_vars_kd)
        self.kd_qlearning.learn(state_kd, actions['kd'], reward, next_state_kd, done=done)

    def update_epsilon(self):
        self.kp_qlearning.update_epsilon()
        self.ki_qlearning.update_epsilon()
        self.kd_qlearning.update_epsilon()

    def update_learning_rate(self):
        self.kp_qlearning.update_learning_rate()
        self.ki_qlearning.update_learning_rate()
        self.kd_qlearning.update_learning_rate()

    def get_qtables(self):
        return {
            'kp': self.kp_qlearning.get_qtable(),
            'ki': self.ki_qlearning.get_qtable(),
            'kd': self.kd_qlearning.get_qtable()
        }

    def reset_episode_records(self):
        self.kp_qlearning.reset_episode_record()
        self.ki_qlearning.reset_episode_record()
        self.kd_qlearning.reset_episode_record()

    def register_episode_decision(self, state_values, actions):
        state_kp = discretize_state(state_values, self.state_config, self.state_vars_kp)
        state_ki = discretize_state(state_values, self.state_config, self.state_vars_ki)
        state_kd = discretize_state(state_values, self.state_config, self.state_vars_kd)
        self.kp_qlearning.register_episode_decision(state_kp, actions['kp'])
        self.ki_qlearning.register_episode_decision(state_ki, actions['ki'])
        self.kd_qlearning.register_episode_decision(state_kd, actions['kd'])

    def apply_expansion_factor(self, factor):
        self.kp_qlearning.apply_expansion_factor(factor)
        self.ki_qlearning.apply_expansion_factor(factor)
        self.kd_qlearning.apply_expansion_factor(factor)

    # Implementación de los métodos abstractos de RLAgent
    def perceive(self, state):
        # Simplemente almacenamos el estado para uso posterior.
        self.last_state = state

    def decide(self):
        # Si se ha percibido un estado, se retornan las acciones basadas en ese estado.
        if hasattr(self, 'last_state'):
            return self.choose_actions(self.last_state)
        return {'kp': 1, 'ki': 1, 'kd': 1}

    def receive_reward(self, reward, next_state):
        # En este ejemplo, la actualización se realiza externamente; este método puede dejarse vacío.
        pass

    def update_policy(self):
        self.update_epsilon()
        self.update_learning_rate()

    def is_active(self):
        return True
