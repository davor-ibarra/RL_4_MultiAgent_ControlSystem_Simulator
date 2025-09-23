import numpy as np
import math, json
from interfaces import RLAgent, RewardFunction
from factories import ComponentRegistry
from typing import Dict, List, Tuple, Any, Union, Optional
from itertools import product

@ComponentRegistry.register_component("reward", "GaussianReward")
class GaussianRewardFunction(RewardFunction):
    def __init__(self, state_variables: List[str], std_devs: List[float], weights: List[float], use_next_state:bool = True):
        if not (len(state_variables) == len(std_devs) == len(weights)):
            raise ValueError("Las listas de variables, std_devs y weights deben tener la misma longitud.")

        self.state_variables = state_variables
        self.std_devs = std_devs
        self.weights = weights
        self.use_next_state = use_next_state

    @staticmethod
    def gaussian_reward(x: float, std_dev: float) -> float:
        return math.exp(-(x**2) / (2 * std_dev**2))

    def compute_reward(self, state: List[float], action: float, next_state: List[float], **kwargs) -> float:
        total_reward = 0.0
        current_state = next_state if self.use_next_state else state
        state_dict = kwargs.get("state_dict", {})

        for var, std_dev, weight in zip(self.state_variables, self.std_devs, self.weights):
            if var in state_dict:
                total_reward += weight * self.gaussian_reward(state_dict[var], std_dev)
        return total_reward

@ComponentRegistry.register_component("agent", "QLearning")
class QLearning(RLAgent):
    def __init__(self, state_config: Dict, action_space: List[float], learning_rate: float, discount_factor: float,
                 exploration_rate: float, exploration_decay: float, min_exploration_rate:float,
                 variables_to_include:List[str], q_table_filename:str = None, success_reward_factor: float = 10.0):
        self.state_config = state_config
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.variables_to_include = variables_to_include
        self.state_shape = self.calculate_state_shape()
        self.q_table = self.initialize_q_table(q_table_filename)
        self.current_state = None
        self.episode_history: List[Tuple[Tuple[int, ...], int]] = []
        self.success_reward_factor = success_reward_factor
        self.q_table_filename = q_table_filename

    def _discretize_state(next_state_dict, state_config, variables_to_include):
        pass

    def calculate_state_shape(self) -> Tuple[int, ...]:
        shape = []
        for var in self.variables_to_include:
            config = self.state_config.get(var)
            if config and config.get('enabled'):
                shape.append(config['bins'])
            else:
                shape.append(1)
        return tuple(shape)

    def initialize_q_table(self, filename: str = None) -> np.ndarray:
        num_actions = len(self.action_space)
        shape = self.state_shape + (num_actions,)
        if filename:
            try:
                self.load_q_table(filename)  # Intenta cargar desde JSON
                return self.q_table #Retorna la q_table cargada
            except (IOError, ValueError):
                print(f"No se pudo cargar la Q-table desde {filename}. Inicializando una nueva.")
        return np.zeros(shape)

    def save_q_table(self, filename: str) -> None:
        """Guarda la Q-table en un archivo JSON."""
        try:
            with open(filename, 'w') as f:
                json.dump(self._q_table_to_dict(self.q_table), f, indent=4)
            print(f"Q-table guardada en {filename}")
        except IOError:
            print(f"Error al guardar la Q-table en {filename}")

    def load_q_table(self, filename: str) -> None:
        """Carga la Q-table desde un archivo JSON."""
        try:
            with open(filename, 'r') as f:
                q_table_dict = json.load(f)
            self.q_table = self._q_table_from_dict(q_table_dict)
            print(f"Q-table cargada desde {filename}")
        except (IOError, ValueError) as e:
            print(f"No se pudo cargar la Q-table desde {filename}. Error: {e}")
            raise #Importante para que se detenga la ejecución

    def _q_table_to_dict(self, q_table: np.ndarray) -> Dict:
        """Convierte la Q-table (NumPy array) a un diccionario."""
        return {"shape": list(q_table.shape), "data": q_table.tolist()}

    def _q_table_from_dict(self, q_table_dict: Dict) -> np.ndarray:
        """Convierte un diccionario a una Q-table (NumPy array)."""
        return np.array(q_table_dict["data"]).reshape(q_table_dict["shape"])


    def perceive(self, state: List[float])-> None:
      state_dict = {var_name: value for var_name, value in zip(["x", "x_dot", "theta", "theta_dot"], state)}
      #Se agregan las ganancias si es que estan dentro de las variables a incluir
      for var in self.variables_to_include:
        if var in ["kp","ki","kd"]:
          state_dict[var] = state[var]
      self.current_state = self._discretize_state(state_dict, self.state_config, self.variables_to_include)

    def decide(self, state: List[float]) -> Dict[str, Any]:
        self.perceive(state)
        if np.random.random() < self.exploration_rate:
            action_index = np.random.choice(len(self.action_space))
        else:
            action_index = np.argmax(self.q_table[self.current_state])

        self.episode_history.append((self.current_state, action_index))
        return {"action_index": action_index}

    def receive_reward(self, reward: float, next_state: Union[List[float], None]) -> None:
        """
        Recibe la recompensa y el siguiente estado, y actualiza la Q-table.
        Maneja el caso donde next_state es None (fin del episodio dentro del intervalo).
        """
        next_state_dict = {var_name: value for var_name, value in zip(["x", "x_dot", "theta", "theta_dot"], next_state)} if next_state is not None else {}
        for var in self.variables_to_include:
            if var in ["kp","ki","kd"]:
                next_state_dict[var] = next_state[var] if next_state is not None else self.current_state[self.variables_to_include.index(var)]

        next_state_discrete = self._discretize_state(next_state_dict, self.state_config, self.variables_to_include) if next_state is not None else self.current_state
        action_taken = np.argmax(self.q_table[self.current_state])

        q_current = self.q_table[self.current_state + (action_taken,)]

        if next_state is None:
            q_next_max = 0
        else:
            q_next_max = np.max(self.q_table[next_state_discrete])

        q_new = (1 - self.learning_rate) * q_current + self.learning_rate * (reward + self.discount_factor * q_next_max)
        self.q_table[self.current_state + (action_taken,)] = q_new


    def update_policy(self) -> None:
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def is_active(self) -> bool:
        return True

    def reset(self) -> None:
      self.current_state = None
      self.episode_history = []

    def reward_successful_trajectory(self, success_factor: float) -> None:
        for state, action_index in self.episode_history:
            self.q_table[state + (action_index,)] *= success_factor

    def get_action_from_index(self, index: int) -> float:
        return self.action_space[index]

@ComponentRegistry.register_component("agent", "PIDQLearning")
class PIDQLearning(RLAgent):
    def __init__(self, state_config: Dict, kp_agent: QLearning, ki_agent: QLearning, kd_agent: QLearning, enabled: bool = True):
        self.enabled = enabled
        self.kp_agent = kp_agent
        self.ki_agent = ki_agent
        self.kd_agent = kd_agent
        self.sub_agents = [self.kp_agent, self.ki_agent, self.kd_agent]
        self.single_q_agent = all(agent == self.kp_agent for agent in self.sub_agents)

    def perceive(self, state: List[float]) -> None:
        if self.enabled:
            for agent in self.sub_agents:
                agent.perceive(state)

    def decide(self, state: List[float]) -> Dict[str, Any]:
        actions = {}
        if self.enabled:
            if self.single_q_agent:
              all_actions = self.kp_agent.decide(state)
              actions = {
                    'kp': self.kp_agent.get_action_from_index(all_actions['action_0']),
                    'ki': self.ki_agent.get_action_from_index(all_actions['action_1']),
                    'kd': self.kd_agent.get_action_from_index(all_actions['action_2'])
              }
            else:
              kp_action_index = self.kp_agent.decide(state)['action_index']
              ki_action_index = self.ki_agent.decide(state)['action_index']
              kd_action_index = self.kd_agent.decide(state)['action_index']
              actions =  {
                  'kp': self.kp_agent.get_action_from_index(kp_action_index),
                  'ki': self.ki_agent.get_action_from_index(ki_action_index),
                  'kd': self.kd_agent.get_action_from_index(kd_action_index)
              }
        return actions

    def receive_reward(self, reward: float, next_state: List[float]) -> None:
        if self.enabled:
            for agent in self.sub_agents:
                agent.receive_reward(reward, next_state)

    def update_policy(self) -> None:
        if self.enabled:
            for agent in self.sub_agents:
                agent.update_policy()

    def is_active(self) -> bool:
        return self.enabled

    def reset(self) -> None:
        for agent in self.sub_agents:
            agent.reset()

    def reward_successful_trajectory(self, success_factor: float) -> None:
      if self.enabled:
        for agent in self.sub_agents:
            agent.reward_successful_trajectory(success_factor)