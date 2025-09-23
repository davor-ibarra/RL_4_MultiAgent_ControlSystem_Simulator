from abc import ABC, abstractmethod

# Interfaz para sistemas dinámicos
class DynamicSystem(ABC):
    @abstractmethod
    def initialize_state(self):
        pass

    @abstractmethod
    def update_state(self, dt, control_input):
        pass

    @abstractmethod
    def get_state(self):
        pass

    def connect_controller(self, controller):
        self.controller = controller

# Interfaz para controladores
class Controller(ABC):
    @abstractmethod
    def compute(self, error):
        pass

    @abstractmethod
    def update_parameters(self, **kwargs):
        pass

# Interfaz para agentes de aprendizaje por refuerzo
class RLAgent(ABC):
    @abstractmethod
    def perceive(self, state):
        pass

    @abstractmethod
    def decide(self):
        pass

    @abstractmethod
    def receive_reward(self, reward, next_state):
        pass

    @abstractmethod
    def update_policy(self):
        pass

    @abstractmethod
    def is_active(self):
        pass

# Interfaz para entornos de simulación
class Environment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

# Interfaz para funciones de recompensa
class RewardFunction(ABC):
    @abstractmethod
    def compute_reward(self, state, action, next_state, **kwargs):
        pass

# Interfaz para recolectores de métricas
class MetricsCollector(ABC):
    @abstractmethod
    def record(self, **kwargs):
        pass
