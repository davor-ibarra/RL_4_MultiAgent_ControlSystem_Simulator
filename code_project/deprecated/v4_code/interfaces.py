from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Union
class DynamicSystem(ABC):
    @abstractmethod
    def initialize_state(self) -> None: pass
    @abstractmethod
    def update_state(self, dt: float, control_input: float) -> List[float]: pass
    @abstractmethod
    def get_state(self) -> List[float]: pass
    @abstractmethod
    def connect_controller(self, controller: "Controller") -> None: pass

class Controller(ABC):
    @abstractmethod
    def compute(self, state: List[float]) -> float: pass
    @abstractmethod
    def update_parameters(self, **kwargs: Dict[str, Any]) -> None: pass
    @abstractmethod
    def reset(self) -> None: pass
class RLAgent(ABC):
    @abstractmethod
    def perceive(self, state: List[float]) -> None: pass
    @abstractmethod
    def decide(self, state: List[float]) -> Dict[str, Any]: pass
    @abstractmethod
    def receive_reward(self, reward: float, next_state: List[float]) -> None: pass
    @abstractmethod
    def update_policy(self) -> None: pass
    @abstractmethod
    def is_active(self) -> bool: pass
    @abstractmethod
    def reset(self) -> None: pass
    @abstractmethod
    def reward_successful_trajectory(self, success_factor: float) -> None: pass #Nuevo metodo
class Environment(ABC):
    @abstractmethod
    def reset(self) -> List[float]: pass
    @abstractmethod
    def step(self, action: float) -> Tuple[List[float], float, bool, Dict[str, Any]]: pass
    @abstractmethod
    def check_termination(self, state: List[float]) -> bool: pass
class RewardFunction(ABC):
    @abstractmethod
    def compute_reward(self, state: List[float], action: float, next_state: List[float], **kwargs: Dict[str, Any]) -> float: pass
class MetricsCollector(ABC):
    @abstractmethod
    def record(self, **kwargs: Dict[str, Any]) -> None: pass

class ConfigValidator(ABC):
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> str: pass