from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class DynamicSystem(ABC):
    """Base interface for all dynamic systems."""
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Get current state of the system."""
        pass
    
    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Apply control action to the system."""
        pass
    
    @abstractmethod
    def update(self, dt: float) -> None:
        """Update system state for the given time step."""
        pass
    
    @abstractmethod
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset system to initial state."""
        pass
    
    @abstractmethod
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for each state variable (min, max)."""
        pass
    
    @abstractmethod
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for each action variable (min, max)."""
        pass

class Controller(ABC):
    """Base interface for all controllers."""
    
    @abstractmethod
    def compute_action(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute control action based on current state and reference."""
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update controller parameters."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current controller parameters."""
        pass

class RLAgent(ABC):
    """Base interface for reinforcement learning agents."""
    
    @abstractmethod
    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """Choose action based on current state."""
        pass
    
    @abstractmethod
    def learn(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool) -> None:
        """Learn from experience tuple."""
        pass
    
    @abstractmethod
    def update_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Update agent hyperparameters."""
        pass
    
    @abstractmethod
    def save_state(self, path: str) -> None:
        """Save agent state/model."""
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> None:
        """Load agent state/model."""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        pass

class Environment(ABC):
    """Base interface for simulation environments."""
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render environment state."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get observation space bounds."""
        pass
    
    @abstractmethod
    def get_action_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action space bounds."""
        pass

class RewardFunction(ABC):
    """Base interface for reward functions."""
    
    @abstractmethod
    def calculate(self, state: np.ndarray, action: np.ndarray, 
                next_state: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate reward based on transition."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get reward function parameters."""
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update reward function parameters."""
        pass

class MetricsCollector(ABC):
    """Base interface for collecting and storing metrics."""
    
    @abstractmethod
    def record(self, metrics: Dict[str, Any]) -> None:
        """Record metrics for current step/episode."""
        pass
    
    @abstractmethod
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for current episode."""
        pass
    
    @abstractmethod
    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """Get all recorded metrics."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metrics collector."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save metrics to file."""
        pass

class ConfigValidator(ABC):
    """Base interface for configuration validation."""
    
    @abstractmethod
    def validate_system_config(self, config: Dict[str, Any]) -> bool:
        """Validate dynamic system configuration."""
        pass
    
    @abstractmethod
    def validate_controller_config(self, config: Dict[str, Any]) -> bool:
        """Validate controller configuration."""
        pass
    
    @abstractmethod
    def validate_agent_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        pass
    
    @abstractmethod
    def validate_environment_config(self, config: Dict[str, Any]) -> bool:
        """Validate environment configuration."""
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        pass
