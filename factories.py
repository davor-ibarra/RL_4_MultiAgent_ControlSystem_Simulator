from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
import importlib
import json
from interfaces import DynamicSystem, Controller, RLAgent, Environment, RewardFunction

class SystemFactory:
    """Factory for creating dynamic system instances."""
    
    def __init__(self):
        self._systems: Dict[str, Type[DynamicSystem]] = {}
    
    def register_system(self, system_name: str, system_class: Type[DynamicSystem]) -> None:
        """Register a new dynamic system type."""
        self._systems[system_name] = system_class
    
    def create_system(self, system_name: str, config: Dict[str, Any]) -> DynamicSystem:
        """Create a new dynamic system instance."""
        if system_name not in self._systems:
            raise ValueError(f"Unknown system type: {system_name}")
        
        system_class = self._systems[system_name]
        return system_class(**config)
    
    def get_registered_systems(self) -> list:
        """Get list of registered system types."""
        return list(self._systems.keys())

class ControllerFactory:
    """Factory for creating controller instances."""
    
    def __init__(self):
        self._controllers: Dict[str, Type[Controller]] = {}
    
    def register_controller(self, controller_name: str, controller_class: Type[Controller]) -> None:
        """Register a new controller type."""
        self._controllers[controller_name] = controller_class
    
    def create_controller(self, controller_name: str, config: Dict[str, Any]) -> Controller:
        """Create a new controller instance."""
        if controller_name not in self._controllers:
            raise ValueError(f"Unknown controller type: {controller_name}")
        
        controller_class = self._controllers[controller_name]
        return controller_class(**config)
    
    def get_registered_controllers(self) -> list:
        """Get list of registered controller types."""
        return list(self._controllers.keys())

class AgentFactory:
    """Factory for creating RL agent instances."""
    
    def __init__(self):
        self._agents: Dict[str, Type[RLAgent]] = {}
    
    def register_agent(self, agent_name: str, agent_class: Type[RLAgent]) -> None:
        """Register a new agent type."""
        self._agents[agent_name] = agent_class
    
    def create_agent(self, agent_name: str, config: Dict[str, Any]) -> RLAgent:
        """Create a new agent instance."""
        if agent_name not in self._agents:
            raise ValueError(f"Unknown agent type: {agent_name}")
        
        agent_class = self._agents[agent_name]
        return agent_class(**config)
    
    def get_registered_agents(self) -> list:
        """Get list of registered agent types."""
        return list(self._agents.keys())

class EnvironmentFactory:
    """Factory for creating environment instances."""
    
    def __init__(self):
        self._environments: Dict[str, Type[Environment]] = {}
    
    def register_environment(self, env_name: str, env_class: Type[Environment]) -> None:
        """Register a new environment type."""
        self._environments[env_name] = env_class
    
    def create_environment(self, env_name: str, config: Dict[str, Any]) -> Environment:
        """Create a new environment instance."""
        if env_name not in self._environments:
            raise ValueError(f"Unknown environment type: {env_name}")
        
        env_class = self._environments[env_name]
        return env_class(**config)
    
    def get_registered_environments(self) -> list:
        """Get list of registered environment types."""
        return list(self._environments.keys())

class RewardFactory:
    """Factory for creating reward function instances."""
    
    def __init__(self):
        self._reward_functions: Dict[str, Type[RewardFunction]] = {}
    
    def register_reward_function(self, reward_name: str, reward_class: Type[RewardFunction]) -> None:
        """Register a new reward function type."""
        self._reward_functions[reward_name] = reward_class
    
    def create_reward_function(self, reward_name: str, config: Dict[str, Any]) -> RewardFunction:
        """Create a new reward function instance."""
        if reward_name not in self._reward_functions:
            raise ValueError(f"Unknown reward function type: {reward_name}")
        
        reward_class = self._reward_functions[reward_name]
        return reward_class(**config)
    
    def get_registered_reward_functions(self) -> list:
        """Get list of registered reward function types."""
        return list(self._reward_functions.keys())

class ComponentRegistry:
    """Central registry for all system components."""
    
    def __init__(self):
        self.system_factory = SystemFactory()
        self.controller_factory = ControllerFactory()
        self.agent_factory = AgentFactory()
        self.environment_factory = EnvironmentFactory()
        self.reward_factory = RewardFactory()
    
    def load_components_from_config(self, config_path: str) -> None:
        """Load and register components from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load and register systems
        for system_info in config.get('systems', []):
            module = importlib.import_module(system_info['module'])
            system_class = getattr(module, system_info['class'])
            self.system_factory.register_system(system_info['name'], system_class)
        
        # Load and register controllers
        for controller_info in config.get('controllers', []):
            module = importlib.import_module(controller_info['module'])
            controller_class = getattr(module, controller_info['class'])
            self.controller_factory.register_controller(controller_info['name'], controller_class)
        
        # Load and register agents
        for agent_info in config.get('agents', []):
            module = importlib.import_module(agent_info['module'])
            agent_class = getattr(module, agent_info['class'])
            self.agent_factory.register_agent(agent_info['name'], agent_class)
        
        # Load and register environments
        for env_info in config.get('environments', []):
            module = importlib.import_module(env_info['module'])
            env_class = getattr(module, env_info['class'])
            self.environment_factory.register_environment(env_info['name'], env_class)
        
        # Load and register reward functions
        for reward_info in config.get('reward_functions', []):
            module = importlib.import_module(reward_info['module'])
            reward_class = getattr(module, reward_info['class'])
            self.reward_factory.register_reward_function(reward_info['name'], reward_class)
    
    def get_component_info(self) -> Dict[str, list]:
        """Get information about all registered components."""
        return {
            'systems': self.system_factory.get_registered_systems(),
            'controllers': self.controller_factory.get_registered_controllers(),
            'agents': self.agent_factory.get_registered_agents(),
            'environments': self.environment_factory.get_registered_environments(),
            'reward_functions': self.reward_factory.get_registered_reward_functions()
        }
