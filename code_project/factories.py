import importlib

class SystemFactory:
    @staticmethod
    def create_system(system_type, **kwargs):
        module = importlib.import_module('dynamic_system')
        system_class = getattr(module, system_type)
        return system_class(**kwargs)

class ControllerFactory:
    @staticmethod
    def create_controller(controller_type, **kwargs):
        module = importlib.import_module('controller')
        controller_class = getattr(module, controller_type)
        return controller_class(**kwargs)

class AgentFactory:
    @staticmethod
    def create_agent(agent_type, **kwargs):
        module = importlib.import_module('reinforcement_learning')
        agent_class = getattr(module, agent_type)
        return agent_class(**kwargs)

class RewardFactory:
    @staticmethod
    def create_reward_function(function_type, **kwargs):
        module = importlib.import_module('reinforcement_learning')
        reward_class = getattr(module, function_type)
        return reward_class(**kwargs)

class EnvironmentFactory:
    @staticmethod
    def create_environment(environment_type, **kwargs):
        module = importlib.import_module('dynamic_system')
        env_class = getattr(module, environment_type)
        return env_class(**kwargs)

class ComponentRegistry:
    _registry = {}

    @staticmethod
    def register_component(name, component):
        ComponentRegistry._registry[name] = component

    @staticmethod
    def get_component(name):
        return ComponentRegistry._registry.get(name)
