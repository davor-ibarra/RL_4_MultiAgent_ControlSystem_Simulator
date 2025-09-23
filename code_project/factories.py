import importlib
from typing import Dict, Any, Type, List
from interfaces import DynamicSystem, Controller, RLAgent, Environment, RewardFunction, MetricsCollector
from reinforcement_learning import QLearning, PIDQLearning

class ComponentRegistry:
    _registry: Dict[str, Dict[str, Type[Any]]] = {
        "system": {}, "controller": {}, "agent": {}, "environment": {}, "reward": {}, "metrics": {}
    }
    @staticmethod
    def register_component(component_type: str, name: str, component: Type[Any]) -> None:
        if component_type not in ComponentRegistry._registry:
            raise ValueError(f"Tipo de componente inválido: {component_type}")
        ComponentRegistry._registry[component_type][name] = component
    @staticmethod
    def get_component(component_type: str, name: str) -> Type[Any]:
        if component_type not in ComponentRegistry._registry:
            raise ValueError(f"Tipo de componente inválido: {component_type}")
        if name not in ComponentRegistry._registry[component_type]:
            raise ValueError(f"Componente '{name}' no encontrado para el tipo '{component_type}'.")
        return ComponentRegistry._registry[component_type][name]
    @staticmethod
    def list_components(component_type: str) -> List[str]:
        if component_type not in ComponentRegistry._registry:
            raise ValueError(f"Tipo de componente inválido: {component_type}")
        return list(ComponentRegistry._registry[component_type].keys())

class SystemFactory:
    @staticmethod
    def create_system(system_type: str, **kwargs: Any) -> DynamicSystem:
        try:
            system_class = ComponentRegistry.get_component("system", system_type)
            return system_class(**kwargs)
        except ValueError as e:
            raise ValueError(f"Error al crear el sistema: {e}")

class ControllerFactory:
    @staticmethod
    def create_controller(controller_type: str, **kwargs: Any) -> Controller:
        try:
            controller_class = ComponentRegistry.get_component("controller", controller_type)
            return controller_class(**kwargs)
        except ValueError as e:
            raise ValueError(f"Error al crear el controlador: {e}")

class AgentFactory:
    @staticmethod
    def create_agent(agent_config: Dict[str, Any], state_config:Dict) -> RLAgent:
        agent_type = agent_config['type']
        agent_params = agent_config.get('parameters', {})

        try:
            if agent_type == "PIDQLearning":
                # Configuración especial para PIDQLearning (agentes compuestos)
                sub_agents_config = agent_config.get('sub_agents', {})
                sub_agents = {}
                for gain, sub_agent_type_and_params in sub_agents_config.items():
                  sub_agent_type = sub_agent_type_and_params['type']
                  sub_agent_params = sub_agent_type_and_params.get('parameters', {})
                  # Si se especifica un solo agente Q para todas las ganancias
                  if sub_agent_type == 'SingleQAgent':
                      #Variables a incluir en el estado
                      variables_to_include = sub_agent_params.get('variables_to_include', [])
                      # Se crea un agente Q para todas las acciones
                      all_actions = [sub_agents_config[g]['parameters']['actions'] for g in ['kp', 'ki', 'kd']]

                      # Crear el único agente Q con el producto cartesiano de las acciones
                      q_agent = QLearning(
                            state_config=state_config,
                            action_space=all_actions,  # Todas las combinaciones posibles
                            learning_rate=sub_agent_params['learning_rate'],
                            discount_factor=sub_agent_params['discount_factor'],
                            exploration_rate=sub_agent_params['exploration_rate'],
                            exploration_decay=sub_agent_params['exploration_decay'],
                            min_exploration_rate=sub_agent_params['min_exploration_rate'],
                            variables_to_include= variables_to_include
                      )
                      # Llenar los sub-agentes con copias del agente único para mantener la estructura
                      for g in sub_agents_config.keys():
                        sub_agents[g] = q_agent
                      break  # Importante salir del bucle
                  else:
                    #Variables a incluir en el estado para cada sub agente
                    variables_to_include = sub_agent_params.get('variables_to_include', [])
                    sub_agents[gain] = QLearning(
                        state_config=state_config,
                        action_space = sub_agent_params['actions'],
                        learning_rate=sub_agent_params['learning_rate'],
                        discount_factor=sub_agent_params['discount_factor'],
                        exploration_rate=sub_agent_params['exploration_rate'],
                        exploration_decay=sub_agent_params['exploration_decay'],
                        min_exploration_rate=sub_agent_params['min_exploration_rate'],
                        variables_to_include= variables_to_include,
                        q_table_filename=sub_agent_params.get('q_table_filename')
                    )
                return PIDQLearning(
                    state_config=state_config,
                    kp_agent=sub_agents.get('kp'),
                    ki_agent=sub_agents.get('ki'),
                    kd_agent=sub_agents.get('kd'),
                    enabled=agent_params.get('enabled', True)
                )
            else:
              agent_class = ComponentRegistry.get_component("agent", agent_type)
              return agent_class(state_config = state_config, **agent_params)

        except ValueError as e:
            raise ValueError(f"Error al crear el agente: {e}")

class RewardFactory:
    @staticmethod
    def create_reward_function(function_type: str, **kwargs: Any) -> RewardFunction:
        try:
            reward_class = ComponentRegistry.get_component("reward", function_type)
            return reward_class(**kwargs)
        except ValueError as e:
            raise ValueError(f"Error al crear la función de recompensa: {e}")

class EnvironmentFactory:
    @staticmethod
    def create_environment(environment_type: str, **kwargs: Any) -> Environment:
        try:
            env_class = ComponentRegistry.get_component("environment", environment_type)
            return env_class(**kwargs)
        except ValueError as e:
            raise ValueError(f"Error al crear el entorno: {e}")