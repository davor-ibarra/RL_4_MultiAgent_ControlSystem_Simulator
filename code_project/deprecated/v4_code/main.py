import os
from configuration import ConfigurationManager
from factories import SystemFactory, ControllerFactory, AgentFactory, RewardFactory
from orchestrator import SimulationOrchestrator

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config_manager = ConfigurationManager(config_path)
    config = config_manager.load_config()

    system = SystemFactory.create_system(config['system']['type'], **config['system']['parameters'])
    controller = ControllerFactory.create_controller(config['controller']['type'], **config['controller']['parameters'])
    agent = AgentFactory.create_agent(config['rl'], config['state'])  # Pasamos la config completa de RL y del estado
    reward_function = RewardFactory.create_reward_function(config['reward']['type'], **config['reward']['parameters'])

    orchestrator = SimulationOrchestrator(system, controller, agent, reward_function, None, config)
    orchestrator.run_simulation()

if __name__ == "__main__":
    main()