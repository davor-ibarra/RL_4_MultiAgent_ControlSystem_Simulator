from configuration import ConfigurationManager, ValidationManager
from factories import SystemFactory, ControllerFactory, AgentFactory, RewardFactory
from orchestrator import SimulationOrchestrator
from dynamic_system import InvertedPendulum
from controller import PIDController
from reinforcement_learning import PIDQLearning
from simulator import PendulumControl, Run_Simulation
import os

def main():
    # Cargar y validar la configuración
    config_path = os.getcwd() + '\\V3_code\\' + 'config.json'
    config_manager = ConfigurationManager(config_path)
    config = config_manager.load_config()
    ValidationManager(config).validate()

    # Crear el sistema dinámico mediante la factoría
    dynamic_system = SystemFactory.create_system("InvertedPendulum", **config['physical'])
    
    # Crear el controlador mediante la factoría
    controller = ControllerFactory.create_controller("PIDController", **config['controller'],
                                                     dt=config['simulation']['dt'])
    
    # Crear el agente RL mediante la factoría
    agent = AgentFactory.create_agent("PIDQLearning", 
                                      state_config=config['state'],
                                      num_actions=config['rl']['num_actions'],
                                      discount_factor=config['rl']['discount_factor'],
                                      epsilon=config['rl']['epsilon'],
                                      epsilon_decay=config['rl']['epsilon_decay'],
                                      epsilon_min=config['rl']['epsilon_min'],
                                      use_epsilon_decay=config['rl']['use_epsilon_decay'],
                                      learning_rate=config['rl']['learning_rate'],
                                      learning_rate_decay=config['rl']['learning_rate_decay'],
                                      learning_rate_min=config['rl']['learning_rate_min'],
                                      use_learning_rate_decay=config['rl']['use_learning_rate_decay'],
                                      consider_done=True)
    
    # Crear la función de recompensa mediante RewardFactory
    reward_function = RewardFactory.create_reward_function("GaussianRewardFunction",
                                                           reward_weights=config['rl']['reward_weights'],
                                                           reward_scales=config['rl']['reward_scales'])
    # En este ejemplo, usamos el sistema dinámico como entorno
    environment = dynamic_system

    # Configurar el simulador con integración de controlador y agente RL
    controller_params = config['controller']
    simulation_params = config['simulation']
    pendulum_control = PendulumControl(
        pendulum=dynamic_system,
        q_learner=agent,
        kp=controller_params['kp'],
        ki=controller_params['ki'],
        kd=controller_params['kd'],
        setpoint=controller_params['setpoint'],
        dt=simulation_params['dt'],
        gain_step=controller_params['gain_step'],
        state_config=config['state'],
        variable_step=controller_params['variable_step'],
        extract_qtables=simulation_params['extract_qtables'],
        extract_frequency=simulation_params['extract_frequency'],
        use_angle_limit=simulation_params['use_angle_limit'],
        angle_limit=simulation_params['angle_limit'],
        use_cart_limit=simulation_params['use_cart_limit'],
        cart_limit=simulation_params['cart_limit'],
        use_controller=simulation_params['use_controller'],
        use_qlearning=simulation_params['use_qlearning'],
        max_episodes=simulation_params['max_episodes'],
        reset_gains_each_episode=controller_params['reset_gains_each_episode'],
        reward_weights=config['rl']['reward_weights'],
        reward_scales=config['rl']['reward_scales'],
        decision_interval=simulation_params['decision_interval'],
        success_scaling_factor=1.5
    )

    # Opción 1: Usar el Run_Simulation de simulator.py
    simulation = Run_Simulation(config, pendulum_control)
    simulation.run()

    # Opción 2 (alternativa): Usar el SimulationOrchestrator (para una arquitectura multi-entorno)
    # from orchestrator import SimulationOrchestrator
    # simulation_orchestrator = SimulationOrchestrator(
    #     environment=environment,
    #     controller=controller,
    #     agent_manager=agent,
    #     reward_calculator=reward_function,
    #     metrics_collector=None,
    #     config=config
    # )
    # simulation_orchestrator.run_simulation()

if __name__ == "__main__":
    main()
