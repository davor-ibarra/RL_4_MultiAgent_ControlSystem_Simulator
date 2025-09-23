import time
import json
import os
import psutil
import platform
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from interfaces import Environment, Controller, RLAgent, RewardFunction, MetricsCollector
from dataclasses import dataclass

@dataclass
class StepData:
    state: List[float]
    control_action: float
    agent_actions: Dict[str, Any]
    reward: float
    next_state: List[float]

@dataclass
class EpisodeData:
    episode: int
    steps: List[StepData]
    duration: float
    total_reward: float
    terminated: bool
    termination_reason: str = ""

class SimulationOrchestrator:
    def __init__(self, environment: Environment, controller: Controller, agent_manager: RLAgent,
                 reward_calculator: RewardFunction, metrics_collector: MetricsCollector, config: Dict[str,Any]):
        self.environment = environment
        self.controller = controller
        self.agent_manager = agent_manager
        self.reward_calculator = reward_calculator
        self.metrics_collector = metrics_collector
        self.config = config
        self.episodes = config['simulation']['max_episodes']
        self.time_step = config['simulation']['time_step']
        self.decision_interval = config['simulation']['decision_interval']
        self.success_reward_factor = config['rl'].get('success_reward_factor', 10.0)
        self.q_table_save_frequency = self.config['rl'].get('q_table_save_frequency', 100) #Cada cuantos episodios se guarda
        self.results_folder = os.path.join(config['simulation'].get('results_folder', 'results_history'),
                                           datetime.now().strftime('%Y%m%d-%H%M'))
        os.makedirs(self.results_folder, exist_ok=True)
        self.performance_data = {
            'episode_times': [],
            'cpu_usages': [],
            'memory_usages': [],
            'file_save_times': []
        }
        self.simulation_data: List[EpisodeData] = []
        self.state_vars = list(config['state'].keys())

        # Cargar la Q-table al inicio (si existe y si el agente es QLearning/PIDQLearning)
        if hasattr(self.agent_manager, 'kp_agent') and self.agent_manager.kp_agent.q_table_filename:  # Para PIDQLearning
            self.agent_manager.kp_agent.load_q_table(self.agent_manager.kp_agent.q_table_filename)
            if not self.agent_manager.single_q_agent: #Si tiene multiples agentes
              self.agent_manager.ki_agent.load_q_table(self.agent_manager.ki_agent.q_table_filename)
              self.agent_manager.kd_agent.load_q_table(self.agent_manager.kd_agent.q_table_filename)
        elif hasattr(self.agent_manager, 'q_table_filename') and self.agent_manager.q_table_filename:  # Para QLearning
            self.agent_manager.load_q_table(self.agent_manager.q_table_filename)



    def run_simulation(self) -> None:
        total_start = time.time()
        episodes_per_file = self.config['simulation'].get('episodes_per_file', 100)
        current_batch: List[EpisodeData] = []

        for episode in range(self.episodes):
            episode_data = self.run_episode(episode)
            self.simulation_data.append(episode_data)
            current_batch.append(episode_data)

            if (episode + 1) % episodes_per_file == 0:
                self.save_batch(current_batch, (episode + 1) // episodes_per_file)
                current_batch = []

            # Guardar la Q-table cada q_table_save_frequency episodios
            if (episode + 1) % self.q_table_save_frequency == 0:
                if hasattr(self.agent_manager, 'kp_agent') and self.agent_manager.kp_agent.q_table_filename: #Si es PID
                    self.agent_manager.kp_agent.save_q_table(self.agent_manager.kp_agent.q_table_filename)
                    if not self.agent_manager.single_q_agent: #Si tiene multiples agentes
                        self.agent_manager.ki_agent.save_q_table(self.agent_manager.ki_agent.q_table_filename)
                        self.agent_manager.kd_agent.save_q_table(self.agent_manager.kd_agent.q_table_filename)

                elif hasattr(self.agent_manager, 'q_table_filename') and self.agent_manager.q_table_filename:  # Para QLearning
                    self.agent_manager.save_q_table(self.agent_manager.q_table_filename)

        if current_batch:
            self.save_batch(current_batch, (episode + 1) // episodes_per_file + 1)

        total_time = time.time() - total_start
        self.save_metadata(total_time)


    def run_episode(self, episode: int) -> EpisodeData:
        state = self.environment.reset()
        self.controller.reset()
        self.agent_manager.reset()
        start_time = time.time()
        episode_record = EpisodeData(episode=episode, steps=[], duration=0.0, total_reward=0.0, terminated=False)
        done = False
        termination_reason = ""
        step_counter = 0
        next_state = None  # Inicializar next_state

        while not done:
            control_action = self.controller.compute(state)
            agent_actions = {}

            if step_counter % self.decision_interval == 0:
                if self.agent_manager.is_active():
                    agent_actions = self.agent_manager.decide(state)
                    self.controller.update_parameters(**agent_actions)

            next_state, _, _, info = self.environment.step(control_action)
            done = self.environment.check_termination(next_state)

            # Crear state_dict para la función de recompensa.  Incluir ganancias si son parte del estado.
            state_dict = dict(zip(self.state_vars, next_state if next_state is not None else state))  # Usa next_state si existe
            if 'kp' in self.state_vars:
                state_dict['kp'] = self.controller.kp
            if 'ki' in self.state_vars:
                state_dict['ki'] = self.controller.ki
            if 'kd' in self.state_vars:
                state_dict['kd'] = self.controller.kd

            reward = self.reward_calculator.compute_reward(state, control_action, next_state, state_dict=state_dict)

            if done:
                termination_reason = info.get("termination_reason", "")

            if self.agent_manager.is_active():
                # Pasar next_state (podría ser None si el episodio termina)
                self.agent_manager.receive_reward(reward, next_state)
                self.agent_manager.update_policy()


            step_data = StepData(state=state, control_action=control_action, agent_actions=agent_actions,
                                 reward=reward, next_state=next_state)
            episode_record.steps.append(step_data)
            episode_record.total_reward += reward

            state = next_state
            step_counter += 1

        if done and termination_reason == "stabilized":
            self.agent_manager.reward_successful_trajectory(self.success_reward_factor)

        episode_record.duration = time.time() - start_time
        episode_record.terminated = done
        episode_record.termination_reason = termination_reason
        return episode_record

    def save_batch(self, batch: List[EpisodeData], batch_number: int) -> None:
        start_save = time.time()
        filename = os.path.join(self.results_folder, f'simulation_data_{batch_number}.json')
        with open(filename, 'w') as f:
            json.dump([episode.__dict__ for episode in batch], f, indent=4, default=self.serialize)
        self.performance_data['file_save_times'].append(time.time() - start_save)

    def save_metadata(self, total_time: float) -> None:
        def calc_stats(data: List[float]) -> Dict[str, Any]:
            if not data:
                return {'min': None, 'max': None, 'mean': None, 'std': None}
            return {'min': min(data), 'max': max(data), 'mean': np.mean(data), 'std': np.std(data)}

        metadata = {
            'environment': {
                'code_version': "1.0.3",
                'timestamp': datetime.now().isoformat(),
                'n_cpu': psutil.cpu_count(),
                'n_ram': psutil.virtual_memory().total,
                'python_version': platform.python_version(),
            },
            'performance': {
                'total_simulation_time': total_time,
                'average_episode_time': calc_stats(self.performance_data['episode_times']),
                'file_saving': calc_stats(self.performance_data['file_save_times'])
            },
            'config_parameters': self.config
        }
        with open(os.path.join(self.results_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def serialize(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, StepData):
            return obj.__dict__
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")