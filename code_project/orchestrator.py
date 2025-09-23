import time
import json
import os
import psutil
import platform
import numpy as np
from datetime import datetime

class SimulationOrchestrator:
    def __init__(self, environment, controller, agent_manager, reward_calculator, metrics_collector, config):
        self.environment = environment
        self.controller = controller
        self.agent_manager = agent_manager  # Puede ser un gestor de agentes (multiagente)
        self.reward_calculator = reward_calculator
        self.metrics_collector = metrics_collector
        self.config = config
        self.episodes = config['simulation']['max_episodes']
        self.results_folder = os.path.join(config['simulation'].get('results_folder', 'results_history'),
                                           datetime.now().strftime('%Y%m%d-%H%M'))
        os.makedirs(self.results_folder, exist_ok=True)
        self.performance_data = {
            'episode_times': [],
            'cpu_usages': [],
            'memory_usages': [],
            'file_save_times': []
        }
        self.simulation_data = []

    def run_simulation(self):
        total_start = time.time()
        episodes_per_file = 100
        current_batch = []
        for episode in range(self.episodes):
            episode_data = self.run_episode(episode)
            current_batch.append(episode_data)
            if (episode + 1) % episodes_per_file == 0:
                self.save_batch(current_batch, (episode + 1) // episodes_per_file)
                current_batch = []
        if current_batch:
            self.save_batch(current_batch, (episode + 1) // episodes_per_file + 1)
        total_time = time.time() - total_start
        self.save_metadata(total_time)

    def run_episode(self, episode):
        self.environment.reset()
        self.agent_manager.reset_agents()
        start_time = time.time()
        state = self.environment.get_state()
        episode_record = {'episode': episode, 'steps': []}
        done = False
        while not done:
            control_action = self.controller.compute(state)
            agent_actions = self.agent_manager.get_actions(state)
            # El controlador puede actualizarse con la sugerencia del agente
            self.controller.update_parameters(**agent_actions)
            next_state = self.environment.step(control_action)
            reward = self.reward_calculator.compute_reward(state, control_action, next_state)
            self.agent_manager.receive_reward(reward, next_state)
            episode_record['steps'].append({
                'state': state,
                'control_action': control_action,
                'agent_actions': agent_actions,
                'reward': reward,
                'next_state': next_state
            })
            state = next_state
            done = self.environment.check_termination(state)
        episode_record['duration'] = time.time() - start_time
        return episode_record

    def save_batch(self, batch, batch_number):
        start_save = time.time()
        filename = os.path.join(self.results_folder, f'simulation_data_{batch_number}.json')
        with open(filename, 'w') as f:
            json.dump(batch, f, indent=4)
        self.performance_data['file_save_times'].append(time.time() - start_save)

    def save_metadata(self, total_time):
        def calc_stats(data):
            if not data:
                return {'min': None, 'max': None, 'mean': None, 'std': None}
            return {'min': min(data), 'max': max(data), 'mean': np.mean(data), 'std': np.std(data)}
        metadata = {
            'environment': {
                'code_version': "1.0.0",
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
