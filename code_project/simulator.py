import time
import json
import os
import psutil
import platform
import numpy as np
from datetime import datetime
from controller import PIDController
from dynamic_system import InvertedPendulum
from reinforcement_learning import PIDQLearning
from reinforcement_learning import calculate_reward

class PendulumControl:
    def __init__(self, pendulum, q_learner, kp, ki, kd, setpoint, dt, gain_step, 
                 state_config, variable_step=False, extract_qtables=False, extract_frequency=1,
                 use_angle_limit=True, angle_limit=np.pi/3, use_cart_limit=True, cart_limit=5.0,
                 use_controller=True, use_qlearning=True, max_episodes=100, reset_gains_each_episode=True,
                 reward_weights=None, reward_scales=None, decision_interval=5.0, success_scaling_factor=1.25):
        self.pendulum = pendulum
        self.q_learner = q_learner
        self.dt = dt
        self.setpoint = setpoint
        self.max_episodes = max_episodes
        self.reset_gains_each_episode = reset_gains_each_episode
        self.initial_kp, self.initial_ki, self.initial_kd = kp, ki, kd
        self.gain_step = gain_step
        self.variable_step = variable_step
        self.use_controller = use_controller
        self.use_qlearning = use_qlearning
        self.state_config = state_config
        self.extract_qtables = extract_qtables
        self.extract_frequency = extract_frequency
        self.use_cart_limit = use_cart_limit
        self.cart_limit = cart_limit
        self.use_angle_limit = use_angle_limit
        self.angle_limit = angle_limit
        self.reward_weights = reward_weights
        self.reward_scales = reward_scales
        self.decision_interval = decision_interval
        self.stabilization_angle_threshold = 0.001  
        self.stabilization_velocity_threshold = 0.005  
        self.controller = None
        self.success_scaling_factor = success_scaling_factor

    def _init_controller(self, kp, ki, kd):
        if self.use_controller:
            return PIDController(kp, ki, kd, self.setpoint, self.dt, gain_step=0.0, variable_step=False,
                                 reset_gains_each_episode=self.reset_gains_each_episode)
        return None

    def _adjust_gain(self, current_gain, action, gain_type):
        if action == 0:
            return max(current_gain - self.gain_step, self.state_config[gain_type]['min'])
        elif action == 1:
            return current_gain
        elif action == 2:
            return min(current_gain + self.gain_step, self.state_config[gain_type]['max'])
        return current_gain

    def compute_reward(self, next_state, force, t):
        return calculate_reward(next_state[2], next_state[3], next_state[0], next_state[1],
                                  force, t, self.reward_weights, self.reward_scales)

    def _simulate_episode(self, x0, total_time, cart_limit, episode):
        episode_start_time = time.time()
        controller_time_accum = 0.0
        qlearning_time_accum = 0.0
        print(f'Episode: {episode}')
        t = 0
        x = x0.copy()
        cumulative_reward = 0
        interval_reward = 0
        next_decision_time = self.decision_interval

        kp, ki, kd = self.initial_kp, self.initial_ki, self.initial_kd
        self.controller = self._init_controller(kp, ki, kd)

        if self.use_qlearning:
            self.q_learner.reset_episode_records()

        state_values = {'angle': x[2], 'angular_velocity': x[3], 'kp': kp, 'ki': ki, 'kd': kd}

        episode_data = {
            'episode': episode,
            'time': [t],
            'cart_position': [x[0]],
            'cart_velocity': [x[1]],
            'pendulum_angle': [x[2]],
            'pendulum_velocity': [x[3]],
            'error': [x[2] - self.setpoint],
            'force': [0],
            'kp': [kp],
            'ki': [ki],
            'kd': [kd],
            'action_kp': [None],
            'action_ki': [None],
            'action_kd': [None],
            'reward': [0],
            'cumulative_reward': [0]
        }

        actions = {'kp': 1, 'ki': 1, 'kd': 1}
        done = False
        done_reason = 'Time Finished'

        while t < total_time:
            error = x[2] - self.setpoint
            controller_start = time.time()
            u = self.controller.compute(error) if self.controller else 0
            controller_end = time.time()
            controller_time_accum += (controller_end - controller_start)

            new_x = self.pendulum.update_state(t, u)
            reward = self.compute_reward(new_x, u, t)
            cumulative_reward += reward
            interval_reward += reward

            t += self.dt
            episode_data['time'].append(t)
            episode_data['cart_position'].append(new_x[0])
            episode_data['cart_velocity'].append(new_x[1])
            episode_data['pendulum_angle'].append(new_x[2])
            episode_data['pendulum_velocity'].append(new_x[3])
            episode_data['error'].append(new_x[2] - self.setpoint)
            episode_data['force'].append(u)
            episode_data['kp'].append(kp)
            episode_data['ki'].append(ki)
            episode_data['kd'].append(kd)
            episode_data['reward'].append(reward)
            episode_data['cumulative_reward'].append(cumulative_reward)
            episode_data['action_kp'].append(None)
            episode_data['action_ki'].append(None)
            episode_data['action_kd'].append(None)

            if self.use_angle_limit and abs(new_x[2]) > self.angle_limit:
                print(f"Angle limit exceeded at time {t:.3f}")
                done = True
                done_reason = 'Angle exceeded'
            if self.use_cart_limit and abs(new_x[0]) > self.cart_limit:
                print(f"Cart position limit exceeded at time {t:.3f}")
                done = True
                done_reason = 'Cart exceeded'
            if abs(new_x[2]) < self.stabilization_angle_threshold and abs(new_x[3]) < self.stabilization_velocity_threshold:
                print(f"Pendulum stabilized at time {t:.3f}")
                done = True
                done_reason = 'Succeeded stabilization'

            if t >= next_decision_time:
                next_state_values = {'angle': new_x[2], 'angular_velocity': new_x[3], 'kp': kp, 'ki': ki, 'kd': kd}
                qlearning_start = time.time()
                if self.use_qlearning:
                    self.q_learner.learn(state_values, actions, interval_reward, next_state_values, done=False)
                    self.q_learner.register_episode_decision(state_values, actions)
                if self.use_qlearning:
                    actions = self.q_learner.choose_actions(next_state_values)
                else:
                    actions = {'kp': 1, 'ki': 1, 'kd': 1}
                episode_data['action_kp'][-1] = actions['kp']
                episode_data['action_ki'][-1] = actions['ki']
                episode_data['action_kd'][-1] = actions['kd']
                kp = self._adjust_gain(kp, actions['kp'], 'kp')
                ki = self._adjust_gain(ki, actions['ki'], 'ki')
                kd = self._adjust_gain(kd, actions['kd'], 'kd')
                if self.controller:
                    self.controller.update_parameters(kp=kp, ki=ki, kd=kd)
                interval_reward = 0
                next_decision_time += self.decision_interval
                state_values = next_state_values.copy()
                qlearning_end = time.time()
                qlearning_time_accum += (qlearning_end - qlearning_start)

            x = new_x.copy()
            if done:
                next_state_values = {'angle': new_x[2], 'angular_velocity': new_x[3], 'kp': kp, 'ki': ki, 'kd': kd}
                if self.use_qlearning:
                    self.q_learner.learn(state_values, actions, interval_reward, next_state_values, done=True)
                break

        if done and done_reason == 'Succeeded stabilization' and self.use_qlearning:
            print("Applying expansion factor to episode decisions.")
            self.q_learner.apply_expansion_factor(self.success_scaling_factor)

        print(f"Episode: {episode} , Time: {t:.3f} s , Cumulative Reward: {cumulative_reward:.3f} , Reason: {done_reason}")
        pendulum_time = time.time() - episode_start_time
        episode_data['pendulum_time'] = pendulum_time
        episode_data['controller_times'] = controller_time_accum
        episode_data['qlearning_times'] = qlearning_time_accum
        if self.use_qlearning:
            self.q_learner.update_epsilon()
            self.q_learner.update_learning_rate()
            if self.extract_qtables and (episode % self.extract_frequency == 0 or episode == self.max_episodes - 1):
                episode_data['qtables'] = self.q_learner.get_qtables()
        return episode_data

class Run_Simulation:
    def __init__(self, config, pendulum_control):
        self.config = config
        self.control = pendulum_control
        self.performance_data = {
            'episode_times': [],
            'pendulum_times': [],
            'controller_times': [],
            'qlearning_times': [],
            'cpu_usages': [],
            'memory_usages': [],
            'file_save_times': []
        }
        self.simulation_data = []
        self.total_simulation_time = 0
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        results_folder = self.config['simulation'].get('results_folder', 'results_history')
        self.results_folder = os.path.join(results_folder, self.timestamp)
        os.makedirs(self.results_folder, exist_ok=True)

    def simulate_with_learning(self):
        x0 = self.config['physical']['x0']
        total_time = self.config['simulation']['total_time']
        cart_limit = self.config['simulation']['cart_limit']
        max_episodes = self.config['simulation']['max_episodes']
        for episode in range(max_episodes):
            episode_start = time.time()
            cpu_start = psutil.cpu_percent(interval=None)
            memory_start = psutil.virtual_memory().used
            episode_data = self.control._simulate_episode(x0.copy(), total_time, cart_limit, episode)
            episode_time = time.time() - episode_start
            cpu_end = psutil.cpu_percent(interval=None)
            memory_end = psutil.virtual_memory().used
            self.performance_data['episode_times'].append(episode_time)
            self.performance_data['pendulum_times'].append(episode_data.get('pendulum_time', 0))
            self.performance_data['controller_times'].append(episode_data.get('controller_times', 0))
            self.performance_data['qlearning_times'].append(episode_data.get('qlearning_times', 0))
            self.performance_data['cpu_usages'].append((cpu_start + cpu_end) / 2)
            self.performance_data['memory_usages'].append(memory_end - memory_start)
            yield episode_data

    def run(self):
        total_start = time.time()
        episodes_per_file = 100
        current_batch = []
        for i, episode_data in enumerate(self.simulate_with_learning()):
            self.simulation_data.append(episode_data)
            current_batch.append(episode_data)
            if (i + 1) % episodes_per_file == 0:
                self.save_batch(current_batch, (i + 1) // episodes_per_file)
                current_batch = []
        if current_batch:
            self.save_batch(current_batch, (i + 1) // episodes_per_file + 1)
        self.total_simulation_time = time.time() - total_start
        self.save_metadata()

    def save_batch(self, batch, batch_number):
        start_save = time.time()
        if not batch:
            return
        filename = os.path.join(self.results_folder, f'simulation_data_{batch_number}.json')
        with open(filename, 'w') as f:
            json.dump(batch, f, indent=4)
        self.performance_data['file_save_times'].append(time.time() - start_save)

    def save_metadata(self):
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
                'library_versions': {
                    'numpy': np.__version__,
                    'psutil': psutil.__version__,
                    'platform': platform.python_version(),
                }
            },
            'performance': {
                'total_simulation_time': self.total_simulation_time,
                'average_episode_time': calc_stats(self.performance_data['episode_times']),
                'pendulum_simulation': calc_stats(self.performance_data['pendulum_times']),
                'controller': calc_stats(self.performance_data['controller_times']),
                'qlearning': calc_stats(self.performance_data['qlearning_times']),
                'file_saving': calc_stats(self.performance_data['file_save_times'])
            },
            'config_parameters': self.config
        }
        with open(os.path.join(self.results_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Simulation completed. Results saved in {self.results_folder}")
