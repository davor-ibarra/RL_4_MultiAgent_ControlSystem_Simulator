from typing import Dict, Any, Optional, List
import time
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from code_project.base_interfaces import DynamicSystem, Controller, RLAgent, Environment, RewardFunction, MetricsCollector
from code_project.factories import ComponentRegistry
from code_project.config_system import ConfigurationManager, ValidationResult

@dataclass
class SimulationMetrics:
    """Container for simulation metrics."""
    episode: int
    total_reward: float
    window_rewards: List[float]
    episode_length: int
    mean_reward: float
    success_rate: float
    computation_time: float
    additional_metrics: Dict[str, Any]

class SimulationOrchestrator:
    """Orchestrates the execution of reinforcement learning simulations."""
    
    def __init__(self, config_manager: ConfigurationManager, component_registry: ComponentRegistry):
        self.config_manager = config_manager
        self.component_registry = component_registry
        self.logger = self._setup_logger()
        
        # Componentes principales
        self.system: Optional[DynamicSystem] = None
        self.controller: Optional[Controller] = None
        self.agent: Optional[RLAgent] = None
        self.environment: Optional[Environment] = None
        self.reward_function: Optional[RewardFunction] = None
        
        # Métricas y resultados
        self.metrics: List[SimulationMetrics] = []
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.results_path: Optional[Path] = None
    
    def _setup_logger(self) -> logging.Logger:
        """Configura el sistema de logging."""
        logger = logging.getLogger('SimulationOrchestrator')
        logger.setLevel(logging.INFO)
        
        # Crear el directorio de logs si no existe
        Path('logs').mkdir(exist_ok=True)
        
        # Handler para archivo
        fh = logging.FileHandler(f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        
        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def initialize(self, config_path: str) -> ValidationResult:
        """Inicializa la simulación con la configuración proporcionada."""
        self.logger.info(f"Initializing simulation with config from: {config_path}")
        
        # Cargar y validar configuración
        self.config_manager.load_config(config_path)
        validation_result = self.config_manager.validate()
        
        if not validation_result.is_valid:
            self.logger.error("Configuration validation failed:")
            for error in validation_result.errors:
                self.logger.error(f"  - {error}")
            return validation_result
        
        try:
            # Crear componentes
            self._create_components()
            
            # Crear directorio para resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_path = Path(f'results/simulation_{timestamp}')
            self.results_path.mkdir(parents=True, exist_ok=True)
            
            # Guardar configuración inicial
            with open(self.results_path / 'config.json', 'w') as f:
                json.dump(self.config_manager.get_config(), f, indent=4)
            
            self.logger.info("Simulation initialized successfully")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            return ValidationResult(False, [str(e)], [])
    
    def _create_components(self) -> None:
        """Crea las instancias de los componentes basados en la configuración."""
        config = self.config_manager.get_config()
        
        # Crear sistema dinámico
        system_config = config['system']
        self.system = self.component_registry.system_factory.create_system(
            system_config['type'],
            system_config['parameters']
        )
        
        # Crear controlador
        controller_config = config['controller']
        self.controller = self.component_registry.controller_factory.create_controller(
            controller_config['type'],
            controller_config['parameters']
        )
        
        # Crear agente
        agent_config = config['agent']
        self.agent = self.component_registry.agent_factory.create_agent(
            agent_config['type'],
            agent_config['parameters']
        )
        
        # Crear entorno
        env_config = config['environment']
        self.environment = self.component_registry.environment_factory.create_environment(
            env_config['type'],
            env_config['parameters']
        )
        
        # Crear función de recompensa
        reward_config = config['reward']
        self.reward_function = self.component_registry.reward_factory.create_reward_function(
            reward_config['type'],
            reward_config['parameters']
        )

    def run_episode(self) -> SimulationMetrics:
        """Ejecuta un episodio completo de simulación con subepisodios."""
        episode_start_time = time.time()
        total_reward = 0
        step_count = 0
        window_rewards = []
        
        # Configuración de subepisodios
        config = self.config_manager.get_config()
        decision_interval = config['simulation']['decision_interval']
        dt = config['simulation']['dt']
        steps_per_decision = int(decision_interval / dt)
        
        # Resetear entorno y obtener estado inicial
        state = self.environment.reset()
        done = False
        
        while not done:
            # Inicio de subepisodio
            window_start_state = state
            window_reward = 0
            action = self.agent.choose_action(state)
            
            # Ejecutar pasos dentro del subepisodio
            for _ in range(steps_per_decision):
                if done:
                    break
                    
                # Aplicar acción al entorno
                next_state, step_reward, done, info = self.environment.step(action)
                
                # Acumular recompensas
                window_reward += step_reward
                total_reward += step_reward
                step_count += 1
                
                # Actualizar estado
                state = next_state
                
                # Verificar límite de pasos si existe
                max_steps = config['simulation'].get('max_steps_per_episode')
                if max_steps and step_count >= max_steps:
                    done = True
                    break
            
            # Fin del subepisodio - Actualizar agente con la recompensa de la ventana
            window_rewards.append(window_reward)
            if not done:
                self.agent.learn(window_start_state, action, window_reward, state, False)
            else:
                self.agent.learn(window_start_state, action, window_reward, state, True)
        
        # Calcular métricas del episodio
        computation_time = time.time() - episode_start_time
        mean_reward = total_reward / step_count if step_count > 0 else 0
        
        metrics = SimulationMetrics(
            episode=self.current_episode,
            total_reward=total_reward,
            window_rewards=window_rewards,
            episode_length=step_count,
            mean_reward=mean_reward,
            success_rate=1.0 if total_reward > self.best_reward else 0.0,
            computation_time=computation_time,
            additional_metrics=info
        )
        
        # Actualizar mejor recompensa
        self.best_reward = max(self.best_reward, total_reward)
        
        # Guardar métricas
        self.metrics.append(metrics)
        
        # Guardar resultados si es necesario
        if self.current_episode % self.config_manager.get_config()['simulation']['save_frequency'] == 0:
            self.save_results()
        
        self.current_episode += 1
        return metrics
    
    def run_simulation(self) -> List[SimulationMetrics]:
        """Ejecuta la simulación completa."""
        self.logger.info("Starting simulation")
        
        config = self.config_manager.get_config()
        max_episodes = config['simulation']['max_episodes']
        
        try:
            for episode in range(max_episodes):
                self.current_episode = episode
                metrics = self.run_episode()
                
                # Log del progreso
                self.logger.info(
                    f"Episode {episode}/{max_episodes} - "
                    f"Total Reward: {metrics.total_reward:.2f} - "
                    f"Steps: {metrics.episode_length} - "
                    f"Mean Window Reward: {np.mean(metrics.window_rewards):.2f}"
                )
                
                # Verificar condiciones de terminación temprana
                if self._check_early_stopping():
                    self.logger.info("Early stopping conditions met")
                    break
            
            self.save_results()
            self.logger.info("Simulation completed successfully")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            self.save_results()  # Guardar resultados parciales
            raise
    
    def _check_early_stopping(self) -> bool:
        """Verifica condiciones de terminación temprana."""
        if len(self.metrics) < 2:
            return False
        
        config = self.config_manager.get_config()
        early_stopping = config['simulation'].get('early_stopping', {})
        
        if not early_stopping:
            return False
        
        # Verificar convergencia de recompensa
        if early_stopping.get('reward_convergence'):
            window = early_stopping['window']
            threshold = early_stopping['threshold']
            
            if len(self.metrics) >= window:
                recent_rewards = [m.total_reward for m in self.metrics[-window:]]
                std_reward = np.std(recent_rewards)
                if std_reward < threshold:
                    return True
        
        return False
    
    def save_results(self) -> None:
        """Guarda los resultados de la simulación."""
        if not self.results_path:
            return
        
        # Guardar métricas
        metrics_path = self.results_path / f'metrics_episode_{self.current_episode}.json'
        with open(metrics_path, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=4)
        
        # Guardar estado del agente
        agent_path = self.results_path / f'agent_state_episode_{self.current_episode}'
        self.agent.save_state(str(agent_path))
        
        # Guardar configuración actual
        config_path = self.results_path / f'config_episode_{self.current_episode}.json'
        with open(config_path, 'w') as f:
            json.dump(self.config_manager.get_config(), f, indent=4)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Carga un checkpoint de la simulación."""
        checkpoint_path = Path(checkpoint_path)
        
        # Cargar configuración
        with open(checkpoint_path / 'config.json', 'r') as f:
            config = json.load(f)
        self.config_manager.set_config(config)
        
        # Recrear componentes
        self._create_components()
        
        # Cargar estado del agente
        self.agent.load_state(str(checkpoint_path / 'agent_state'))
        
        # Cargar métricas
        with open(checkpoint_path / 'metrics.json', 'r') as f:
            metrics_data = json.load(f)
            self.metrics = [SimulationMetrics(**m) for m in metrics_data]
        
        self.current_episode = len(self.metrics)
        self.best_reward = max(m.total_reward for m in self.metrics) if self.metrics else float('-inf')
