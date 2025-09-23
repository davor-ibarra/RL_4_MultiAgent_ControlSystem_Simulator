from interfaces.environment import Environment # Importar Interfaz
from interfaces.dynamic_system import DynamicSystem # Type hint
from interfaces.controller import Controller     # Type hint
from interfaces.rl_agent import RLAgent          # Type hint
from interfaces.reward_function import RewardFunction # Type hint

import numpy as np
import logging # Import logging
from typing import Tuple, Dict, Any, Optional

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PendulumEnvironment(Environment): # Implementar Interfaz Environment
    """
    Implementación del entorno de simulación para el péndulo invertido.
    Utiliza componentes inyectados (System, Controller, Agent, RewardFunction)
    para gestionar la dinámica, control, recompensa y estado del agente.
    """
    def __init__(self,
                 # --- Dependencias Inyectadas ---
                 system: DynamicSystem,
                 controller: Controller,
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 # --- Parámetros desde Config ---
                 dt: float,
                 reset_gains: bool,
                 config: Dict[str, Any] # Config completa para acceso interno
                 ):
        """
        Inicializa el entorno del péndulo invertido. Recibe componentes vía inyección.

        Args:
            system: Instancia del sistema dinámico (InvertedPendulumSystem).
            controller: Instancia del controlador (PIDController).
            agent: Instancia del agente RL (PIDQLearningAgent).
            reward_function: Instancia de la función de recompensa.
            dt: Paso de tiempo de la simulación.
            reset_gains: Flag para indicar si resetear las ganancias del controlador
                         al inicio de cada episodio.
            config: Diccionario de configuración principal.
        """
        logger.info("Inicializando PendulumEnvironment...")
        self.system = system
        self.controller = controller
        self.agent = agent
        self.reward_function = reward_function
        self.dt = dt
        self.reset_gains = reset_gains
        self.config = config # Guardar config para usar en check_termination
        self.state: Optional[np.ndarray] = None # Estado actual del sistema
        self.t: float = 0.0 # Tiempo actual de la simulación dentro del episodio

        # Validar dependencias básicas (ya no necesario validar tipos aquí si DI lo hace)
        # if not all(isinstance(comp, expected_type) for comp, expected_type in [
        #     (system, DynamicSystem), (controller, Controller), ...
        # ]): ...

        if not isinstance(dt, (float, int)) or dt <= 0:
            logger.error(f"dt inválido ({dt}) proporcionado a PendulumEnvironment.")
            raise ValueError("dt debe ser un número positivo.")

        logger.info("PendulumEnvironment inicializado exitosamente.")


    def step(self) -> Tuple[Any, Tuple[float, float], Any]:
        """
        Avanza la simulación un paso de tiempo dt. Implementa método de interfaz.
        """
        if self.state is None:
            msg = "Environment.step() llamado antes de reset()."
            logger.error(msg)
            raise RuntimeError(msg)

        force = 0.0; reward = 0.0; stability_score = 0.0
        try:
            # 1. Calcular Acción de Control
            force = self.controller.compute_action(self.state)

            # 2. Aplicar Acción al Sistema Dinámico
            next_state_vector = self.system.apply_action(self.state, force, self.t, self.dt)

            # 3. Calcular Recompensa y Estabilidad
            # Pasar self.state (estado ANTES de la acción) y next_state_vector
            reward, stability_score = self.reward_function.calculate(self.state, force, next_state_vector, self.t)

            # 4. Actualizar Estado Interno y Tiempo
            self.state = np.array(next_state_vector) # Asegurar numpy array
            self.t += self.dt

            # Asegurar valores finitos antes de devolver
            reward = float(reward) if np.isfinite(reward) else 0.0
            stability_score = float(stability_score) if np.isfinite(stability_score) else 0.0
            force = float(force) if np.isfinite(force) else 0.0

            return self.state, (reward, stability_score), force

        except Exception as e:
             logger.error(f"Error crítico durante environment.step() en t={self.t:.4f}: {e}", exc_info=True)
             # Devolver estado actual y valores neutros/malos? O relanzar?
             # Relanzar permite al SimulationManager manejar el fallo.
             raise RuntimeError(f"Fallo en environment step a t={self.t:.4f}") from e


    def reset(self, initial_conditions: Any) -> Any:
        """Resetea el estado del sistema, tiempo, etc. Implementa método de interfaz."""
        # ... (código sin cambios funcionales) ...
        logger.debug(f"Reseteando PendulumEnvironment con condiciones iniciales: {initial_conditions}")
        try:
            # Resetear sistema dinámico
            self.state = self.system.reset(initial_conditions)
            self.t = 0.0
            # Resetear controlador (ganancias o solo estado interno)
            if self.reset_gains: self.controller.reset()
            else: self.controller.reset_internal_state()
            # Resetear agente (epsilon/alpha decay)
            self.agent.reset_agent()
            logger.debug(f"Estado inicial tras reset: {np.round(self.state, 4)}")
            return self.state
        except Exception as e:
            logger.error(f"Error crítico durante environment.reset(): {e}", exc_info=True)
            raise RuntimeError(f"Fallo crítico durante el reseteo del entorno: {e}") from e


    def check_termination(self, config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Verifica condiciones de terminación. Implementa método de interfaz."""
        # ... (código sin cambios funcionales, usa self.config interno) ...
        if self.state is None or len(self.state) < 4:
            logger.warning("check_termination llamado con estado inválido.")
            return False, False, False

        sim_config = self.config.get('simulation', {})
        stab_config = self.config.get('stabilization_criteria', {})
        ctrl_params = self.config.get('environment', {}).get('controller', {}).get('params', {})
        setpoint = ctrl_params.get('setpoint', 0.0)

        angle_limit = sim_config.get('angle_limit', 1.0)
        use_angle_limit = sim_config.get('use_angle_limit', True)
        angle_exceeded = use_angle_limit and (abs(self.state[2]) > angle_limit)

        cart_limit = sim_config.get('cart_limit', 5.0)
        use_cart_limit = sim_config.get('use_cart_limit', True)
        cart_exceeded = use_cart_limit and (abs(self.state[0]) > cart_limit)

        stabilized = False
        if stab_config:
            angle_threshold = stab_config.get('angle_threshold', 0.01)
            velocity_threshold = stab_config.get('velocity_threshold', 0.01)
            angle_stable = abs(self.state[2] - setpoint) < angle_threshold
            velocity_stable = abs(self.state[3]) < velocity_threshold
            stabilized = angle_stable and velocity_stable

        # logger.debug(f"Term Check: AngleEx={angle_exceeded}, CartEx={cart_exceeded}, Stabilized={stabilized}")
        return angle_exceeded, cart_exceeded, stabilized


    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delega la actualización a RewardFunction. Implementa método de interfaz."""
        # ... (código sin cambios funcionales) ...
        if hasattr(self.reward_function, 'update_calculator_stats'):
            try:
                # logger.debug(f"Delegando actualización de stats a {type(self.reward_function).__name__} post episodio {current_episode}.")
                self.reward_function.update_calculator_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logger.error(f"Error llamando update_calculator_stats en RewardFunction: {e}", exc_info=True)
        # else: logger.debug("RewardFunction no tiene método update_calculator_stats.")