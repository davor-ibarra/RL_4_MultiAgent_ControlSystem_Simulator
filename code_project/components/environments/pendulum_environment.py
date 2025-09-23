from interfaces.environment import Environment # Importar Interfaz
from interfaces.dynamic_system import DynamicSystem # Type hint
from interfaces.controller import Controller     # Type hint
from interfaces.rl_agent import RLAgent          # Type hint
from interfaces.reward_function import RewardFunction # Type hint

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

# 6.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class PendulumEnvironment(Environment): # Implementar Interfaz Environment
    """
    Entorno de simulación para el péndulo invertido. Implementa Environment.
    Utiliza componentes inyectados (System, Controller, Agent, RewardFunction).
    """
    def __init__(self,
                 # --- Dependencias Inyectadas ---
                 system: DynamicSystem,
                 controller: Controller,
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 # --- Parámetros desde Config ---
                 dt: float,
                 reset_gains: bool, # Si resetear ganancias PID cada episodio
                 config: Dict[str, Any] # Config completa para acceso interno
                 ):
        """
        Inicializa el entorno del péndulo invertido.

        Args:
            system: Instancia del sistema dinámico.
            controller: Instancia del controlador.
            agent: Instancia del agente RL.
            reward_function: Instancia de la función de recompensa.
            dt: Paso de tiempo de la simulación.
            reset_gains: Flag para resetear ganancias del controlador.
            config: Diccionario de configuración principal.

        Raises:
            TypeError: Si las dependencias inyectadas no son del tipo esperado.
            ValueError: Si dt no es válido.
        """
        logger.info("Inicializando PendulumEnvironment...")
        # 6.2: Validar tipos de dependencias inyectadas (Fail-Fast)
        if not isinstance(system, DynamicSystem): raise TypeError("system debe implementar DynamicSystem")
        if not isinstance(controller, Controller): raise TypeError("controller debe implementar Controller")
        if not isinstance(agent, RLAgent): raise TypeError("agent debe implementar RLAgent")
        if not isinstance(reward_function, RewardFunction): raise TypeError("reward_function debe implementar RewardFunction")

        self.system = system
        self.controller = controller
        self.agent = agent
        self.reward_function = reward_function

        # 6.3: Validar dt (Fail-Fast)
        if not isinstance(dt, (float, int)) or dt <= 0:
            raise ValueError(f"dt inválido ({dt}) proporcionado a PendulumEnvironment.")
        self._dt = dt # Almacenar dt internamente

        self.reset_gains = bool(reset_gains)
        self.config = config # Guardar config para usar en check_termination
        self.state: Optional[np.ndarray] = None # Estado actual del sistema
        self.t: float = 0.0 # Tiempo actual dentro del episodio

        logger.info(f"PendulumEnvironment inicializado con dt={self._dt}, reset_gains={self.reset_gains}.")

    # 6.4: Exponer dt como propiedad (útil para SimulationManager)
    @property
    def dt(self) -> float:
        """Devuelve el paso de tiempo del entorno."""
        return self._dt

    def step(self) -> Tuple[Any, Tuple[float, float], Any]:
        """Avanza la simulación un paso dt."""
        if self.state is None:
            # Fail-Fast si se llama antes de reset
            msg = "Environment.step() llamado antes de reset()."
            logger.critical(msg)
            raise RuntimeError(msg)

        current_state_copy = np.copy(self.state) # Copiar estado actual para cálculo de recompensa

        try:
            # 1. Calcular Acción de Control (usando interfaz)
            force = self.controller.compute_action(current_state_copy)
            # Asegurar que la fuerza es finita
            force = float(force) if np.isfinite(force) else 0.0

            # 2. Aplicar Acción al Sistema Dinámico (usando interfaz)
            # El sistema maneja errores internos y devuelve estado válido
            next_state_vector = self.system.apply_action(current_state_copy, force, self.t, self._dt)

            # 3. Calcular Recompensa y Estabilidad (usando interfaz)
            # Pasar estado *antes* (current_state_copy) y estado *después* (next_state_vector)
            # La función calculate debe devolver valores finitos
            reward, stability_score = self.reward_function.calculate(
                current_state_copy, force, next_state_vector, self.t
            )

            # 4. Actualizar Estado Interno y Tiempo
            self.state = np.array(next_state_vector) # Asegurar numpy array
            self.t += self._dt

            reward_f = float(reward) if np.isfinite(reward) else 0.0
            stability_score_f = float(stability_score) if np.isfinite(stability_score) else 0.0

            # Devolver estado, (reward, w_stab), info(force)
            return self.state, (reward_f, stability_score_f), force

        except Exception as e:
             # Capturar errores inesperados durante el paso
             logger.critical(f"Error CRÍTICO durante environment.step() en t={self.t:.4f}: {e}", exc_info=True)
             # Relanzar como RuntimeError para que SimulationManager lo maneje
             raise RuntimeError(f"Fallo en environment step a t={self.t:.4f}") from e


    def reset(self, initial_conditions: Any) -> Any:
        """Resetea el entorno, sistema, controlador y agente."""
        logger.debug(f"Reseteando PendulumEnvironment con condiciones iniciales: {initial_conditions}")
        try:
            # Resetear sistema dinámico (valida y normaliza estado)
            self.state = self.system.reset(initial_conditions)
            self.t = 0.0

            # Resetear controlador (según config reset_gains)
            if self.reset_gains:
                self.controller.reset() # Resetea ganancias y estado interno
            else:
                self.controller.reset_internal_state() # Resetea solo estado interno

            # Resetear agente (epsilon/alpha decay)
            self.agent.reset_agent()

            #logger.debug(f"Estado inicial tras reset: {np.round(self.state, 4)}")
            return np.copy(self.state) # Devolver copia del estado inicial

        except Exception as e:
            # Capturar errores críticos durante el reset
            logger.critical(f"Error crítico durante environment.reset(): {e}", exc_info=True)
            raise RuntimeError(f"Fallo crítico durante el reseteo del entorno: {e}") from e


    def check_termination(self, config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Verifica condiciones de terminación basadas en config y estado actual."""
        # 6.5: Usar config pasada como argumento (o self.config si se prefiere)
        #      Validar estado interno.
        if self.state is None or len(self.state) < 4:
            logger.warning("check_termination llamado con estado inválido (None o corto).")
            return False, False, False # No terminar si estado inválido

        sim_config = config.get('simulation', {})
        stab_config = config.get('stabilization_criteria', {})
        env_config = config.get('environment', {})
        ctrl_params = env_config.get('controller', {}).get('params', {})
        setpoint = ctrl_params.get('setpoint', 0.0)

        # Límites de estado
        angle_limit = sim_config.get('angle_limit', np.pi / 2.0) # Default a 90 grados
        use_angle_limit = sim_config.get('use_angle_limit', True)
        angle_exceeded = use_angle_limit and (abs(self.state[2]) > angle_limit)

        cart_limit = sim_config.get('cart_limit', 2.4) # Default a límites estándar Gym
        use_cart_limit = sim_config.get('use_cart_limit', True)
        cart_exceeded = use_cart_limit and (abs(self.state[0]) > cart_limit)

        limit_exceeded = angle_exceeded or cart_exceeded

        # Criterio de estabilización (si está configurado)
        stabilized = False
        if isinstance(stab_config, dict) and stab_config: # Verificar que sea dict y no vacío
            angle_threshold = stab_config.get('angle_threshold', 0.05) # ~3 grados
            velocity_threshold = stab_config.get('velocity_threshold', 0.05)
            # Comprobar si ambos (ángulo y velocidad angular) están cerca del setpoint (0)
            angle_stable = abs(self.state[2] - setpoint) < angle_threshold
            velocity_stable = abs(self.state[3]) < velocity_threshold # Velocidad angular
            stabilized = angle_stable and velocity_stable

        # logger.debug(f"Term Check: LimitEx={limit_exceeded} (Angle={angle_exceeded}, Cart={cart_exceeded}), GoalReached={stabilized}")
        # Devolver: (límite excedido, meta alcanzada, otra_condicion=False)
        return limit_exceeded, stabilized, False


    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delega la actualización de stats a RewardFunction."""
        # 6.6: Usar interfaz RewardFunction (ya validada en init)
        try:
            # logger.debug(f"Delegando actualización de stats a {type(self.reward_function).__name__}")
            self.reward_function.update_calculator_stats(episode_metrics_dict, current_episode)
        except Exception as e:
            logger.error(f"Error llamando update_calculator_stats en RewardFunction: {e}", exc_info=True)
            # No relanzar, es una operación de soporte