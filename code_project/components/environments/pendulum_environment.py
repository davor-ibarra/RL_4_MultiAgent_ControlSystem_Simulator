from interfaces.environment import Environment
from interfaces.dynamic_system import DynamicSystem # Type hint
from interfaces.controller import Controller     # Type hint
from interfaces.rl_agent import RLAgent          # Type hint
from interfaces.reward_function import RewardFunction # Type hint

import numpy as np
import logging # Import logging
from typing import Tuple, Dict, Any, Optional

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class PendulumEnvironment(Environment):
    """
    Implementación del entorno de simulación para el péndulo invertido.
    Utiliza componentes inyectados (System, Controller, Agent, RewardFunction)
    para gestionar la dinámica, control, recompensa y estado del agente.
    """
    def __init__(self,
                 system: DynamicSystem,
                 controller: Controller,
                 agent: RLAgent,
                 reward_function: RewardFunction,
                 dt: float,
                 reset_gains: bool,
                 config: Dict[str, Any] # Config completa para acceso a límites, etc.
                 ):
        """
        Inicializa el entorno del péndulo invertido.

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

        # Validar dependencias básicas
        if not all(isinstance(comp, expected_type) for comp, expected_type in [
            (system, DynamicSystem), (controller, Controller),
            (agent, RLAgent), (reward_function, RewardFunction)
        ]):
            logger.error("PendulumEnvironment recibió un componente con tipo incorrecto.")
            raise TypeError("Tipo de componente inválido proporcionado a PendulumEnvironment.")
        if not isinstance(dt, (float, int)) or dt <= 0:
            logger.error(f"dt inválido ({dt}) proporcionado a PendulumEnvironment.")
            raise ValueError("dt debe ser un número positivo.")

        logger.info("PendulumEnvironment inicializado exitosamente.")


    def step(self) -> Tuple[Any, Tuple[float, float], Any]:
        """
        Avanza la simulación un paso de tiempo dt. Calcula la acción de control,
        aplica la dinámica del sistema y calcula la recompensa/estabilidad.

        Returns:
            Tuple[Any, Tuple[float, float], Any]:
              (next_state_vector, (reward, stability_score), control_force)
        """
        if self.state is None:
            logger.error("Environment.step() llamado antes de reset().")
            # Intentar auto-resetear si es posible desde config
            initial_conditions = self.config.get('initial_conditions', {}).get('x0')
            if initial_conditions:
                self.reset(initial_conditions)
                logger.warning("Environment fue auto-reseteado en step() usando config.")
            else:
                raise RuntimeError("Environment.step() llamado antes de reset() y no se pudo auto-resetear.")

        # 1. Calcular Acción de Control usando el estado actual y las ganancias *actuales* del controlador
        try:
            # El SimulationManager ya actualizó las ganancias del controlador *antes* de este paso
            # si se tomó una acción de cambio de ganancia.
            force = self.controller.compute_action(self.state)
        except Exception as e:
            logger.error(f"Error calculando acción de control en t={self.t:.4f}: {e}", exc_info=True)
            force = 0.0 # Aplicar acción neutral si el controlador falla

        # 2. Aplicar Acción al Sistema Dinámico para obtener el siguiente estado
        try:
            next_state_vector = self.system.apply_action(self.state, force, self.t, self.dt)
        except Exception as e:
            logger.error(f"Error aplicando acción al sistema dinámico en t={self.t:.4f}: {e}", exc_info=True)
            # ¿Qué hacer si la dinámica falla? ¿Usar estado anterior? ¿Terminar episodio?
            # Usar estado anterior puede llevar a bucles o comportamiento extraño.
            # Devolver el estado actual y quizás una recompensa muy negativa.
            next_state_vector = self.state # Mantener estado actual
            # Podríamos forzar done=True aquí o retornar un flag de error

        # 3. Calcular Recompensa y Puntuación de Estabilidad usando la RewardFunction
        reward = 0.0
        stability_score = 1.0 # Default si falla el cálculo
        try:
            # La función calculate devuelve (reward, stability_score)
            reward, stability_score = self.reward_function.calculate(self.state, force, next_state_vector, self.t)
        except Exception as e:
            logger.error(f"Error calculando recompensa/estabilidad en t={self.t:.4f}: {e}", exc_info=True)
            reward = 0.0 # Usar recompensa neutral o negativa en caso de error
            stability_score = 0.0 # Usar estabilidad mínima en caso de error

        # 4. Actualizar Estado Interno y Tiempo del Entorno
        self.state = np.array(next_state_vector) # Asegurar que es numpy array
        self.t += self.dt

        # Devolver la tupla requerida por la interfaz
        return self.state, (reward, stability_score), force


    def reset(self, initial_conditions: Any) -> Any:
        """Resetea el estado del sistema, tiempo, y opcionalmente el controlador y agente."""
        logger.debug(f"Reseteando PendulumEnvironment con condiciones iniciales: {initial_conditions}")
        try:
            # Resetear sistema dinámico
            self.state = np.array(self.system.reset(initial_conditions))
            self.t = 0.0

            # Resetear controlador (ganancias o solo estado interno)
            if self.reset_gains:
                self.controller.reset() # Resetea ganancias y estado interno
                logger.debug("Controlador reseteado (ganancias y estado interno).")
            else:
                self.controller.reset_internal_state() # Resetea solo errores/integral
                logger.debug("Estado interno del controlador reseteado (ganancias mantenidas).")

            # Resetear agente (e.g., decaimiento de epsilon/alpha)
            self.agent.reset_agent()
            logger.debug(f"Agente reseteado (epsilon={self.agent.epsilon:.4f}, LR={self.agent.learning_rate:.4f}).")

            logger.debug(f"Estado inicial tras reset: {np.round(self.state, 4)}")
            return self.state

        except Exception as e:
            logger.error(f"Error crítico durante environment.reset(): {e}", exc_info=True)
            # Es importante relanzar para que SimulationManager sepa que falló
            raise RuntimeError(f"Fallo crítico durante el reseteo del entorno: {e}") from e


    def check_termination(self, config: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """Verifica condiciones de terminación (límites de ángulo/carro, estabilización)."""
        # Validar estado actual
        if self.state is None or len(self.state) < 4:
            logger.warning("check_termination llamado con estado inválido. Devolviendo (False, False, False).")
            return False, False, False # No terminar si el estado es inválido

        # Extraer configuraciones relevantes de forma segura
        sim_config = config.get('simulation', {})
        stab_config = config.get('stabilization_criteria', {})
        # Setpoint se usa para criterio de estabilización
        ctrl_params = config.get('environment', {}).get('controller', {}).get('params', {})
        setpoint = ctrl_params.get('setpoint', 0.0)

        # --- Comprobación de Límites ---
        angle_limit = sim_config.get('angle_limit', 1.0) # Default a ~60 grados si no está
        use_angle_limit = sim_config.get('use_angle_limit', True)
        angle_exceeded = use_angle_limit and (abs(self.state[2]) > angle_limit)

        cart_limit = sim_config.get('cart_limit', 5.0)
        use_cart_limit = sim_config.get('use_cart_limit', True)
        cart_exceeded = use_cart_limit and (abs(self.state[0]) > cart_limit)

        # --- Comprobación de Estabilización ---
        stabilized = False
        if stab_config: # Solo comprobar si la sección existe
            angle_threshold = stab_config.get('angle_threshold', 0.01) # Default pequeño
            velocity_threshold = stab_config.get('velocity_threshold', 0.01) # Default pequeño

            # Comprobar si ángulo y velocidad angular están cerca de cero (o setpoint)
            angle_stable = abs(self.state[2] - setpoint) < angle_threshold
            velocity_stable = abs(self.state[3]) < velocity_threshold # Velocidad angular (índice 3)

            stabilized = angle_stable and velocity_stable
            # logger.debug(f"Stab Check: Angle OK={angle_stable} (Err={abs(self.state[2]-setpoint):.4f}<{angle_threshold}), Vel OK={velocity_stable} (Vel={abs(self.state[3]):.4f}<{velocity_threshold}) -> Stabilized={stabilized}")


        # logger.debug(f"Termination Check: AngleEx={angle_exceeded}, CartEx={cart_exceeded}, Stabilized={stabilized}")
        return angle_exceeded, cart_exceeded, stabilized


    def update_reward_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delega la actualización de estadísticas a la función de recompensa."""
        # logger.debug(f"Solicitando actualización de stats del reward calculator tras episodio {current_episode}.")
        # Verificar si el método existe antes de llamar
        if hasattr(self.reward_function, 'update_calculator_stats'):
            try:
                self.reward_function.update_calculator_stats(episode_metrics_dict, current_episode)
                # logger.debug("Llamada a reward_function.update_calculator_stats realizada.")
            except Exception as e:
                logger.error(f"Error llamando a update_calculator_stats en la función de recompensa: {e}", exc_info=True)
        else:
            logger.debug("La función de recompensa no tiene método 'update_calculator_stats'.")