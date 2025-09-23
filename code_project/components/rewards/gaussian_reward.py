import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction
from interfaces.stability_calculator import BaseStabilityCalculator # Importar interfaz base

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class GaussianReward(RewardFunction):
    """
    Calcula la recompensa instantánea y la puntuación de estabilidad.
    Puede operar en dos modos según la configuración:
    1. 'gaussian': Calcula la recompensa como una suma ponderada de funciones
                   Gaussianas aplicadas a las variables de estado y acción.
    2. 'stability_calculator': Delega el cálculo de la recompensa a un
                               StabilityCalculator inyectado (si existe).

    Independientemente del modo de recompensa, si se proporciona un StabilityCalculator,
    se usa para calcular la puntuación de estabilidad (w_stab).
    """
    def __init__(self,
                 init_config: Dict[str, Any], # Config específica para esta clase
                 stability_calculator: Optional[BaseStabilityCalculator] = None
                 ):
        """
        Inicializa la función de recompensa Gaussiana/Basada en Estabilidad.

        Args:
            init_config (Dict[str, Any]): Diccionario de configuración específico,
                                          espera claves como 'params' (con 'weights', 'scales'),
                                          'use_stability_based_reward', 'stability_calculator_config'.
                                          (Normalmente construido por RewardFactory).
            stability_calculator (Optional[BaseStabilityCalculator]): Instancia del
                                       calculador de estabilidad (puede ser None).
        """
        logger.info("Inicializando GaussianReward...")
        try:
            # --- Procesar Configuración ---
            self.params = init_config.get('params', {})
            self.weights = self.params.get('weights', {}) # Pesos para modo Gaussiano
            self.scales = self.params.get('scales', {})   # Escalas para modo Gaussiano

            self.stability_calculator = stability_calculator
            # Determinar modo de cálculo de recompensa
            self.use_stability_reward = init_config.get('use_stability_based_reward', False)
            # Guardar config del stability calculator (no se usa activamente aquí, pero podría ser útil)
            self.stability_calculator_config = init_config.get('stability_calculator_config', {})

            # --- Mapeo de Estado (para modo Gaussiano) ---
            self.state_indices = {
                'cart_position': 0, 'cart_velocity': 1,
                'angle': 2, 'angular_velocity': 3
            }
            # Claves esperadas en weights/scales si se usa el modo Gaussiano
            self.required_gaussian_keys = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity', 'force', 'time']

            # --- Validaciones ---
            if self.use_stability_reward:
                # Si se pide recompensa basada en estabilidad, el calculador DEBE existir
                if self.stability_calculator is None:
                    logger.error("GaussianReward CRITICAL: Configurado para usar recompensa de estabilidad, ¡pero no se proporcionó StabilityCalculator!")
                    # Forzar a modo Gaussiano como fallback MUY POBRE, idealmente esto falla en la factory
                    self.use_stability_reward = False
                    logger.warning("Fallback a modo de recompensa Gaussiano debido a falta de StabilityCalculator.")
                    self._validate_gaussian_params() # Validar params gaussianos si hacemos fallback
                else:
                    # Verificar que el calculador proporcionado tenga el método necesario
                    if not hasattr(self.stability_calculator, 'calculate_stability_based_reward'):
                        logger.error(f"GaussianReward CRITICAL: StabilityCalculator ({type(self.stability_calculator).__name__}) "
                                     f"no tiene método 'calculate_stability_based_reward'.")
                        self.use_stability_reward = False # Fallback
                        logger.warning("Fallback a modo de recompensa Gaussiano.")
                        self._validate_gaussian_params()
                    else:
                        logger.info(f"GaussianReward configurado para usar recompensa desde: {type(self.stability_calculator).__name__}")
            else:
                # Si se usa modo Gaussiano, validar pesos y escalas
                logger.info("GaussianReward configurado para usar cálculo Gaussiano.")
                self._validate_gaussian_params()

            # Advertir si no hay calculador de estabilidad para w_stab
            if self.stability_calculator is None:
                logger.warning("GaussianReward: No se proporcionó StabilityCalculator. Puntuación de estabilidad (w_stab) será siempre 1.0.")
            elif not hasattr(self.stability_calculator, 'calculate_instantaneous_stability'):
                 logger.error(f"GaussianReward CRITICAL: StabilityCalculator ({type(self.stability_calculator).__name__}) "
                               f"no tiene método 'calculate_instantaneous_stability'. w_stab será siempre 1.0.")
                 self.stability_calculator = None # Anularlo si no tiene el método


        except Exception as e:
            logger.error(f"GaussianReward: Error durante inicialización: {e}", exc_info=True)
            raise RuntimeError("Fallo al inicializar GaussianReward") from e

    def _validate_gaussian_params(self):
        """Valida la presencia de claves y valores en weights y scales para modo Gaussiano."""
        logger.debug("Validando parámetros para modo Gaussiano...")
        valid = True
        for key in self.required_gaussian_keys:
            # Validar pesos
            if key not in self.weights:
                logger.warning(f"Modo Gaussiano: Falta peso para '{key}'. Usando 0.0.")
                self.weights[key] = 0.0
            elif not isinstance(self.weights[key], (int, float)):
                logger.warning(f"Modo Gaussiano: Peso para '{key}' ({self.weights[key]}) no es numérico. Usando 0.0.")
                self.weights[key] = 0.0

            # Validar escalas (deben ser positivas)
            if key not in self.scales:
                logger.warning(f"Modo Gaussiano: Falta escala para '{key}'. Usando 1.0.")
                self.scales[key] = 1.0
            elif not isinstance(self.scales[key], (int, float)) or self.scales[key] <= 0:
                logger.warning(f"Modo Gaussiano: Escala para '{key}' ({self.scales[key]}) no es número positivo. Usando 1.0.")
                self.scales[key] = 1.0
        logger.debug(f"Pesos Gaussianos finales: {self.weights}")
        logger.debug(f"Escalas Gaussianas finales: {self.scales}")


    # --- Implementación de Métodos de la Interfaz ---

    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """Calcula recompensa y w_stab según el modo configurado y el stability_calculator."""

        stability_score = 1.0 # Valor por defecto
        reward = 0.0        # Valor por defecto

        # --- [1] Calcular Puntuación de Estabilidad (w_stab) ---
        # Siempre intentar calcular si hay un calculador válido
        if self.stability_calculator:
            try:
                # Usar el estado *siguiente* (next_state) para calcular la estabilidad resultante
                stability_score = self.stability_calculator.calculate_instantaneous_stability(next_state)
                # Asegurar que está en [0, 1] y es float
                stability_score = float(np.clip(stability_score, 0.0, 1.0))
                if pd.isna(stability_score): stability_score = 0.0 # Si devuelve NaN, usar 0
            except Exception as e:
                logger.error(f"Error calculando w_stab desde {type(self.stability_calculator).__name__}: {e}", exc_info=True)
                stability_score = 0.0 # Mínima estabilidad en caso de error

        # --- [2] Calcular Recompensa (Reward) ---
        if self.use_stability_reward:
            # --- [2a] Usar Recompensa Basada en Estabilidad ---
            # Este branch solo se alcanza si self.stability_calculator existe y tiene el método
            try:
                # Usar el estado *siguiente* (next_state) para calcular la recompensa
                reward = self.stability_calculator.calculate_stability_based_reward(next_state) # type: ignore
                reward = float(reward) # Asegurar float
                if pd.isna(reward) or not np.isfinite(reward): reward = 0.0 # Usar 0 si es NaN/inf
            except Exception as e:
                logger.error(f"Error calculando recompensa desde {type(self.stability_calculator).__name__}: {e}", exc_info=True)
                reward = 0.0 # Recompensa mínima en caso de error
        else:
            # --- [2b] Usar Cálculo Gaussiano ---
            try:
                # Validar estado siguiente
                if not isinstance(next_state, (np.ndarray, list)) or len(next_state) < 4:
                     raise IndexError(f"Formato de next_state inválido o incompleto: {next_state}")

                # Calcular términos normalizados usando escalas (asegurando división > 0)
                angle_norm = next_state[self.state_indices['angle']] / self.scales['angle']
                vel_norm = next_state[self.state_indices['angular_velocity']] / self.scales['angular_velocity']
                pos_cart_norm = next_state[self.state_indices['cart_position']] / self.scales['cart_position']
                vel_cart_norm = next_state[self.state_indices['cart_velocity']] / self.scales['cart_velocity']
                # Acción (fuerza)
                force_norm = float(action) / self.scales['force']
                # Tiempo
                time_norm = t / self.scales['time']

                # Calcular exponenciales (reward = exp(-normalized_value^2))
                # Usar math.exp y limitar argumento para evitar overflow
                def safe_exp(arg):
                    try: return math.exp(-min(arg**2, 700.0))
                    except OverflowError: return 0.0

                term_angle = safe_exp(angle_norm)
                term_ang_vel = safe_exp(vel_norm)
                term_cart_pos = safe_exp(pos_cart_norm)
                term_cart_vel = safe_exp(vel_cart_norm)
                term_force = safe_exp(force_norm)
                term_time = safe_exp(time_norm)

                # Combinar términos con pesos
                reward = (self.weights['angle'] * term_angle +
                          self.weights['angular_velocity'] * term_ang_vel +
                          self.weights['cart_position'] * term_cart_pos +
                          self.weights['cart_velocity'] * term_cart_vel +
                          self.weights['force'] * term_force +
                          self.weights['time'] * term_time)

                reward = float(reward) # Asegurar float
                if pd.isna(reward) or not np.isfinite(reward): reward = 0.0 # Usar 0 si es NaN/inf

            except IndexError as e:
                logger.error(f"GaussianReward: Error de índice accediendo a next_state: {e}. Estado: {next_state}")
                reward = 0.0
            except KeyError as e:
                logger.error(f"GaussianReward: Error de clave accediendo a weights/scales: {e}. Verificar _validate_gaussian_params.")
                reward = 0.0
            except Exception as e:
                logger.error(f"Error inesperado calculando recompensa Gaussiana: {e}", exc_info=True)
                reward = 0.0

        # logger.debug(f"Calculate Result: Reward={reward:.4f}, Stability={stability_score:.4f}")
        return reward, stability_score


    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delega la actualización de estadísticas al stability_calculator si existe y es aplicable."""
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
            # logger.debug(f"Delegando actualización de stats a {type(self.stability_calculator).__name__} tras episodio {current_episode}")
            try:
                self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logger.error(f"Error llamando a update_reference_stats en stability calculator: {e}", exc_info=True)
        #else:
            # logger.debug("No hay stability calculator o no tiene método update_reference_stats.")