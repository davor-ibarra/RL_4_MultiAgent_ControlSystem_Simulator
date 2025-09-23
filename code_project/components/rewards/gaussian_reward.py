import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction # Importar Interfaz
from interfaces.stability_calculator import BaseStabilityCalculator # Importar interfaz base

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class GaussianReward(RewardFunction): # Implementar Interfaz RewardFunction
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
                 init_config: Dict[str, Any], # Config específica para esta clase (de RewardFactory)
                 stability_calculator: Optional[BaseStabilityCalculator] = None # Dependencia Inyectada
                 ):
        """
        Inicializa la función de recompensa Gaussiana/Basada en Estabilidad.

        Args:
            init_config (Dict[str, Any]): Diccionario de configuración específico,
                                          espera claves como 'params' (con 'weights', 'scales'),
                                          'use_stability_based_reward'.
            stability_calculator (Optional[BaseStabilityCalculator]): Instancia inyectada del
                                       calculador de estabilidad (puede ser None).
        """
        logger.info("Inicializando GaussianReward...")
        try:
            # --- Almacenar Dependencias y Procesar Configuración ---
            self.stability_calculator = stability_calculator # Almacenar instancia inyectada
            self.params = init_config.get('params', {})
            self.weights = self.params.get('weights', {}) # Pesos para modo Gaussiano
            self.scales = self.params.get('scales', {})   # Escalas para modo Gaussiano

            # Determinar modo de cálculo de recompensa desde init_config
            self.use_stability_reward = init_config.get('use_stability_based_reward', False)

            # --- Mapeo de Estado (para modo Gaussiano) ---
            self.state_indices = {
                'cart_position': 0, 'cart_velocity': 1,
                'angle': 2, 'angular_velocity': 3
            }
            self.required_gaussian_keys = ['angle', 'angular_velocity', 'cart_position', 'cart_velocity', 'force', 'time']

            # --- Validaciones (ya realizadas en RewardFactory, pero doble check es opcional) ---
            if self.use_stability_reward:
                if self.stability_calculator is None:
                    # Este error debería ser prevenido por RewardFactory
                    msg = "CRITICAL: GaussianReward configurado para usar recompensa de estabilidad, ¡pero no se proporcionó StabilityCalculator!"
                    logger.error(msg)
                    raise ValueError(msg)
                if not hasattr(self.stability_calculator, 'calculate_stability_based_reward'):
                    # Este error también debería ser prevenido por RewardFactory
                    msg = f"CRITICAL: StabilityCalculator ({type(self.stability_calculator).__name__}) no tiene 'calculate_stability_based_reward'."
                    logger.error(msg)
                    raise AttributeError(msg)
                logger.info(f"GaussianReward configurado para usar recompensa desde: {type(self.stability_calculator).__name__}")
            else:
                logger.info("GaussianReward configurado para usar cálculo Gaussiano.")
                # Validar params gaussianos internamente (opcional si la factory ya valida)
                self._validate_gaussian_params()

            # Validar que el stability_calculator (si existe) tiene el método para w_stab
            if self.stability_calculator and not hasattr(self.stability_calculator, 'calculate_instantaneous_stability'):
                 msg = f"CRITICAL: StabilityCalculator ({type(self.stability_calculator).__name__}) no tiene 'calculate_instantaneous_stability'."
                 logger.error(msg)
                 raise AttributeError(msg)
            elif self.stability_calculator is None:
                 logger.warning("GaussianReward: No se proporcionó StabilityCalculator. w_stab será siempre 1.0.")


        except Exception as e:
            logger.error(f"GaussianReward: Error durante inicialización: {e}", exc_info=True)
            # Relanzar para que falle la creación del contenedor si es crítico
            raise RuntimeError("Fallo al inicializar GaussianReward") from e

    def _validate_gaussian_params(self):
        """Valida la presencia y tipo de claves/valores para modo Gaussiano."""
        # ... (lógica mantenida como estaba) ...
        logger.debug("Validando parámetros para modo Gaussiano...")
        for key in self.required_gaussian_keys:
            if key not in self.weights:
                logger.warning(f"Modo Gaussiano: Falta peso '{key}'. Usando 0.0.")
                self.weights[key] = 0.0
            elif not isinstance(self.weights[key], (int, float)):
                logger.warning(f"Modo Gaussiano: Peso '{key}' ({self.weights[key]}) no numérico. Usando 0.0.")
                self.weights[key] = 0.0
            if key not in self.scales:
                logger.warning(f"Modo Gaussiano: Falta escala '{key}'. Usando 1.0.")
                self.scales[key] = 1.0
            elif not isinstance(self.scales[key], (int, float)) or self.scales[key] <= 0:
                logger.warning(f"Modo Gaussiano: Escala '{key}' ({self.scales[key]}) no positiva. Usando 1.0.")
                self.scales[key] = 1.0

    # --- Implementación de Métodos de la Interfaz RewardFunction ---

    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """Calcula recompensa y w_stab según modo y stability_calculator inyectado."""
        # ... (lógica mantenida como estaba, usa self.stability_calculator inyectado) ...
        stability_score = 1.0; reward = 0.0

        # 1. Calcular w_stab (si hay calculador)
        if self.stability_calculator:
            try:
                stability_score = self.stability_calculator.calculate_instantaneous_stability(next_state)
                stability_score = float(np.clip(stability_score, 0.0, 1.0))
                if pd.isna(stability_score): stability_score = 0.0
            except Exception as e:
                logger.error(f"Error calculando w_stab desde {type(self.stability_calculator).__name__}: {e}", exc_info=True)
                stability_score = 0.0

        # 2. Calcular Recompensa
        if self.use_stability_reward:
            # Se asume que self.stability_calculator no es None aquí debido a la validación en __init__
            try:
                reward = self.stability_calculator.calculate_stability_based_reward(next_state) # type: ignore
                reward = float(reward)
                if pd.isna(reward) or not np.isfinite(reward): reward = 0.0
            except Exception as e:
                logger.error(f"Error calculando reward desde {type(self.stability_calculator).__name__}: {e}", exc_info=True)
                reward = 0.0
        else: # Usar cálculo Gaussiano
            try:
                if not isinstance(next_state, (np.ndarray, list)) or len(next_state) < 4:
                     raise IndexError(f"Formato next_state inválido: {next_state}")
                # Función auxiliar para exp seguro
                def safe_exp(arg):
                     try: return math.exp(-min(arg**2, 700.0))
                     except OverflowError: return 0.0
                # Calcular términos normalizados y exponenciales
                terms = {key: safe_exp(next_state[self.state_indices[key]] / self.scales[key]) for key in self.state_indices}
                terms['force'] = safe_exp(float(action) / self.scales['force'])
                terms['time'] = safe_exp(t / self.scales['time'])
                # Calcular recompensa ponderada
                reward = sum(self.weights[key] * terms[key] for key in self.required_gaussian_keys)
                reward = float(reward)
                if pd.isna(reward) or not np.isfinite(reward): reward = 0.0
            except IndexError as e: logger.error(f"GaussReward: Índice error: {e}"); reward = 0.0
            except KeyError as e: logger.error(f"GaussReward: Clave error: {e}"); reward = 0.0
            except Exception as e: logger.error(f"Error cálculo Gaussiano: {e}", exc_info=True); reward = 0.0

        # logger.debug(f"Calculate Result: Reward={reward:.4f}, Stability={stability_score:.4f}")
        return reward, stability_score

    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delega la actualización al stability_calculator inyectado."""
        # ... (lógica mantenida como estaba) ...
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
            try:
                # logger.debug(f"Delegando update_stats a {type(self.stability_calculator).__name__}")
                self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logger.error(f"Error llamando update_reference_stats en calculator: {e}", exc_info=True)
        # else: logger.debug("No hay stability calculator o no tiene método update_reference_stats.")