import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Importar Interfaz Base
from typing import Any, Dict

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator): # Heredar de BaseStabilityCalculator
    """
    Calcula una puntuación de estabilidad simple w_stab basada en la fórmula:
    w_stab = exp(- sum( lambda_s * (state_s / scale_s)**2 ))
    Este calculador NO es adaptativo.
    Asume que el estado es un vector/lista: [cart_pos, cart_vel, angle, angular_vel].
    """
    def __init__(self, simple_exp_params: Dict[str, Any]):
        """
        Inicializa el calculador con pesos lambda y escalas desde la config.

        Args:
            simple_exp_params (Dict): Diccionario de la sección 'simple_exponential_params',
                                      espera claves 'lambda_weights' y 'scales'.
        """
        logger.info("Inicializando SimpleExponentialStabilityCalculator...")
        try:
            self.lambda_weights = simple_exp_params['lambda_weights']
            self.scales = simple_exp_params['scales']

            # Mapeo de nombres de variables a índices del vector de estado esperado
            self.var_indices = {
                'cart_position': 0,
                'cart_velocity': 1,
                'angle': 2,
                'angular_velocity': 3
            }
            expected_vars = set(self.var_indices.keys())

            # Validar presencia y tipo de parámetros principales
            if not isinstance(self.lambda_weights, dict):
                 raise ValueError("'lambda_weights' debe ser un diccionario.")
            if not isinstance(self.scales, dict):
                 raise ValueError("'scales' debe ser un diccionario.")

            # Validar claves y valores dentro de los diccionarios
            if set(self.lambda_weights.keys()) != expected_vars:
                logger.warning(f"SimpleExponentialStabilityCalculator: Discrepancia entre variables esperadas ({expected_vars}) "
                                f"y claves en lambda_weights ({set(self.lambda_weights.keys())}). Variables faltantes tendrán lambda=0.")
            if set(self.scales.keys()) != expected_vars:
                 logger.warning(f"SimpleExponentialStabilityCalculator: Discrepancia entre variables esperadas ({expected_vars}) "
                              f"y claves en scales ({set(self.scales.keys())}). Variables faltantes usarán escala=1.0.")

            # Asegurar que los valores sean numéricos y las escalas sean positivas
            for var in expected_vars:
                # Asegurar que lambda_weights tenga la clave, default a 0.0
                if var not in self.lambda_weights:
                    self.lambda_weights[var] = 0.0
                elif not isinstance(self.lambda_weights[var], (int, float)):
                     logger.warning(f"Valor lambda para '{var}' ({self.lambda_weights[var]}) no es numérico. Usando 0.0.")
                     self.lambda_weights[var] = 0.0

                # Asegurar que scales tenga la clave, default a 1.0
                if var not in self.scales:
                    self.scales[var] = 1.0
                elif not isinstance(self.scales[var], (int, float)) or self.scales[var] <= 0:
                     logger.warning(f"Valor de escala para '{var}' ({self.scales[var]}) no es numérico positivo. Usando 1.0.")
                     self.scales[var] = 1.0

            logger.info("SimpleExponentialStabilityCalculator inicializado.")
            logger.debug(f"Lambda Weights: {self.lambda_weights}")
            logger.debug(f"Scales: {self.scales}")

        except KeyError as e:
            logger.error(f"SimpleExponentialStabilityCalculator: Falta clave de parámetro requerida: {e} en simple_exp_params: {list(simple_exp_params.keys())}")
            raise ValueError(f"Falta parámetro requerido para SimpleExponentialStabilityCalculator: {e}") from e
        except ValueError as e: # Capturar errores de validación
             logger.error(f"SimpleExponentialStabilityCalculator: Error de valor en configuración: {e}")
             raise
        except Exception as e:
             logger.error(f"SimpleExponentialStabilityCalculator: Error inesperado durante inicialización: {e}", exc_info=True)
             raise RuntimeError("Fallo al inicializar SimpleExponentialStabilityCalculator") from e

    def _scale_state_variable(self, value: float, var_name: str) -> float:
        """Escala una variable de estado usando la escala configurada."""
        # Ya validamos que self.scales[var_name] existe y es > 0 en __init__
        scale = self.scales[var_name]
        # No necesitamos check de división por cero aquí debido a la validación
        return value / scale

    # --- Implementación de Métodos de la Interfaz ---

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calcula w_stab = exp(- sum( lambda_s * (state_s / scale_s)^2 ))."""
        # Validar entrada de estado
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.error(f"SimpleExponentialStabilityCalculator: Formato de estado inválido: {state}")
            return 0.0 # Mínima estabilidad

        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                # Usar lambda weight configurado (default a 0 si faltaba)
                lambda_w = self.lambda_weights[var_name]
                if lambda_w > 0: # Solo procesar si lambda es positivo
                    value = float(state[index]) # Asegurar float
                    scaled_value = self._scale_state_variable(value, var_name)
                    deviation_sum_sq_weighted += lambda_w * (scaled_value ** 2)

        except IndexError:
            logger.error(f"SimpleExponentialStabilityCalculator: Índice de estado fuera de rango al acceder a {var_name} (índice {index}). Estado: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"SimpleExponentialStabilityCalculator: Error inesperado calculando suma ponderada para w_stab: {e}", exc_info=True)
            return 0.0

        # Calcular exponencial y manejar overflow
        try:
            # Limitar argumento del exponente
            exponent_arg = -min(deviation_sum_sq_weighted, 700.0)
            stability_score = math.exp(exponent_arg)
        except OverflowError:
            logger.warning(f"SimpleExponentialStabilityCalculator: Overflow calculando exp(-{deviation_sum_sq_weighted:.4f}) para w_stab. Devolviendo 0.")
            stability_score = 0.0
        except Exception as e:
             logger.error(f"SimpleExponentialStabilityCalculator: Error inesperado en cálculo de exp() para w_stab: {e}", exc_info=True)
             stability_score = 0.0

        # Asegurar que la puntuación está en [0, 1]
        return max(0.0, min(float(stability_score), 1.0))

    def calculate_stability_based_reward(self, state: Any) -> float:
        """
        Este calculador simple no tiene un concepto separado de recompensa basada
        en estabilidad (no tiene un parámetro 'lambda' como IRA).
        Podríamos devolver 0.0 o la propia puntuación de estabilidad w_stab.
        Devolver w_stab parece razonable como una recompensa intrínseca de estabilidad.
        """
        # Devolver la misma puntuación calculada por calculate_instantaneous_stability
        return self.calculate_instantaneous_stability(state)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Este calculador no es adaptativo, así que este método no hace nada."""
        pass # Implementación vacía requerida por la interfaz

    def get_current_adaptive_stats(self) -> Dict:
         """Este calculador no es adaptativo, devuelve un diccionario vacío."""
         return {} # Implementación requerida por la interfaz