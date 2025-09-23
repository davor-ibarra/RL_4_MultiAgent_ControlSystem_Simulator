import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Importar Interfaz Base
from typing import Any, Dict
import pandas as pd # Para isnan

# 4.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator): # Implementar Interfaz BaseStabilityCalculator
    """
    Calcula estabilidad simple w_stab = exp(- sum( lambda_s * (state_s / scale_s)**2 )).
    No es adaptativo. Implementa BaseStabilityCalculator.
    """
    def __init__(self, simple_exp_params: Dict[str, Any]):
        """
        Inicializa el calculador con parámetros validados.

        Args:
            simple_exp_params (Dict): Diccionario de la sección 'simple_exponential_params'.

        Raises:
            ValueError: Si faltan parámetros requeridos o son inválidos.
            TypeError: Si los tipos de parámetros son incorrectos.
        """
        logger.info("Inicializando SimpleExponentialStabilityCalculator...")
        try:
            # 4.2: Extraer y validar parámetros (Fail-Fast)
            if 'lambda_weights' not in simple_exp_params or not isinstance(simple_exp_params['lambda_weights'], dict):
                raise ValueError("Falta 'lambda_weights' o no es dict en simple_exp_params.")
            self.lambda_weights = simple_exp_params['lambda_weights']

            if 'scales' not in simple_exp_params or not isinstance(simple_exp_params['scales'], dict):
                raise ValueError("Falta 'scales' o no es dict en simple_exp_params.")
            self.scales = simple_exp_params['scales']

            # --- Mapeos ---
            self.var_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
            expected_vars = set(self.var_indices.keys())

            # --- Validaciones y Defaults ---
            # Validar y establecer defaults para lambda_weights
            valid_weights = {}
            for var in expected_vars:
                lw = self.lambda_weights.get(var, 0.0) # Default a 0.0 si falta
                if not isinstance(lw, (int, float)):
                     logger.warning(f"Lambda weight para '{var}' ({lw}) no numérico. Usando 0.0.")
                     lw = 0.0
                valid_weights[var] = float(lw)
            self.lambda_weights = valid_weights

            # Validar y establecer defaults para scales
            valid_scales = {}
            for var in expected_vars:
                sc = self.scales.get(var, 1.0) # Default a 1.0 si falta
                if not isinstance(sc, (int, float)) or sc <= 0:
                     logger.warning(f"Escala para '{var}' ({sc}) no positiva. Usando 1.0.")
                     sc = 1.0
                valid_scales[var] = float(sc)
            self.scales = valid_scales

            logger.info("SimpleExponentialStabilityCalculator inicializado.")
            logger.debug(f"Lambda Weights: {self.lambda_weights}, Scales: {self.scales}")

        except (KeyError, ValueError, TypeError) as e:
            logger.critical(f"Error crítico inicializando SimpleExponential: {e}", exc_info=True)
            raise # Relanzar para detener (Fail-Fast)
        except Exception as e:
             logger.critical(f"Error inesperado inicializando SimpleExponential: {e}", exc_info=True)
             raise RuntimeError("Fallo inesperado al inicializar SimpleExponential") from e

    def _scale_state_variable(self, value: float, var_name: str) -> float:
        """Escala una variable de estado usando la escala configurada."""
        # Asume que var_name existe y self.scales[var_name] > 0 (validado en init)
        if pd.isna(value) or not np.isfinite(value):
             logger.warning(f"Valor inválido (NaN/inf) para escalar '{var_name}': {value}.")
             return 0.0 # Devolver 0 si el valor es inválido
        return value / self.scales[var_name]


    # --- Implementación de Métodos de la Interfaz BaseStabilityCalculator ---

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calcula w_stab = exp(- sum( lambda_s * (state_s / scale_s)^2 ))."""
        # 4.3: Validar entrada 'state'
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.warning(f"SimpleExp calc_stab: estado inválido o corto: {state}")
            return 0.0 # Devolver valor malo

        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                lambda_w = self.lambda_weights.get(var_name, 0.0) # Obtener peso (ya validado)
                if lambda_w > 0:
                    value = float(state[index]) # Convertir a float
                    scaled_value = self._scale_state_variable(value, var_name) # Maneja NaN/inf
                    deviation_sum_sq_weighted += lambda_w * (scaled_value ** 2)

            # Limitar argumento de exp
            max_exponent_arg = 700.0
            exponent_arg = -min(deviation_sum_sq_weighted, max_exponent_arg)
            stability_score = math.exp(exponent_arg)

        except IndexError:
            logger.error(f"SimpleExp calc_stab: Índice fuera de rango. Estado: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"SimpleExp calc_stab: Error inesperado cálculo estabilidad: {e}", exc_info=True)
            return 0.0

        # Asegurar resultado en [0, 1] y tipo float
        return float(np.clip(stability_score, 0.0, 1.0))

    def calculate_stability_based_reward(self, state: Any) -> float:
        """Devuelve w_stab ya que no tiene un parámetro lambda separado para recompensa."""
        # Reutiliza el cálculo de w_stab como recompensa
        return self.calculate_instantaneous_stability(state)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """No adaptativo, no hace nada."""
        pass # Implementación vacía requerida por la interfaz

    def get_current_adaptive_stats(self) -> Dict:
         """No adaptativo, devuelve diccionario vacío."""
         return {} # Implementación requerida por la interfaz