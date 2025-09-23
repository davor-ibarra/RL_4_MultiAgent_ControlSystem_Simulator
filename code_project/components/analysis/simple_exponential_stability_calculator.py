import numpy as np
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Importar Interfaz Base
from typing import Any, Dict

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class SimpleExponentialStabilityCalculator(BaseStabilityCalculator): # Implementar Interfaz BaseStabilityCalculator
    """
    Calcula una puntuación de estabilidad simple w_stab basada en la fórmula:
    w_stab = exp(- sum( lambda_s * (state_s / scale_s)**2 ))
    Este calculador NO es adaptativo.
    """
    def __init__(self, simple_exp_params: Dict[str, Any]):
        """
        Inicializa el calculador con pesos lambda y escalas desde la config.

        Args:
            simple_exp_params (Dict): Diccionario de la sección 'simple_exponential_params'.
        """
        logger.info("Inicializando SimpleExponentialStabilityCalculator...")
        try:
            # --- Parámetros y Configuración ---
            # (Lógica de extracción y validación mantenida como estaba)
            self.lambda_weights = simple_exp_params['lambda_weights']
            self.scales = simple_exp_params['scales']
            if not isinstance(self.lambda_weights, dict): raise ValueError("'lambda_weights' debe ser dict.")
            if not isinstance(self.scales, dict): raise ValueError("'scales' debe ser dict.")

            # --- Mapeos (Mantenidos) ---
            self.var_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
            expected_vars = set(self.var_indices.keys())

            # --- Validaciones Cruzadas (Mantenidas) ---
            if set(self.lambda_weights.keys()) != expected_vars: logger.warning(f"Discrepancia claves lambda_weights vs esperadas.")
            if set(self.scales.keys()) != expected_vars: logger.warning(f"Discrepancia claves scales vs esperadas.")
            for var in expected_vars:
                 # Default y validación para lambda_weights
                 lw = self.lambda_weights.get(var, 0.0)
                 if not isinstance(lw, (int, float)): logger.warning(f"Lambda para '{var}' ({lw}) no numérico. Usando 0.0."); lw = 0.0
                 self.lambda_weights[var] = lw
                 # Default y validación para scales
                 sc = self.scales.get(var, 1.0)
                 if not isinstance(sc, (int, float)) or sc <= 0: logger.warning(f"Escala para '{var}' ({sc}) no positiva. Usando 1.0."); sc = 1.0
                 self.scales[var] = sc

            logger.info("SimpleExponentialStabilityCalculator inicializado.")
            # logger.debug(f"Lambda Weights: {self.lambda_weights}, Scales: {self.scales}")

        except KeyError as e:
            logger.error(f"SimpleExponential: Falta clave requerida: {e}", exc_info=True)
            raise ValueError(f"Falta parámetro requerido para SimpleExponential: {e}") from e
        except ValueError as e:
             logger.error(f"SimpleExponential: Error de valor en config: {e}", exc_info=True)
             raise
        except Exception as e:
             logger.error(f"SimpleExponential: Error inesperado init: {e}", exc_info=True)
             raise RuntimeError("Fallo al inicializar SimpleExponentialStabilityCalculator") from e

    def _scale_state_variable(self, value: float, var_name: str) -> float:
        """Escala una variable de estado. Lógica interna sin cambios."""
        # (Asume que var_name existe y self.scales[var_name] > 0 por validación init)
        return value / self.scales[var_name]

    # --- Implementación de Métodos de la Interfaz BaseStabilityCalculator ---

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calcula w_stab = exp(- sum( lambda_s * (state_s / scale_s)^2 ))."""
        # ... (código sin cambios funcionales) ...
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4: logger.error(f"SimpleExp calc_stab: estado inválido {state}"); return 0.0
        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                lambda_w = self.lambda_weights[var_name] # Ya validado en init
                if lambda_w > 0:
                    value = float(state[index])
                    scaled_value = self._scale_state_variable(value, var_name)
                    deviation_sum_sq_weighted += lambda_w * (scaled_value ** 2)
        except IndexError: logger.error(f"SimpleExp calc_stab: Índice fuera de rango {var_name}. Estado: {state}"); return 0.0
        except Exception as e: logger.error(f"SimpleExp calc_stab: Error suma ponderada: {e}", exc_info=True); return 0.0
        try:
            exponent_arg = -min(deviation_sum_sq_weighted, 700.0)
            stability_score = math.exp(exponent_arg)
        except OverflowError: logger.warning(f"SimpleExp calc_stab: Overflow exp(-{deviation_sum_sq_weighted:.4f})."); stability_score = 0.0
        except Exception as e: logger.error(f"SimpleExp calc_stab: Error exp(): {e}", exc_info=True); stability_score = 0.0
        return max(0.0, min(float(stability_score), 1.0))

    def calculate_stability_based_reward(self, state: Any) -> float:
        """Devuelve w_stab ya que no tiene un parámetro lambda separado para recompensa."""
        # ... (código sin cambios) ...
        return self.calculate_instantaneous_stability(state)

    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """No adaptativo, no hace nada."""
        # ... (código sin cambios) ...
        pass

    def get_current_adaptive_stats(self) -> Dict:
         """No adaptativo, devuelve diccionario vacío."""
         # ... (código sin cambios) ...
         return {}