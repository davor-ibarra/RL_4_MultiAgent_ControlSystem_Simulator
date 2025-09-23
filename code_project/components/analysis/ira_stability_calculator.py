import numpy as np
import pandas as pd
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Importar Interfaz Base
from typing import Any, Dict, Optional
import copy # Para copiar el dict inicial de stats

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class IRAStabilityCalculator(BaseStabilityCalculator): # Implementar Interfaz BaseStabilityCalculator
    """
    Calcula la estabilidad instantánea (w_stab) y recompensa basada en estabilidad
    usando desviaciones normalizadas (z-scores) ponderadas. Puede usar estadísticas
    fijas (mu, sigma) o adaptarlas episódicamente (si está habilitado).
    """
    def __init__(self, ira_params: Dict[str, Any]):
        """
        Inicializa el calculador IRA con parámetros desde la configuración.

        Args:
            ira_params (Dict): Diccionario de la sección 'ira_params' de la config.
        """
        logger.info("Inicializando IRAStabilityCalculator...")
        try:
            # --- Parámetros y Configuración ---
            # (Lógica de extracción y validación mantenida como estaba)
            self.weights = ira_params['weights']
            self.lambda_param = ira_params.get('lambda', 1.0)
            self.z_score_epsilon = ira_params.get('z_score_epsilon', 1e-6)
            adaptive_cfg = ira_params.get('adaptive_stats', {})
            self.adaptive_enabled = adaptive_cfg.get('enabled', False)
            self.adaptive_min_episode = adaptive_cfg.get('min_episode', 0)
            self.min_sigma = adaptive_cfg.get('min_sigma', 1e-4)
            if self.min_sigma <= 0: logger.warning(f"min_sigma ({self.min_sigma}) debe ser > 0. Usando 1e-4."); self.min_sigma = 1e-4
            initial_ref_stats_raw = ira_params.get('initial_reference_stats')
            if not initial_ref_stats_raw or not isinstance(initial_ref_stats_raw, dict):
                 raise ValueError("Falta 'initial_reference_stats' o no es dict.")
            self.current_ref_stats = copy.deepcopy(initial_ref_stats_raw)

            # --- Mapeos (Mantenidos) ---
            self.var_to_metric_map = {'angle': 'pendulum_angle', 'angular_velocity': 'pendulum_velocity', 'cart_position': 'cart_position', 'cart_velocity': 'cart_velocity'}
            self.var_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
            expected_vars = set(self.var_indices.keys())

            # --- Validaciones Cruzadas (Mantenidas) ---
            if set(self.weights.keys()) != expected_vars: logger.warning(f"Discrepancia claves weights vs esperadas.")
            if set(self.current_ref_stats.keys()) != expected_vars: raise ValueError(f"Faltan entradas en initial_reference_stats. Esperadas: {expected_vars}.")
            for var, stats in self.current_ref_stats.items():
                 if not isinstance(stats, dict) or 'mu' not in stats or 'sigma' not in stats: raise ValueError(f"Entrada inválida para '{var}' en stats iniciales.")
                 if not isinstance(stats['mu'], (int, float)) or not isinstance(stats['sigma'], (int, float)): raise ValueError(f"'mu'/'sigma' para '{var}' no numéricos.")
                 if stats['sigma'] <= 0: logger.warning(f"Sigma inicial para '{var}' ({stats['sigma']}) no positivo. Usando min_sigma."); self.current_ref_stats[var]['sigma'] = self.min_sigma
                 elif stats['sigma'] < self.min_sigma: logger.warning(f"Sigma inicial para '{var}' ({stats['sigma']}) < min_sigma ({self.min_sigma}). Usando min_sigma."); self.current_ref_stats[var]['sigma'] = self.min_sigma

            logger.info(f"IRAStabilityCalculator inicializado. Adaptativo: {self.adaptive_enabled} (Ep Min={self.adaptive_min_episode})")
            # logger.debug(f"Pesos: {self.weights}, Lambda: {self.lambda_param}, Stats iniciales: {self.current_ref_stats}")

        except KeyError as e:
            logger.error(f"IRAStabilityCalculator: Falta clave requerida: {e}", exc_info=True)
            raise ValueError(f"Falta parámetro requerido para IRA: {e}") from e
        except ValueError as e:
             logger.error(f"IRAStabilityCalculator: Error de valor en config: {e}", exc_info=True)
             raise
        except Exception as e:
             logger.error(f"IRAStabilityCalculator: Error inesperado init: {e}", exc_info=True)
             raise RuntimeError("Fallo al inicializar IRAStabilityCalculator") from e

    def _normalize_state_variable(self, value: float, var_name: str) -> float:
        """Normaliza (calcula z-score). Lógica interna sin cambios."""
        # ... (código sin cambios) ...
        if var_name not in self.current_ref_stats: return 0.0
        stats = self.current_ref_stats[var_name]
        mu = stats.get('mu', 0.0); sigma_actual = stats.get('sigma', 1.0)
        sigma_effective = max(sigma_actual, self.min_sigma, self.z_score_epsilon)
        return (value - mu) / sigma_effective

    # --- Implementación de Métodos de la Interfaz BaseStabilityCalculator ---

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calcula w_stab = exp(- sum( Ws * Zs^2 ))."""
        # ... (código sin cambios funcionales) ...
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4: logger.error(f"IRA calc_stab: estado inválido {state}"); return 0.0
        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0)
                if weight > 0:
                    value = float(state[index])
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)
        except IndexError: logger.error(f"IRA calc_stab: Índice fuera de rango {var_name}. Estado: {state}"); return 0.0
        except Exception as e: logger.error(f"IRA calc_stab: Error suma ponderada: {e}", exc_info=True); return 0.0
        try:
            exponent_arg = -min(deviation_sum_sq_weighted, 700.0)
            stability_score = math.exp(exponent_arg)
        except OverflowError: logger.warning(f"IRA calc_stab: Overflow exp(-{deviation_sum_sq_weighted:.4f})."); stability_score = 0.0
        except Exception as e: logger.error(f"IRA calc_stab: Error exp(): {e}", exc_info=True); stability_score = 0.0
        return max(0.0, min(float(stability_score), 1.0))


    def calculate_stability_based_reward(self, state: Any) -> float:
        """Calcula recompensa = exp(- lambda * sum( Ws * Zs^2 ))."""
        # ... (código sin cambios funcionales) ...
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4: logger.error(f"IRA calc_reward: estado inválido {state}"); return 0.0
        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0)
                if weight > 0:
                    value = float(state[index])
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)
        except IndexError: logger.error(f"IRA calc_reward: Índice fuera de rango {var_name}. Estado: {state}"); return 0.0
        except Exception as e: logger.error(f"IRA calc_reward: Error suma ponderada: {e}", exc_info=True); return 0.0
        try:
            lambda_eff = max(abs(self.lambda_param), 1e-9)
            max_deviation_sum = 700.0 / lambda_eff
            exponent_arg = -self.lambda_param * min(deviation_sum_sq_weighted, max_deviation_sum)
            reward = math.exp(exponent_arg)
        except OverflowError: logger.warning(f"IRA calc_reward: Overflow exp(-{self.lambda_param:.2f} * {deviation_sum_sq_weighted:.4f})."); reward = 0.0
        except Exception as e: logger.error(f"IRA calc_reward: Error exp(): {e}", exc_info=True); reward = 0.0
        return max(0.0, float(reward))


    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Actualiza mu y sigma si adaptativo está habilitado."""
        # ... (código sin cambios funcionales) ...
        if not self.adaptive_enabled: return
        if current_episode < self.adaptive_min_episode: return

        logger.info(f"Intentando actualizar stats IRA tras episodio {current_episode}...")
        updated_any = False
        for var_name_cfg, metric_name in self.var_to_metric_map.items():
            if var_name_cfg in self.current_ref_stats:
                if metric_name in episode_metrics_dict:
                    values = episode_metrics_dict[metric_name]
                    try:
                         if not isinstance(values, list): logger.warning(f"Datos para '{metric_name}' no son lista."); continue
                         numeric_values = pd.to_numeric(pd.Series(values), errors='coerce')
                         valid_values = numeric_values.dropna()[np.isfinite(numeric_values.dropna())]
                    except Exception as e: logger.error(f"Error convirtiendo/filtrando datos '{metric_name}': {e}"); continue

                    if len(valid_values) > 1:
                        try:
                            new_mu = float(np.mean(valid_values)); new_sigma = float(np.std(valid_values))
                            effective_sigma = max(new_sigma, self.min_sigma)
                            # Actualizar solo si hay cambio significativo para evitar logs innecesarios
                            if abs(self.current_ref_stats[var_name_cfg]['mu'] - new_mu) > 1e-9 or \
                               abs(self.current_ref_stats[var_name_cfg]['sigma'] - effective_sigma) > 1e-9:
                                self.current_ref_stats[var_name_cfg]['mu'] = new_mu
                                self.current_ref_stats[var_name_cfg]['sigma'] = effective_sigma
                                updated_any = True
                                logger.debug(f"  Stats '{var_name_cfg}': mu={new_mu:.4f}, sigma={effective_sigma:.4f}")
                        except Exception as e: logger.error(f"Error calculando stats '{metric_name}': {e}")
                    # else: logger.debug(f"Datos insuficientes ({len(valid_values)}) para '{metric_name}' ep {current_episode}.")
                # else: logger.warning(f"Métrica '{metric_name}' no encontrada para actualizar stats '{var_name_cfg}'.")
        if updated_any: logger.info(f"Stats IRA actualizadas tras episodio {current_episode}.")


    def get_current_adaptive_stats(self) -> Dict:
         """Devuelve una copia de las estadísticas de referencia actuales."""
         # ... (código sin cambios) ...
         return copy.deepcopy(self.current_ref_stats)