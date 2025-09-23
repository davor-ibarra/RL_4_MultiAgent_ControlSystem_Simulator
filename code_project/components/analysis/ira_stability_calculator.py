import numpy as np
import pandas as pd
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Importar Interfaz Base
from typing import Any, Dict, Optional
import copy # Para copiar el dict inicial de stats

# 3.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class IRAStabilityCalculator(BaseStabilityCalculator): # Implementar Interfaz BaseStabilityCalculator
    """
    Calcula estabilidad IRA y recompensa. Implementa BaseStabilityCalculator.
    Puede adaptar estadísticas mu/sigma si está configurado.
    """
    def __init__(self, ira_params: Dict[str, Any]):
        """
        Inicializa el calculador IRA con parámetros validados.

        Args:
            ira_params (Dict): Diccionario de la sección 'ira_params' de la config.

        Raises:
            ValueError: Si faltan parámetros requeridos o son inválidos.
            TypeError: Si los tipos de parámetros son incorrectos.
        """
        logger.info("Inicializando IRAStabilityCalculator...")
        try:
            # --- Parámetros y Configuración ---
            # 3.2: Extraer y validar parámetros esenciales (Fail-Fast)
            if 'weights' not in ira_params or not isinstance(ira_params['weights'], dict):
                 raise ValueError("Falta 'weights' o no es dict en ira_params.")
            self.weights = ira_params['weights']

            if 'initial_reference_stats' not in ira_params or not isinstance(ira_params['initial_reference_stats'], dict):
                 raise ValueError("Falta 'initial_reference_stats' o no es dict en ira_params.")
            initial_ref_stats_raw = ira_params['initial_reference_stats']

            self.lambda_param = float(ira_params.get('lambda', 1.0)) # Default lambda a 1.0
            self.z_score_epsilon = float(ira_params.get('z_score_epsilon', 1e-6))
            if self.z_score_epsilon <= 0: logger.warning(f"z_score_epsilon ({self.z_score_epsilon}) debe ser > 0. Usando 1e-6."); self.z_score_epsilon = 1e-6

            # --- Configuración Adaptativa ---
            adaptive_cfg = ira_params.get('adaptive_stats', {})
            if not isinstance(adaptive_cfg, dict): raise TypeError("'adaptive_stats' debe ser un diccionario.")
            self.adaptive_enabled = bool(adaptive_cfg.get('enabled', False))
            self.adaptive_min_episode = int(adaptive_cfg.get('min_episode', 0))
            self.min_sigma = float(adaptive_cfg.get('min_sigma', 1e-4))
            if self.min_sigma <= 0: logger.warning(f"min_sigma ({self.min_sigma}) debe ser > 0. Usando 1e-4."); self.min_sigma = 1e-4

            # --- Validar y almacenar stats iniciales ---
            self.current_ref_stats = self._validate_and_copy_stats(initial_ref_stats_raw)

            # --- Mapeos ---
            self.var_to_metric_map = {'angle': 'pendulum_angle', 'angular_velocity': 'pendulum_velocity', 'cart_position': 'cart_position', 'cart_velocity': 'cart_velocity'}
            self.var_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
            expected_vars = set(self.var_indices.keys())

            # --- Validaciones Cruzadas Finales ---
            if set(self.weights.keys()) != expected_vars:
                 logger.warning(f"Claves en 'weights' no coinciden con variables esperadas {expected_vars}. Weights: {self.weights.keys()}")
            if set(self.current_ref_stats.keys()) != expected_vars:
                 # Esto no debería ocurrir si _validate_and_copy_stats funciona
                 raise ValueError(f"Error interno: Stats validadas no coinciden con variables esperadas.")

            logger.info(f"IRAStabilityCalculator inicializado. Adaptativo: {self.adaptive_enabled} (Ep Min={self.adaptive_min_episode})")
            logger.debug(f"Pesos: {self.weights}, Lambda: {self.lambda_param}, Stats iniciales: {self.current_ref_stats}")

        except (KeyError, ValueError, TypeError) as e:
            logger.critical(f"Error crítico inicializando IRAStabilityCalculator: {e}", exc_info=True)
            raise # Relanzar para detener (Fail-Fast)
        except Exception as e:
             logger.critical(f"Error inesperado inicializando IRAStabilityCalculator: {e}", exc_info=True)
             raise RuntimeError("Fallo inesperado al inicializar IRAStabilityCalculator") from e

    def _validate_and_copy_stats(self, stats_dict: Dict) -> Dict:
        """Valida la estructura y copia las estadísticas iniciales."""
        validated_stats = {}
        expected_vars = {'cart_position', 'cart_velocity', 'angle', 'angular_velocity'}
        if set(stats_dict.keys()) != expected_vars:
             raise ValueError(f"Claves en initial_reference_stats no coinciden con esperadas {expected_vars}. Stats: {stats_dict.keys()}")

        for var, stats in stats_dict.items():
            if not isinstance(stats, dict) or 'mu' not in stats or 'sigma' not in stats:
                 raise ValueError(f"Entrada inválida para '{var}' en stats iniciales: {stats}. Faltan 'mu' o 'sigma'.")
            mu = stats['mu']; sigma = stats['sigma']
            if not isinstance(mu, (int, float)): raise TypeError(f"'mu' para '{var}' no es numérico: {mu}")
            if not isinstance(sigma, (int, float)): raise TypeError(f"'sigma' para '{var}' no es numérico: {sigma}")
            if sigma <= 0:
                 logger.warning(f"Sigma inicial para '{var}' ({sigma}) no es positivo. Usando min_sigma ({self.min_sigma}) en su lugar.")
                 effective_sigma = self.min_sigma
            else:
                 effective_sigma = max(sigma, self.min_sigma) # Aplicar min_sigma también al inicio
            validated_stats[var] = {'mu': float(mu), 'sigma': float(effective_sigma)}
        return copy.deepcopy(validated_stats) # Devolver copia profunda

    def _normalize_state_variable(self, value: float, var_name: str) -> float:
        """Normaliza (calcula z-score) usando las stats actuales."""
        # Asume que var_name existe en current_ref_stats (validado en init)
        stats = self.current_ref_stats[var_name]
        mu = stats.get('mu', 0.0)
        sigma_actual = stats.get('sigma', 1.0) # Debería existir siempre
        # Usar el mayor entre sigma actual, min_sigma y z_score_epsilon para evitar división por cero/muy pequeña
        sigma_effective = max(sigma_actual, self.min_sigma, self.z_score_epsilon)
        # Devolver 0.0 si value es NaN o inf
        if pd.isna(value) or not np.isfinite(value):
            logger.warning(f"Valor inválido (NaN/inf) para normalizar '{var_name}': {value}.")
            return 0.0
        return (value - mu) / sigma_effective


    # --- Implementación de Métodos de la Interfaz BaseStabilityCalculator ---

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calcula w_stab = exp(- sum( Ws * Zs^2 ))."""
        # 3.3: Validar entrada 'state'
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.warning(f"IRA calc_stab: estado inválido o corto: {state}")
            return 0.0 # Devolver valor neutro/malo

        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0) # Obtener peso (default 0)
                if weight > 0:
                    value = float(state[index]) # Convertir a float
                    z_s = self._normalize_state_variable(value, var_name) # Maneja NaN/inf
                    deviation_sum_sq_weighted += weight * (z_s ** 2)

            # Limitar argumento de exp para evitar OverflowError
            max_exponent_arg = 700.0 # math.exp soporta hasta ~709
            exponent_arg = -min(deviation_sum_sq_weighted, max_exponent_arg)
            stability_score = math.exp(exponent_arg)

        except IndexError:
            logger.error(f"IRA calc_stab: Índice fuera de rango. Estado: {state}")
            return 0.0 # Error -> mal valor
        except Exception as e: # Capturar otros errores (e.g., en _normalize)
            logger.error(f"IRA calc_stab: Error inesperado cálculo estabilidad: {e}", exc_info=True)
            return 0.0 # Error -> mal valor

        # Asegurar que el resultado esté en [0, 1] y sea float
        return float(np.clip(stability_score, 0.0, 1.0))


    def calculate_stability_based_reward(self, state: Any) -> float:
        """Calcula recompensa = exp(- lambda * sum( Ws * Zs^2 ))."""
        # 3.4: Reutilizar lógica de cálculo de suma ponderada
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.warning(f"IRA calc_reward: estado inválido o corto: {state}")
            return 0.0 # Recompensa 0 si estado inválido

        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0)
                if weight > 0:
                    value = float(state[index])
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)

            # Aplicar lambda y limitar exponente
            lambda_eff = max(abs(self.lambda_param), 1e-9) # Evitar lambda cero
            max_deviation_sum_scaled = 700.0 / lambda_eff
            exponent_arg = -self.lambda_param * min(deviation_sum_sq_weighted, max_deviation_sum_scaled)
            reward = math.exp(exponent_arg)

        except IndexError:
            logger.error(f"IRA calc_reward: Índice fuera de rango. Estado: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"IRA calc_reward: Error inesperado cálculo recompensa: {e}", exc_info=True)
            return 0.0

        # Devolver recompensa (debe ser >= 0)
        return float(max(0.0, reward))


    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Actualiza mu y sigma si adaptativo está habilitado y se supera min_episode."""
        if not self.adaptive_enabled: return
        if current_episode < self.adaptive_min_episode: return

        logger.info(f"Actualizando stats adaptativas IRA tras episodio {current_episode}...")
        updated_any = False
        for var_name_cfg, metric_name in self.var_to_metric_map.items():
            if var_name_cfg not in self.current_ref_stats: continue # Saltar si no tenemos stats para esta var

            values = episode_metrics_dict.get(metric_name)
            if not isinstance(values, list) or len(values) < 2: # Necesitar al menos 2 puntos para std dev
                #logger.debug(f"Datos insuficientes o inválidos para '{metric_name}' (var: {var_name_cfg}) en ep {current_episode}.")
                continue

            try:
                # Convertir a numérico, eliminar NaN/inf
                numeric_values = pd.to_numeric(pd.Series(values), errors='coerce')
                valid_values = numeric_values.dropna()[np.isfinite(numeric_values.dropna())]

                if len(valid_values) < 2: # Chequear de nuevo después de limpiar
                    #logger.debug(f"Datos válidos insuficientes (<2) para '{metric_name}' (var: {var_name_cfg}) en ep {current_episode}.")
                    continue

                # Calcular nuevos mu y sigma
                new_mu = float(np.mean(valid_values))
                new_sigma = float(np.std(valid_values)) # Usar desviación estándar poblacional (o muestral ddof=1?)
                # Aplicar sigma mínimo
                effective_sigma = max(new_sigma, self.min_sigma)

                # Actualizar si hay cambio significativo
                current_mu = self.current_ref_stats[var_name_cfg]['mu']
                current_sigma = self.current_ref_stats[var_name_cfg]['sigma']
                if not np.isclose(current_mu, new_mu) or not np.isclose(current_sigma, effective_sigma):
                    self.current_ref_stats[var_name_cfg]['mu'] = new_mu
                    self.current_ref_stats[var_name_cfg]['sigma'] = effective_sigma
                    updated_any = True
                    logger.debug(f"  Stats '{var_name_cfg}' actualizadas: mu={new_mu:.4f}, sigma={effective_sigma:.4f}")

            except Exception as e:
                logger.error(f"Error actualizando stats para '{metric_name}' (var: {var_name_cfg}): {e}", exc_info=True)
                # No relanzar, intentar actualizar otras variables

        if updated_any: logger.info(f"Stats adaptativas IRA actualizadas tras episodio {current_episode}.")
        # else: logger.debug(f"No hubo cambios significativos en stats adaptativas IRA tras ep {current_episode}.")


    def get_current_adaptive_stats(self) -> Dict:
         """Devuelve una copia de las estadísticas de referencia actuales."""
         # Devolver copia profunda para evitar modificaciones externas
         return copy.deepcopy(self.current_ref_stats)