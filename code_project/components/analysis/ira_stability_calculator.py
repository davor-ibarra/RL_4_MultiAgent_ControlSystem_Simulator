import numpy as np
import pandas as pd
import math
import logging
from interfaces.stability_calculator import BaseStabilityCalculator # Importar Interfaz Base
from typing import Any, Dict, Optional
import copy # Para copiar el dict inicial de stats

# Obtener logger específico para este módulo
logger = logging.getLogger(__name__)

class IRAStabilityCalculator(BaseStabilityCalculator): # Heredar de BaseStabilityCalculator
    """
    Calcula la estabilidad instantánea (w_stab) y recompensa basada en estabilidad
    usando desviaciones normalizadas (z-scores) ponderadas. Puede usar estadísticas
    fijas (mu, sigma) o adaptarlas episódicamente (si está habilitado).
    Asume que el estado es un vector/lista: [cart_pos, cart_vel, angle, angular_vel].
    """
    def __init__(self, ira_params: Dict[str, Any]):
        """
        Inicializa el calculador IRA con parámetros desde la configuración.

        Args:
            ira_params (Dict): Diccionario de la sección 'ira_params' de la config,
                               espera claves como 'weights', 'lambda', 'z_score_epsilon',
                               'adaptive_stats', 'initial_reference_stats'.
        """
        logger.info("Inicializando IRAStabilityCalculator...")
        try:
            # --- Parámetros Principales ---
            self.weights = ira_params['weights'] # Pesos Ws para cada variable
            # Lambda se usa solo si reward_method='stability_calculator'
            self.lambda_param = ira_params.get('lambda', 1.0) # Default a 1.0 si no está
            self.z_score_epsilon = ira_params.get('z_score_epsilon', 1e-6) # Para evitar división por cero en z-score

            # --- Configuración Adaptativa ---
            adaptive_cfg = ira_params.get('adaptive_stats', {})
            self.adaptive_enabled = adaptive_cfg.get('enabled', False)
            self.adaptive_min_episode = adaptive_cfg.get('min_episode', 0)
            # Mínimo sigma permitido (previene colapso a cero)
            self.min_sigma = adaptive_cfg.get('min_sigma', 1e-4) # Usar un valor razonable
            if self.min_sigma <= 0:
                 logger.warning(f"min_sigma ({self.min_sigma}) debe ser positivo. Usando 1e-4.")
                 self.min_sigma = 1e-4

            # --- Estadísticas de Referencia ---
            # Validar y copiar estadísticas iniciales
            initial_ref_stats_raw = ira_params.get('initial_reference_stats')
            if not initial_ref_stats_raw or not isinstance(initial_ref_stats_raw, dict):
                 raise ValueError("Falta 'initial_reference_stats' o no es un diccionario en ira_params.")
            # Copia profunda para no modificar el original en config
            self.current_ref_stats = copy.deepcopy(initial_ref_stats_raw)

            # Mapeo de nombres de variables en config/estado a nombres de métricas logueadas
            self.var_to_metric_map = {
                'angle': 'pendulum_angle',
                'angular_velocity': 'pendulum_velocity',
                'cart_position': 'cart_position',
                'cart_velocity': 'cart_velocity'
            }
            # Mapeo de nombres de variables a índices del vector de estado esperado
            self.var_indices = {
                'cart_position': 0,
                'cart_velocity': 1,
                'angle': 2,
                'angular_velocity': 3
            }
            expected_vars = set(self.var_indices.keys()) # Variables esperadas en estado

            # --- Validaciones Cruzadas ---
            # Validar que weights y stats iniciales contengan las claves esperadas
            if set(self.weights.keys()) != expected_vars:
                logger.warning(f"IRAStabilityCalculator: Discrepancia entre variables esperadas ({expected_vars}) "
                                f"y claves en weights ({set(self.weights.keys())}). Variables faltantes en weights tendrán peso 0.")
            if set(self.current_ref_stats.keys()) != expected_vars:
                 raise ValueError(f"Faltan entradas en 'initial_reference_stats'. Se esperan: {expected_vars}. "
                                  f"Recibidas: {set(self.current_ref_stats.keys())}")
            # Validar que cada stat inicial tenga mu y sigma
            for var, stats in self.current_ref_stats.items():
                 if not isinstance(stats, dict) or 'mu' not in stats or 'sigma' not in stats:
                      raise ValueError(f"Entrada inválida para '{var}' en initial_reference_stats. Debe ser dict con 'mu' y 'sigma'. Recibido: {stats}")
                 if not isinstance(stats['mu'], (int, float)) or not isinstance(stats['sigma'], (int, float)):
                     raise ValueError(f"Valores 'mu' o 'sigma' para '{var}' no son numéricos.")
                 if stats['sigma'] < self.min_sigma:
                     logger.warning(f"Sigma inicial para '{var}' ({stats['sigma']}) es menor que min_sigma ({self.min_sigma}). Usando min_sigma.")
                     self.current_ref_stats[var]['sigma'] = self.min_sigma # Corregir al inicio
                 elif stats['sigma'] <= 0:
                     logger.warning(f"Sigma inicial para '{var}' ({stats['sigma']}) no es positivo. Usando min_sigma.")
                     self.current_ref_stats[var]['sigma'] = self.min_sigma # Corregir al inicio


            logger.info(f"IRAStabilityCalculator inicializado. Adaptativo: {self.adaptive_enabled} "
                         f"(Ep Min={self.adaptive_min_episode}, Sigma Min={self.min_sigma})")
            logger.debug(f"Pesos (weights): {self.weights}")
            logger.debug(f"Lambda (para reward): {self.lambda_param}")
            logger.debug(f"Stats iniciales: {self.current_ref_stats}")

        except KeyError as e:
            logger.error(f"IRAStabilityCalculator: Falta clave de parámetro requerida: {e} en ira_params: {list(ira_params.keys())}")
            raise ValueError(f"Falta parámetro requerido para IRAStabilityCalculator: {e}") from e
        except ValueError as e: # Capturar errores de validación
             logger.error(f"IRAStabilityCalculator: Error de valor en configuración: {e}")
             raise
        except Exception as e:
             logger.error(f"IRAStabilityCalculator: Error inesperado durante inicialización: {e}", exc_info=True)
             raise RuntimeError("Fallo al inicializar IRAStabilityCalculator") from e

    def _normalize_state_variable(self, value: float, var_name: str) -> float:
        """Normaliza (calcula z-score) para una variable de estado usando las stats de referencia *actuales*."""
        if var_name not in self.current_ref_stats:
            logger.warning(f"Stats de referencia no encontradas para '{var_name}'. Devolviendo z-score=0.")
            return 0.0

        stats = self.current_ref_stats[var_name]
        mu = stats.get('mu', 0.0)
        # Usar max(sigma_actual, min_sigma, epsilon) para evitar división por cero o valor muy pequeño
        sigma_actual = stats.get('sigma', 1.0)
        sigma_effective = max(sigma_actual, self.min_sigma, self.z_score_epsilon)

        # Calcular z-score
        z_score = (value - mu) / sigma_effective
        return z_score

    # --- Implementación de Métodos de la Interfaz ---

    def calculate_instantaneous_stability(self, state: Any) -> float:
        """Calcula w_stab = exp(- sum( weight_s * z_score_s^2 ))."""
        # Validar entrada de estado
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.error(f"IRAStabilityCalculator: Formato de estado inválido para calcular estabilidad: {state}")
            return 0.0 # Mínima estabilidad si el estado es inválido

        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                # Usar peso 0 si no está definido para la variable
                weight = self.weights.get(var_name, 0.0)
                if weight > 0: # Solo procesar si el peso es positivo
                    value = float(state[index]) # Asegurar que es float
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)

        except IndexError:
            logger.error(f"IRAStabilityCalculator: Índice de estado fuera de rango al acceder a {var_name} (índice {index}). Estado: {state}")
            # Retornar 0 parece razonable si el estado está malformado
            return 0.0
        except Exception as e:
            logger.error(f"IRAStabilityCalculator: Error inesperado calculando suma ponderada para estabilidad: {e}", exc_info=True)
            return 0.0 # Retornar 0 en caso de error

        # Calcular exponencial y manejar overflow
        try:
            # Limitar argumento del exponente para prevenir overflow (exp(-700) es muy pequeño)
            exponent_arg = -min(deviation_sum_sq_weighted, 700.0)
            stability_score = math.exp(exponent_arg)
        except OverflowError:
            logger.warning(f"IRAStabilityCalculator: Overflow calculando exp(-{deviation_sum_sq_weighted:.4f}) para w_stab. Devolviendo 0.")
            stability_score = 0.0
        except Exception as e:
             logger.error(f"IRAStabilityCalculator: Error inesperado en cálculo de exp() para w_stab: {e}", exc_info=True)
             stability_score = 0.0


        # Asegurar que la puntuación está en [0, 1]
        return max(0.0, min(float(stability_score), 1.0))


    def calculate_stability_based_reward(self, state: Any) -> float:
        """Calcula recompensa = exp(- lambda * sum( weight_s * z_score_s^2 ))."""
        # Validar entrada de estado
        if not isinstance(state, (np.ndarray, list)) or len(state) < 4:
            logger.error(f"IRAStabilityCalculator (reward): Formato de estado inválido: {state}")
            return 0.0 # Recompensa mínima o negativa? Devolver 0.0

        deviation_sum_sq_weighted = 0.0
        try:
            for var_name, index in self.var_indices.items():
                weight = self.weights.get(var_name, 0.0)
                if weight > 0:
                    value = float(state[index])
                    z_s = self._normalize_state_variable(value, var_name)
                    deviation_sum_sq_weighted += weight * (z_s ** 2)

        except IndexError:
            logger.error(f"IRAStabilityCalculator (reward): Índice de estado fuera de rango al acceder a {var_name} (índice {index}). Estado: {state}")
            return 0.0
        except Exception as e:
            logger.error(f"IRAStabilityCalculator (reward): Error inesperado calculando suma ponderada para reward: {e}", exc_info=True)
            return 0.0

        # Calcular exponencial con lambda y manejar overflow
        try:
            # Limitar argumento basado en lambda para prevenir overflow
            # exp(-L*S) -> max S = 700 / L
            lambda_eff = max(abs(self.lambda_param), 1e-9) # Lambda efectivo > 0
            max_deviation_sum = 700.0 / lambda_eff
            exponent_arg = -self.lambda_param * min(deviation_sum_sq_weighted, max_deviation_sum)
            reward = math.exp(exponent_arg)
        except OverflowError:
            logger.warning(f"IRAStabilityCalculator (reward): Overflow calculando exp(-{self.lambda_param:.2f} * {deviation_sum_sq_weighted:.4f}). Devolviendo 0.")
            reward = 0.0
        except Exception as e:
             logger.error(f"IRAStabilityCalculator (reward): Error inesperado en cálculo de exp() para reward: {e}", exc_info=True)
             reward = 0.0

        # La recompensa basada en estabilidad debería ser no-negativa
        return max(0.0, float(reward))


    def update_reference_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Actualiza mu y sigma si adaptativo está habilitado y se alcanzó el episodio mínimo."""
        if not self.adaptive_enabled:
            # logger.debug("Actualización de stats adaptativas deshabilitada.")
            return
        if current_episode < self.adaptive_min_episode:
            # logger.debug(f"Episodio actual ({current_episode}) < min_episode ({self.adaptive_min_episode}). No se actualizan stats.")
            return

        logger.info(f"Intentando actualizar stats de referencia IRA tras episodio {current_episode}...")
        updated_any = False
        stats_before = copy.deepcopy(self.current_ref_stats) # Para comparar si hubo cambios

        for var_name_cfg, metric_name in self.var_to_metric_map.items():
            # Solo actualizar si la variable está en nuestras stats actuales
            if var_name_cfg in self.current_ref_stats:
                if metric_name in episode_metrics_dict:
                    values = episode_metrics_dict[metric_name]
                    # Convertir a numérico y filtrar NaN/inf
                    try:
                         # Asegurar que es una lista antes de pasar a Serie
                         if not isinstance(values, list):
                              logger.warning(f"Datos para métrica '{metric_name}' (var '{var_name_cfg}') no son una lista. Saltando actualización.")
                              continue

                         numeric_values = pd.to_numeric(pd.Series(values), errors='coerce')
                         valid_values = numeric_values.dropna() # Elimina NaN
                         valid_values = valid_values[np.isfinite(valid_values)] # Elimina inf
                    except Exception as e:
                         logger.error(f"Error convirtiendo/filtrando datos para métrica '{metric_name}' (var '{var_name_cfg}'): {e}")
                         continue # Saltar esta variable si hay error

                    # Necesitamos al menos 2 puntos para calcular std dev de forma fiable
                    if len(valid_values) > 1:
                        try:
                            new_mu = float(np.mean(valid_values))
                            new_sigma = float(np.std(valid_values))

                            # Aplicar sigma mínimo
                            effective_sigma = max(new_sigma, self.min_sigma)

                            # Actualizar stats internas solo si el valor cambió (o es la primera actualización)
                            if abs(self.current_ref_stats[var_name_cfg]['mu'] - new_mu) > 1e-9 or \
                               abs(self.current_ref_stats[var_name_cfg]['sigma'] - effective_sigma) > 1e-9:
                                self.current_ref_stats[var_name_cfg]['mu'] = new_mu
                                self.current_ref_stats[var_name_cfg]['sigma'] = effective_sigma
                                updated_any = True
                                logger.debug(f"  Stats actualizadas para '{var_name_cfg}': mu={new_mu:.4f}, sigma={effective_sigma:.4f}")

                        except Exception as e:
                            logger.error(f"Error calculando nuevas stats para métrica '{metric_name}' (var '{var_name_cfg}'): {e}")
                    else:
                        # Loguear si no hay suficientes datos
                        logger.debug(f"No hay suficientes datos válidos ({len(valid_values)}) para métrica '{metric_name}' en episodio {current_episode} para actualizar stats de '{var_name_cfg}'.")
                else:
                    logger.warning(f"Métrica '{metric_name}' (necesaria para stats adaptativas de '{var_name_cfg}') no encontrada en datos del episodio.")

        if updated_any:
            logger.info(f"Stats de referencia IRA actualizadas tras episodio {current_episode}.")
            # logger.debug(f"Stats ANTES: {stats_before}")
            # logger.debug(f"Stats DESPUÉS: {self.current_ref_stats}")
        # else:
            # logger.debug(f"No hubo cambios en las stats de referencia IRA tras episodio {current_episode}.")


    def get_current_adaptive_stats(self) -> Dict:
         """Devuelve una copia profunda de las estadísticas de referencia actuales."""
         # Devuelve las stats incluso si adaptativo está deshabilitado (serán las iniciales)
         return copy.deepcopy(self.current_ref_stats)