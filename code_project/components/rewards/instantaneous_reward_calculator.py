import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Any, Optional, Dict
from interfaces.reward_function import RewardFunction # Importar Interfaz
from interfaces.stability_calculator import BaseStabilityCalculator # Importar interfaz base

# 7.1: Usar logger específico del módulo
logger = logging.getLogger(__name__)

class InstantaneousRewardCalculator(RewardFunction): # Implementar Interfaz RewardFunction
    """
    Calcula recompensa instantánea R y estabilidad w_stab. Implementa RewardFunction.
    Usa un método de cálculo ('gaussian' o 'stability_calculator') y opcionalmente
    un StabilityCalculator inyectado.
    """
    def __init__(self,
                 calculation_config: Dict[str, Any], # Config de reward_setup.calculation
                 stability_calculator: Optional[BaseStabilityCalculator]
                 ):
        """
        Inicializa el calculador de recompensa instantánea.

        Args:
            calculation_config: Sección 'calculation' de la config.
            stability_calculator: Instancia opcional de StabilityCalculator inyectada.

        Raises:
            ValueError: Si la configuración es inválida o falta 'method'.
            TypeError: Si los tipos en config son incorrectos.
            AttributeError: Si stability_calculator es requerido pero inválido.
            RuntimeError: Por errores inesperados.
        """
        logger.info("Inicializando InstantaneousRewardCalculator...")
        try:
            # --- Validar y Almacenar Dependencias/Config ---
            # 7.2: Validar stability_calculator si se proporciona (tipo ya validado por DI)
            if stability_calculator is not None and not isinstance(stability_calculator, BaseStabilityCalculator):
                 # Este check es redundante si DI funciona, pero por seguridad
                 raise TypeError("stability_calculator debe implementar BaseStabilityCalculator")
            self.stability_calculator = stability_calculator

            if not isinstance(calculation_config, dict):
                 raise ValueError("calculation_config debe ser un diccionario.")
            self.calculation_config = calculation_config
            self.method = calculation_config.get('method')
            if not self.method:
                raise ValueError("Falta 'method' en calculation_config.") # Fail-Fast
            logger.info(f"InstantaneousRewardCalculator modo: {self.method}")

            # --- Configuración Específica por Método ---
            # 7.3: Limpiar atributos no usados por el método seleccionado
            self.gaussian_params = {}
            self.weights = {}
            self.scales = {}
            self.state_indices = {}

            if self.method == 'gaussian':
                self.gaussian_params = calculation_config.get('gaussian_params', {})
                if not isinstance(self.gaussian_params, dict):
                    raise TypeError("gaussian_params debe ser dict para method 'gaussian'.") # Fail-Fast
                # Extraer y validar params gaussianos
                self._load_and_validate_gaussian_params()

            elif self.method == 'stability_calculator':
                if self.stability_calculator is None:
                    # Fail-Fast si se requiere pero no se inyectó
                    msg = "CRITICAL: method='stability_calculator' seleccionado, pero no se inyectó StabilityCalculator."
                    logger.critical(msg); raise ValueError(msg)
                # Validar que tenga el método necesario (ya hecho en RewardFactory, pero doble check)
                if not hasattr(self.stability_calculator, 'calculate_stability_based_reward'):
                    msg = f"CRITICAL: StabilityCalculator ({type(self.stability_calculator).__name__}) sin método 'calculate_stability_based_reward'."
                    logger.critical(msg); raise AttributeError(msg)
                logger.info(f"Recompensa será calculada por: {type(self.stability_calculator).__name__}")

            else:
                # Fail-Fast si el método es desconocido (ya validado en config_loader/factory)
                raise ValueError(f"Método de cálculo desconocido: {self.method}")

            # Validar capacidad de cálculo de w_stab (si calculator existe)
            if self.stability_calculator and not hasattr(self.stability_calculator, 'calculate_instantaneous_stability'):
                 msg = f"CRITICAL: StabilityCalculator inyectado ({type(self.stability_calculator).__name__}) sin método 'calculate_instantaneous_stability'."
                 logger.critical(msg); raise AttributeError(msg)
            elif self.stability_calculator is None:
                 logger.info("No se proporcionó StabilityCalculator. w_stab será 1.0.")

            logger.info("InstantaneousRewardCalculator inicializado.")

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.critical(f"Error crítico inicializando InstantaneousRewardCalculator: {e}", exc_info=True)
            raise RuntimeError("Fallo al inicializar InstantaneousRewardCalculator") from e # Fail-Fast

    def _load_and_validate_gaussian_params(self):
        """Carga y valida parámetros para el modo Gaussiano."""
        #logger.debug("Cargando y validando parámetros Gaussianos...")
        self.weights = self.gaussian_params.get('weights', {})
        self.scales = self.gaussian_params.get('scales', {})
        if not isinstance(self.weights, dict): raise TypeError("gaussian_params.weights debe ser dict.")
        if not isinstance(self.scales, dict): raise TypeError("gaussian_params.scales debe ser dict.")

        self.state_indices = {'cart_position': 0, 'cart_velocity': 1, 'angle': 2, 'angular_velocity': 3}
        # Claves requeridas para pesos y escalas
        required_keys = list(self.state_indices.keys()) + ['force', 'time']

        valid = True
        # Validar pesos (default 0)
        for key in required_keys:
            w = self.weights.get(key, 0.0)
            if not isinstance(w, (int, float)):
                logger.error(f"Gaussian: Peso para '{key}' ({w}) no numérico. Usando 0.0.")
                self.weights[key] = 0.0; valid = False
            else: self.weights[key] = float(w)
        # Validar escalas (default 1, debe ser > 0)
        for key in required_keys:
            s = self.scales.get(key, 1.0)
            if not isinstance(s, (int, float)) or s <= 0:
                logger.error(f"Gaussian: Escala para '{key}' ({s}) debe ser número positivo. Usando 1.0.")
                self.scales[key] = 1.0; valid = False
            else: self.scales[key] = float(s)

        if not valid:
            raise ValueError("Parámetros inválidos encontrados en gaussian_params.")
        #logger.debug(f"Params Gaussianos validados. Weights: {self.weights}, Scales: {self.scales}")


    # --- Implementación de RewardFunction Interface ---

    def calculate(self, state: Any, action: Any, next_state: Any, t: float) -> Tuple[float, float]:
        """Calcula reward_value y w_stab."""
        stability_score = 1.0 # Default w_stab
        reward_value = 0.0    # Default reward

        # 1. Calcular w_stab si es posible y está configurado
        if self.stability_calculator:
            try:
                # Usar interfaz, asume que devuelve float
                w_stab_calc = self.stability_calculator.calculate_instantaneous_stability(next_state)
                # Asegurar valor finito y en [0, 1]
                if pd.notna(w_stab_calc) and np.isfinite(w_stab_calc):
                    stability_score = float(np.clip(w_stab_calc, 0.0, 1.0))
                else:
                    logger.warning(
                        f"StabilityCalculator ({type(self.stability_calculator).__name__}) "
                        f"devolvió w_stab inválido ({w_stab_calc}). Usando 0.0."
                    )
                    stability_score = 0.0
            except Exception as e:
                logger.error(
                    f"Error calculando w_stab desde {type(self.stability_calculator).__name__}: {e}",
                    exc_info=True
                )
                stability_score = 0.0 # Default a 0 en error grave de cálculo de w_stab
        # else: stability_score permanece 1.0 (default si no hay calculator)

        # 2. Calcular Reward Value según el método configurado
        try:
            if self.method == 'gaussian':
                # 7.4: Validar estado y acción
                if not isinstance(next_state, (np.ndarray, list)) or len(next_state) < 4:
                     logger.warning(f"Gaussian: next_state inválido o corto: {next_state}. Reward=0.")
                     return 0.0, stability_score
                if pd.isna(action) or not np.isfinite(action):
                     logger.warning(f"Gaussian: action inválida (NaN/inf): {action}. Usando action=0 para reward.")
                     action = 0.0

                # Helper exp seguro
                def safe_exp(arg):
                    try: 
                        return math.exp(-min(float(arg)**2, 700.0))
                    except: return 0.0

                reward_calc = 0.0
                # Términos de estado
                for key, index in self.state_indices.items():
                    val = next_state[index]
                    if pd.isna(val) or not np.isfinite(val): # Saltar si estado es inválido
                        logger.warning(f"Gaussian: Valor inválido para '{key}' en next_state: {val}. Término será 0.")
                        continue
                    reward_calc += self.weights[key] * safe_exp(val / self.scales[key])
                # Términos de acción y tiempo
                reward_calc += self.weights['force'] * safe_exp(float(action) / self.scales['force'])
                reward_calc += self.weights['time'] * safe_exp(float(t) / self.scales['time'])

                reward_value = float(reward_calc) if np.isfinite(reward_calc) else 0.0

            elif self.method == 'stability_calculator':
                if self.stability_calculator: # Re-chequear por si acaso, aunque init debería fallar
                    reward_calc = self.stability_calculator.calculate_stability_based_reward(next_state) # type: ignore[union-attr]
                    reward_value = float(reward_calc) if pd.notna(reward_calc) and np.isfinite(reward_calc) else 0.0
                else:
                    # Esto indica un error de configuración si se llega aquí
                    logger.error("Método 'stability_calculator' pero no hay instancia de calculator. Reward=0.")
                    reward_value = 0.0
            # else: método desconocido ya manejado en init

        except IndexError as e: logger.error(f"Reward Calc ({self.method}): IndexError acceso estado/acción: {e}"); reward_value = 0.0
        except KeyError as e: logger.error(f"Reward Calc ({self.method}): KeyError acceso config (weights/scales): {e}"); reward_value = 0.0
        except Exception as e: logger.error(f"Reward Calc ({self.method}): Error inesperado: {e}", exc_info=True); reward_value = 0.0

        # logger.debug(f"Calculate Result: Reward={reward_value:.4f}, Stability={stability_score:.4f}")
        return reward_value, stability_score

    def update_calculator_stats(self, episode_metrics_dict: Dict, current_episode: int):
        """Delega actualización de stats al stability calculator."""
        if self.stability_calculator and hasattr(self.stability_calculator, 'update_reference_stats'):
            try:
                # logger.debug(f"Delegando update_stats a {type(self.stability_calculator).__name__}")
                self.stability_calculator.update_reference_stats(episode_metrics_dict, current_episode)
            except Exception as e:
                logger.error(f"Error llamando update_reference_stats en calculator: {e}", exc_info=True)
        # else: logger.debug("No stability calculator or no update_reference_stats method.")