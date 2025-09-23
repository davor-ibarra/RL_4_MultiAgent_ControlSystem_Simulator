from typing import Dict, Any, List, Union
from schema import Schema, And, Use, Optional, SchemaError

class ConfigValidator:
    """Valida la estructura y los tipos del archivo de configuración."""

    def __init__(self, schema_definition: Dict):
        self.schema = Schema(schema_definition)

    def validate(self, config: Dict[str, Any]) -> "ValidationResult":
        """Valida la configuración contra el esquema."""
        try:
            validated_config = self.schema.validate(config)
            return ValidationResult(True, validated_config, "")
        except SchemaError as e:
            return ValidationResult(False, {}, str(e))


#@dataclass
class ValidationResult:
    """Resultado de la validación."""
    is_valid: bool
    validated_config: Dict[str, Any]
    error_message: str = ""

# Definición del esquema de validación
validation_schema = {
    "simulation": {
        "max_episodes": And(Use(int), lambda n: n > 0),
        "time_step": And(Use(float), lambda n: n > 0),
        "decision_interval": And(Use(int), lambda n: n > 0),
        "episodes_per_file": And(Use(int), lambda n: n > 0),
        Optional("results_folder"): str
    },
    "system": {
        "type": str,
        "parameters": {
            "m1": And(Use(float), lambda n: n > 0),
            "m2": And(Use(float), lambda n: n > 0),
            "l": And(Use(float), lambda n: n > 0),
            "g": And(Use(float), lambda n: n > 0),
            "cr": Use(float),
            "ca": Use(float),
            "x0": [Use(float)],
             "angle_limit": Use(float),
            "cart_limit": Use(float)
        }
    },
    "controller": {
        "type": str,
        "parameters": {
            "kp": Use(float),
            "ki": Use(float),
            "kd": Use(float),
            "setpoint": Use(float),
            "dt": And(Use(float), lambda n: n > 0),
            "gain_step": Use(float),
            "reset_gains_each_episode": Use(bool),
            "kp_min": Use(float),
            "kp_max": Use(float),
            "ki_min": Use(float),
            "ki_max": Use(float),
            "kd_min": Use(float),
            "kd_max": Use(float)
        }
    },
    "rl": {
        "agent_type": str,
        Optional("parameters"): {
            Optional("enabled"): Use(bool)
        },
        Optional("success_reward_factor"): And(Use(float), lambda n: n > 0),
        Optional("q_table_save_frequency"): And(Use(int), lambda n: n > 0), #Nueva validación
        Optional("sub_agents"): {
            str: { #Valida que las keys puedan ser cualquier string
                "type": str,
                Optional("parameters"): {
                    Optional("actions"): [Use(float)],
                    Optional("learning_rate"): And(Use(float), lambda n: 0 < n < 1),
                    Optional("discount_factor"): And(Use(float), lambda n: 0 < n < 1),
                    Optional("exploration_rate"): And(Use(float), lambda n: 0 < n <= 1),
                    Optional("exploration_decay"): And(Use(float), lambda n: 0 < n < 1),
                    Optional("min_exploration_rate"): And(Use(float), lambda n: 0 < n <= 1),
                    Optional("variables_to_include"): [str],
                    Optional("q_table_filename"): str
                }
            }
        }
    },
    "reward": {
        "type": str,
        "parameters": {
            "state_variables": [str],
            "std_devs": [And(Use(float), lambda n: n > 0)],
            "weights": [Use(float)],
            "use_next_state": Use(bool)
        }
    },
    "state": {
        str: {  # Permite cualquier clave de variable de estado (x, x_dot, theta, etc.)
            "enabled": Use(bool),
            Optional("min"): Use(float),
            Optional("max"): Use(float),
            Optional("bins"): And(Use(int), lambda n: n > 0)
        }
    }
}