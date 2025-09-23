import json
import os

class ConfigValidationError(Exception):
    pass

class ConfigurationManager:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = None

    def load_config(self):
        if not os.path.exists(self.config_file_path):
            print(os.path.exists(self.config_file_path))
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_file_path}")
        with open(self.config_file_path, 'r') as f:
            self.config = json.load(f)
        return self.config

class ValidationManager:
    def __init__(self, config):
        self.config = config

    def validate(self):
        required_sections = ['simulation', 'physical', 'controller', 'rl', 'state']
        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"Falta la sección '{section}' en la configuración.")
        # Se pueden agregar validaciones adicionales (rango, tipos, etc.)
        return self.config

class ParameterManager:
    def __init__(self, config):
        self.config = config

    def get_parameters(self, section):
        return self.config.get(section, {})
