from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass
import json
import numpy as np
from interfaces import ConfigValidator

@dataclass
class ValidationResult:
    """Stores the result of a configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ConfigurationManager:
    """Manages and validates system configurations."""
    
    def __init__(self, schema_path: Optional[str] = None):
        self._config: Dict[str, Any] = {}
        self._validators: Dict[str, ConfigValidator] = {}
        self._schema: Dict[str, Any] = {}
        
        if schema_path:
            self.load_schema(schema_path)
    
    def load_schema(self, schema_path: str) -> None:
        """Load configuration schema from file."""
        with open(schema_path, 'r') as f:
            self._schema = json.load(f)
    
    def register_validator(self, section: str, validator: ConfigValidator) -> None:
        """Register a validator for a specific configuration section."""
        self._validators[section] = validator
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            self._config = json.load(f)
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration or specific section."""
        if section is None:
            return self._config
        return self._config.get(section, {})
    
    def set_config(self, config: Dict[str, Any], section: Optional[str] = None) -> None:
        """Set configuration or update specific section."""
        if section is None:
            self._config = config
        else:
            if section not in self._config:
                self._config[section] = {}
            self._config[section].update(config)
    
    def validate(self, section: Optional[str] = None) -> ValidationResult:
        """Validate configuration or specific section."""
        errors = []
        warnings = []
        
        if section is not None:
            if section in self._validators:
                return self._validate_section(section)
            else:
                return ValidationResult(False, [f"No validator found for section: {section}"], [])
        
        # Validate all sections
        for section_name, validator in self._validators.items():
            result = self._validate_section(section_name)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_section(self, section: str) -> ValidationResult:
        """Validate a specific configuration section."""
        if section not in self._config:
            return ValidationResult(False, [f"Missing configuration section: {section}"], [])
        
        validator = self._validators[section]
        config = self._config[section]
        
        try:
            is_valid = validator.validate_section(config)
            errors = validator.get_validation_errors()
            return ValidationResult(is_valid, errors, [])
        except Exception as e:
            return ValidationResult(False, [f"Validation error in {section}: {str(e)}"], [])

class DefaultConfigValidator(ConfigValidator):
    """Default implementation of configuration validator."""
    
    def __init__(self, schema: Dict[str, Any]):
        self._schema = schema
        self._errors: List[str] = []
    
    def validate_system_config(self, config: Dict[str, Any]) -> bool:
        """Validate system configuration against schema."""
        self._errors = []
        return self._validate_against_schema(config, self._schema.get('system', {}), 'system')
    
    def validate_controller_config(self, config: Dict[str, Any]) -> bool:
        """Validate controller configuration against schema."""
        self._errors = []
        return self._validate_against_schema(config, self._schema.get('controller', {}), 'controller')
    
    def validate_agent_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration against schema."""
        self._errors = []
        return self._validate_against_schema(config, self._schema.get('agent', {}), 'agent')
    
    def validate_environment_config(self, config: Dict[str, Any]) -> bool:
        """Validate environment configuration against schema."""
        self._errors = []
        return self._validate_against_schema(config, self._schema.get('environment', {}), 'environment')
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self._errors
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any], 
                               path: str) -> bool:
        """Recursively validate configuration against schema."""
        for key, value_schema in schema.items():
            if key not in config:
                if value_schema.get('required', False):
                    self._errors.append(f"Missing required field: {path}.{key}")
                    return False
                continue
            
            value = config[key]
            value_type = value_schema.get('type')
            
            if value_type == 'number':
                if not isinstance(value, (int, float)):
                    self._errors.append(f"Invalid type for {path}.{key}: expected number")
                    return False
                
                if 'min' in value_schema and value < value_schema['min']:
                    self._errors.append(f"Value for {path}.{key} below minimum")
                    return False
                
                if 'max' in value_schema and value > value_schema['max']:
                    self._errors.append(f"Value for {path}.{key} above maximum")
                    return False
            
            elif value_type == 'string':
                if not isinstance(value, str):
                    self._errors.append(f"Invalid type for {path}.{key}: expected string")
                    return False
                
                if 'enum' in value_schema and value not in value_schema['enum']:
                    self._errors.append(f"Invalid value for {path}.{key}: must be one of {value_schema['enum']}")
                    return False
            
            elif value_type == 'array':
                if not isinstance(value, list):
                    self._errors.append(f"Invalid type for {path}.{key}: expected array")
                    return False
                
                if 'items' in value_schema:
                    for i, item in enumerate(value):
                        if not self._validate_against_schema({'item': item}, 
                                                           {'item': value_schema['items']}, 
                                                           f"{path}.{key}[{i}]"):
                            return False
            
            elif value_type == 'object':
                if not isinstance(value, dict):
                    self._errors.append(f"Invalid type for {path}.{key}: expected object")
                    return False
                
                if 'properties' in value_schema:
                    if not self._validate_against_schema(value, value_schema['properties'], 
                                                       f"{path}.{key}"):
                        return False
        
        return True

def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        'simulation': {
            'max_episodes': 1000,
            'dt': 0.01,
            'save_frequency': 100,
            'render': False
        },
        'system': {
            'type': None,  # Must be specified
            'parameters': {}
        },
        'controller': {
            'type': None,  # Must be specified
            'parameters': {}
        },
        'agent': {
            'type': None,  # Must be specified
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'batch_size': 64,
            'memory_size': 10000
        },
        'environment': {
            'type': None,  # Must be specified
            'parameters': {}
        },
        'reward': {
            'type': 'default',
            'parameters': {
                'weights': {
                    'stability': 1.0,
                    'effort': 0.1,
                    'time': 0.01
                }
            }
        }
    }
