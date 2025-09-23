import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ConfiguracionGeneral:
    tiempo_total: float
    paso_tiempo: float
    parametros: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tiempo_total, (int, float)):
            raise TypeError("El 'tiempo_total' debe ser un número.")
        if self.tiempo_total <= 0:
            raise ValueError("El 'tiempo_total' debe ser mayor que cero.")
        if not isinstance(self.paso_tiempo, (int, float)):
            raise TypeError("El 'paso_tiempo' debe ser un número.")
        if self.paso_tiempo <= 0:
            raise ValueError("El 'paso_tiempo' debe ser mayor que cero.")
        if not isinstance(self.parametros, dict):
            raise TypeError("Los 'parametros' deben ser un diccionario.")

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'ConfiguracionGeneral':
        return ConfiguracionGeneral(
            tiempo_total=config.get('tiempo_total', 10.0),
            paso_tiempo=config.get('paso_tiempo', 0.01),
            parametros=config.get('parametros', {})
        )

@dataclass
class ConfiguracionAgente:
    tipo_agente: str
    parametros: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tipo_agente, str):
            raise TypeError("El 'tipo_agente' debe ser una cadena.")
        if not isinstance(self.parametros, dict):
            raise TypeError("Los 'parametros' deben ser un diccionario.")

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'ConfiguracionAgente':
        return ConfiguracionAgente(
            tipo_agente=config.get('tipo_agente', 'AgenteRL'),
            parametros=config.get('parametros', {})
        )

@dataclass
class ConfiguracionSistemaDinamico:
    tipo_sistema: str
    parametros: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tipo_sistema, str):
            raise TypeError("El 'tipo_sistema' debe ser una cadena.")
        if not isinstance(self.parametros, dict):
            raise TypeError("Los 'parametros' deben ser un diccionario.")

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'ConfiguracionSistemaDinamico':
        return ConfiguracionSistemaDinamico(
            tipo_sistema=config.get('tipo_sistema', 'PenduloInvertido'),
            parametros=config.get('parametros', {})
        )

@dataclass
class ConfiguracionControlador:
    tipo_controlador: str
    parametros: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tipo_controlador, str):
            raise TypeError("El 'tipo_controlador' debe ser una cadena.")
        if not isinstance(self.parametros, dict):
            raise TypeError("Los 'parametros' deben ser un diccionario.")

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'ConfiguracionControlador':
        return ConfiguracionControlador(
            tipo_controlador=config.get('tipo_controlador', 'ControladorPID'),
            parametros=config.get('parametros', {})
        )

@dataclass
class Configuracion:
    general: ConfiguracionGeneral
    agentes: List[Dict[str, Any]]  # Lista de configuraciones de agentes
    sistema_dinamico: ConfiguracionSistemaDinamico
    controlador: ConfiguracionControlador

    @staticmethod
    def cargar_desde_archivo(ruta_archivo: str) -> 'Configuracion':
        with open(ruta_archivo, 'r') as archivo:
            datos = json.load(archivo)

        general_config = ConfiguracionGeneral.from_dict(datos.get('general', {}))
        agentes_config = datos.get('agentes', [])
        sistema_config = ConfiguracionSistemaDinamico.from_dict(datos.get('sistema_dinamico', {}))
        controlador_config = ConfiguracionControlador.from_dict(datos.get('controlador', {}))

        return Configuracion(
            general=general_config,
            agentes=agentes_config,
            sistema_dinamico=sistema_config,
            controlador=controlador_config
        )