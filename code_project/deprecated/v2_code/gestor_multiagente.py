from agente import AgenteFactory
from typing import List, Dict, Any

class GestorMultiagente:
    def __init__(self, config_agentes: List[Dict[str, Any]], variables_estado):
        self.agentes = []
        for config_agente in config_agentes:
            agente = AgenteFactory.crear_agente(config_agente, variables_estado)
            self.agentes.append(agente)

    def percibir_entorno(self, estado):
        for agente in self.agentes:
            agente.percibir_entorno(estado)

    def decidir_acciones(self):
        acciones = []
        for agente in self.agentes:
            if agente.esta_activo():
                accion = agente.decidir_accion()
            else:
                accion = None
            acciones.append(accion)
        return acciones

    def recibir_recompensas(self, recompensas, estados_siguientes):
        for agente, recompensa, estado_siguiente in zip(self.agentes, recompensas, estados_siguientes):
            agente.recibir_recompensa(recompensa, estado_siguiente)

    def actualizar_politicas(self):
        for agente in self.agentes:
            agente.actualizar_politica()

    def obtener_acciones_agentes(self):
        acciones_agentes = []
        for agente in self.agentes:
            if agente.esta_activo():
                accion = agente.decidir_accion()
                acciones_agentes.append(accion)
            else:
                acciones_agentes.append(None)
        return acciones_agentes

    def ajustar_hiperparametros(self, rendimientos):
        for agente, rendimiento in zip(self.agentes, rendimientos):
            # *** Implementar ajustes específicos para cada agente según el rendimiento
            pass