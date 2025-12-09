from collections import defaultdict
from interfaces.metrics_collector import MetricsCollector
import numpy as np
import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

class ExtendedMetricsCollector(MetricsCollector):
    def __init__(self, data_save_config: Dict[str, Any]):
        self.collected_data: Dict[str, List[Any]] = defaultdict(list)
        self.current_episode_num: int = -1
        self._low_freq_cache: Dict[str, Any] = {}

        cfg = data_save_config.get('config', {})
        self._is_enabled_json = cfg.get('enabled_json_history', False)

        # --- Build Extraction and Summary Maps ---
        self._extraction_map: Dict[str, List[Dict]] = defaultdict(list)
        self._all_keys_to_log: Set[str] = set()
        self._low_freq_keys: Set[str] = set()
        self._summary_direct_cols: Set[str] = set()
        self._summary_stats_cols: Set[str] = set()

        component_configs = data_save_config.get('components', {})
        if isinstance(component_configs, dict):
            for source_component, metrics_list in component_configs.items():
                if not isinstance(metrics_list, list): continue
                for metric_dict in metrics_list:
                    if not isinstance(metric_dict, dict): continue
                    
                    param_name = metric_dict.get('params')
                    if not param_name: continue

                    out_targets = metric_dict.get('out', [])
                    freqs = metric_dict.get('freq', [])
                    output_key = metric_dict.get('alias', param_name)
                    
                    # 1. Determinar si la métrica necesita ser extraída.
                    needs_extraction = 'summary' in out_targets or 'json_history' in out_targets
                    if not needs_extraction: continue

                    # 2. Programar la extracción de la métrica.
                    task = {'param': param_name, 'source': source_component, 'alias': output_key}
                    for freq in freqs:
                        self._extraction_map[freq].append(task)
                    
                    self._all_keys_to_log.add(output_key)
                    
                    # 3. Identificar si es de baja frecuencia (para padding en json_history).
                    is_low_freq = 'on_decision_interval' in freqs or 'on_final' in freqs
                    if is_low_freq:
                        self._low_freq_keys.add(output_key)

                    # 4. Registrar directivas de resumen.
                    if 'summary' in out_targets:
                        self._summary_direct_cols.add(output_key)
                    if metric_dict.get('summary_stats', False):
                        self._summary_stats_cols.add(output_key)
        
        #logger.info(f"[EMC] Initialized. Total extraction tasks scheduled: {sum(len(v) for v in self._extraction_map.values())}")

    def _dynamic_extractor(self, param_name: str, source_name: str, context: Dict[str, Any]) -> Any:
        # 1. Chequeo de casos especiales y contexto directo
        if param_name == 'episode_id':
            return self.current_episode_num
        if param_name in context:
            return context[param_name]

        # 2. Búsqueda en diccionarios anidados del contexto
        nested_dicts = ['state', 'reward_info_for_learn', 'learn_metrics', 'actions_map']
        for dict_key in nested_dicts:
            if dict_key in context and isinstance(context[dict_key], dict) and param_name in context[dict_key]:
                return context[dict_key][param_name]

        # 3. Extracción desde el método estandarizado get_params_log()
        
        plural_source_key = source_name + 's' # ej. 'controller' -> 'controllers'
        
        # Opción A: La fuente es una colección de componentes (ej. 'controllers')
        if plural_source_key in context and isinstance(context[plural_source_key], dict):
            all_component_logs = {}
            for component in context[plural_source_key].values():
                if hasattr(component, 'get_params_log'):
                    try:
                        all_component_logs.update(component.get_params_log())
                    except Exception as e:
                        logger.debug(f"[EMC._dynamic_extractor] Error calling get_params_log in collection '{plural_source_key}' for '{param_name}': {e}")
            
            if param_name in all_component_logs:
                return all_component_logs[param_name]

        # Opción B: La fuente es un único componente (ej. 'agent')
        elif source_name in context:
            source_obj = context.get(source_name)
            if hasattr(source_obj, 'get_params_log'):
                try:
                    log_params = source_obj.get_params_log()
                    if param_name in log_params:
                        return log_params[param_name]
                except Exception as e:
                    logger.debug(f"[EMC._dynamic_extractor] Error calling get_params_log on '{source_name}' for '{param_name}': {e}")
        
        # Si después de todas las búsquedas no se encuentra, devuelve NaN.
        return np.nan

    def _update_cache(self, event: str, context: Dict[str, Any]):
        # Simplificado: solo actualiza la caché con los valores del evento actual.
        tasks = self._extraction_map.get(event, [])
        for task in tasks:
            value = self._dynamic_extractor(task['param'], task['source'], context)
            self._low_freq_cache[task['alias']] = value

    def log_on_episode_start(self, context: Dict[str, Any]):
        self.reset(context.get('episode_id', -1))

        # Prime the cache with initial values
        self._update_cache('on_decision_interval', context)
        self._update_cache('on_final', context)
        
        # Log the first step (t=0)
        context.setdefault('time', 0.0)
        context.setdefault('reward', 0.0)
        context.setdefault('control_action', 0.0)
        context.setdefault('stability_score', np.nan)
        self.log_on_step(context)

    def log_on_step(self, context: Dict[str, Any]):
        # Extraer métricas de alta frecuencia ('on_step')
        step_tasks = self._extraction_map.get('on_step', [])
        for task in step_tasks:
            key = task['alias']
            value = self._dynamic_extractor(task['param'], task['source'], context)
            self.collected_data[key].append(value)
            
        # Si se guarda historial JSON, rellenar métricas de baja frecuencia desde la caché.
        if self._is_enabled_json:
            for key in self._low_freq_keys:
                # Asegurarse de que la lista existe para evitar KeyErrors
                if key not in self.collected_data:
                    self.collected_data[key] = []
                self.collected_data[key].append(self._low_freq_cache.get(key, np.nan))

    def log_on_decision_boundary(self, context: Dict[str, Any]):
        self._update_cache('on_decision_interval', context)

    def log_on_episode_end(self, context: Dict[str, Any]):
        # 1. Extraer todos los valores finales.
        final_tasks = self._extraction_map.get('on_final', [])
        final_values = {}
        for task in final_tasks:
            final_values[task['alias']] = self._dynamic_extractor(task['param'], task['source'], context)

        # 2. Determinar la longitud del historial si el guardado JSON está activo.
        json_history_len = 0
        if self._is_enabled_json:
            time_data = self.collected_data.get('time')
            if time_data:
                json_history_len = len(time_data)

        # 3. Poblar self.collected_data con estos valores finales.
        for key, value in final_values.items():
            if self._is_enabled_json:
                # Si el historial JSON está habilitado, se crea una lista completa con el valor final
                # para asegurar que la longitud coincida con las métricas 'on_step'.
                # Esto es crucial para métricas como 'termination_reason'.
                if json_history_len > 0:
                    self.collected_data[key] = [value] * json_history_len
                else: # Si no hay historial (ej. episodio de 0 pasos), se añade como valor único.
                    self.collected_data[key].append(value)
            else:
                # Si no se guarda historial JSON, solo se necesita el valor final para el resumen.
                self.collected_data[key].append(value)

    def get_metrics(self) -> Dict[str, List[Any]]:
        # Simply return the collected data. The logging logic now ensures equal length lists.
        return dict(self.collected_data)
    
    def get_summary_directives(self) -> Tuple[Set[str], Set[str]]:
        """Returns the pre-processed summary directives."""
        return self._summary_direct_cols, self._summary_stats_cols

    def reset(self, episode_id: int):
        self.collected_data.clear()
        self.current_episode_num = episode_id
        self._low_freq_cache.clear()