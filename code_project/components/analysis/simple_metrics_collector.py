from interfaces.metrics_collector import MetricsCollector
from collections import defaultdict

class SimpleMetricsCollector(MetricsCollector):
    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, metric_name, metric_value):
        self.metrics[metric_name].append(metric_value)

    def get_metrics(self):
        return dict(self.metrics)

    def reset(self):
        self.metrics.clear()
