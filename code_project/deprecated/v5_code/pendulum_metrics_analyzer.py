import numpy as np
from code_project.deprecated.v5_code.metrics_analyzer import MetricsAnalyzer

class PendulumMetricsAnalyzer(MetricsAnalyzer):
    def evaluate_performance(self, metrics):
        results = {}
        for key, values in metrics.items():
            results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return results
