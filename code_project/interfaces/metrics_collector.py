from abc import ABC, abstractmethod

class MetricsCollector(ABC):
    @abstractmethod
    def log(self, metric_name, metric_value):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def reset(self):
        pass
