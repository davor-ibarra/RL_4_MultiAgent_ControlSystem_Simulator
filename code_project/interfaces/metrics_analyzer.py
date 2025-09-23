from abc import ABC, abstractmethod

class MetricsAnalyzer(ABC):
    @abstractmethod
    def evaluate_performance(self, metrics):
        pass
