from abc import ABC, abstractmethod

class ReportGenerator(ABC):
    @abstractmethod
    def create_report(self, evaluated_metrics, report_path):
        pass
