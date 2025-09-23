import matplotlib.pyplot as plt
from interfaces.report_generator import ReportGenerator
from matplotlib.backends.backend_pdf import PdfPages

class PDFReportGenerator(ReportGenerator):
    def create_report(self, analyzed_metrics, file_path):
        with PdfPages(file_path) as pdf:
            for metric, stats in analyzed_metrics.items():
                fig, ax = plt.subplots()
                labels = stats.keys()
                values = stats.values()
                ax.bar(labels, values, color='skyblue')
                ax.set_title(f'Resumen de {metric}')
                ax.set_ylabel('Valor')
                ax.set_xlabel('Estadística')

                pdf.savefig(fig)
                plt.close(fig)
