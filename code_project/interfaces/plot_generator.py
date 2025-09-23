# interfaces/plot_generator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import pandas as pd

class PlotGenerator(ABC):
    """
    Interface for components that generate and save individual plot files
    based on configuration and data files found in a results folder.
    """

    @abstractmethod
    def generate_plot(self,
                      plot_config: Dict[str, Any],
                      results_folder: str):
        """
        Generates and saves a single plot based on the provided configuration,
        loading the necessary data from files within the results_folder.

        Args:
            plot_config (Dict[str, Any]): Configuration dictionary for the specific plot,
                                          including 'type', 'source', variables, styling,
                                          output_filename, etc.
            results_folder (str): The absolute path to the folder containing the
                                  result files (e.g., summary Excel, detailed JSONs,
                                  heatmap data Excel) needed for the plot.

        Raises:
            FileNotFoundError: If required data files are not found.
            ValueError: If plot_config is invalid or data is unsuitable.
            NotImplementedError: If the plot type specified in plot_config is not supported.
            Exception: For errors during data loading, plot generation, or saving.
        """
        # (1.1) Firma modificada: Se elimina el argumento 'data'.
        # La implementación será responsable de cargar los datos necesarios
        # desde 'results_folder' basado en 'plot_config'.
        pass