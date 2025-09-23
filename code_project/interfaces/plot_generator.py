# interfaces/plot_generator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union # List, Union no se usan aquí pero son comunes
import pandas as pd # No se usa directamente aquí, pero las implementaciones sí.

class PlotGenerator(ABC):
    """
    Interface for components that generate and save individual plot files
    based on configuration and data files found in a results folder.
    """

    @abstractmethod
    def generate_plot(self,
                      plot_config_data: Dict[str, Any], # 'plot_config_data'
                      output_root_path_plot: str # 'output_root_path_plot' (nuevo nombre para results_folder)
                     ):
        """
        Generates and saves a single plot based on the provided configuration,
        loading the necessary data from files within the output_root_path_plot.

        Args:
            plot_config_data (Dict[str, Any]): Configuration dictionary for the specific plot,
                                          including 'type', 'source_file' (e.g., summary Excel, heatmap data), 
                                          variables to plot, styling options, output_filename, etc.
            output_root_path_plot (str): The absolute path to the root folder containing the
                                  result files (e.g., episodes_summary_data.xlsx, 
                                  detailed JSONs if needed, heatmap_data.xlsx) required for the plot.

        Raises:
            FileNotFoundError: If required data files are not found within output_root_path_plot.
            ValueError: If plot_config_data is invalid or data is unsuitable for the plot type.
            NotImplementedError: If the plot type specified in plot_config_data is not supported.
            Exception: For other errors during data loading, plot generation, or saving.
        """
        pass