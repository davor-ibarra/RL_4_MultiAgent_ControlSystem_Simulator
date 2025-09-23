import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker # For MaxNLocator
import os
import numpy as np
import pandas as pd
import logging
import traceback
import gc # Garbage collector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# == Helper Functions ==
# ==============================================================================

def _apply_common_mpl_styles(ax, fig, plot_cfg):
    """Applies common matplotlib styling options from config."""
    cfg = plot_cfg.get('config', {}) # Get the inner config dict
    
    # Titles and Labels
    ax.set_title(cfg.get('title', ''), fontsize=cfg.get('title_fontsize', 14))
    ax.set_xlabel(cfg.get('xlabel', ''), fontsize=cfg.get('xlabel_fontsize', 12))
    ax.set_ylabel(cfg.get('ylabel', ''), fontsize=cfg.get('ylabel_fontsize', 12))

    # Tick
    xtick_rotation = cfg.get('xtick_rotation', 0)
    tick_fs = cfg.get('tick_fontsize', 10)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fs * 0.8) # Optional minor ticks

    # Grid
    grid_visible = cfg.get('grid_on', False)
    ax.grid(visible=grid_visible, linestyle='--', alpha=0.6)

    # Legend
    legend_pos = cfg.get('legend_pos', 'best')
    legend_fs = cfg.get('legend_fontsize', 'small')

    # Axis Limits (Handle None for auto-scaling)
    xmin, xmax = cfg.get('xmin'), cfg.get('xmax')
    ymin, ymax = cfg.get('ymin'), cfg.get('ymax')
    if xmin is not None or xmax is not None: ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None: ax.set_ylim(bottom=ymin, top=ymax)

    # Tick
    num_xticks = cfg.get('num_xticks')
    num_yticks = cfg.get('num_yticks')
    if num_xticks is not None and num_xticks > 0:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=num_xticks, prune='both', integer=True if ax.get_xlabel().lower()=='episode' else False))
    if num_yticks is not None and num_yticks > 0:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=num_yticks, prune='both'))
    # Rotate X-axis labels if configured
    if xtick_rotation != 0: # Aplicar solo si la rotación no es cero
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right", rotation_mode="anchor")
    else:
        # Asegurar que la alineación sea horizontal por defecto si no hay rotación
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Legend
    if cfg.get('show_legend', False):
        legend_title = cfg.get('legend_title', None)
        # Handle 'outside' legend placement
        if legend_pos == 'outside':
            # Place legend outside the top right corner
            ax.legend(title=legend_title, bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=legend_fs) # <<< Usa legend_fs
            # Adjust subplot parameters to make space for the legend
        else:
            ax.legend(title=legend_title, loc=legend_pos, fontsize=legend_fs) # <<< Usa legend_fs
    elif ax.get_legend() is not None:
        # Remove legend if show_legend is false but one was automatically created
        ax.get_legend().remove()


    rect_right = 0.9 if legend_pos == 'outside' and cfg.get('show_legend', False) else 1
    # Usar try-except para tight_layout ya que a veces puede dar warnings/errors
    try:
        fig.tight_layout(rect=[0, 0, rect_right, 1]) # Adjust for outside legend
    except ValueError as e_tl:
        logging.warning(f"Could not apply tight_layout: {e_tl}")


def _extract_heatmap_data(detailed_data: list, x_var: str, y_var: str, filter_reasons: list = None) -> tuple[np.ndarray, np.ndarray]:
    """Extracts and concatenates X, Y data for heatmaps from detailed episode list, with optional filtering."""
    x_all, y_all = [], []

    filtered_episodes = detailed_data
    if filter_reasons:
        original_count = len(filtered_episodes)
        filtered_episodes = [ep for ep in detailed_data if ep.get('termination_reason') in filter_reasons]
        logging.info(f"Heatmap filtering: Kept {len(filtered_episodes)}/{original_count} episodes with reasons {filter_reasons}.")
        if not filtered_episodes:
            logging.warning(f"No episodes left after filtering for reasons: {filter_reasons}")
            return np.array([]), np.array([])

    for episode in filtered_episodes:
        x_raw = episode.get(x_var)
        y_raw = episode.get(y_var)

        if isinstance(x_raw, (list, np.ndarray)) and isinstance(y_raw, (list, np.ndarray)):
            min_len = min(len(x_raw), len(y_raw))
            if min_len > 0:
                x_num = pd.to_numeric(x_raw[:min_len], errors='coerce')
                y_num = pd.to_numeric(y_raw[:min_len], errors='coerce')
                valid_mask = np.isfinite(x_num) & np.isfinite(y_num)
                if np.any(valid_mask):
                    x_all.append(x_num[valid_mask])
                    y_all.append(y_num[valid_mask])

    if not x_all:
        return np.array([]), np.array([])

    return np.concatenate(x_all), np.concatenate(y_all)


def _save_plot(fig, results_folder, filename_base):
    """Saves the plot figure to the specified folder."""
    try:
        filepath = os.path.join(results_folder, filename_base)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename_base}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory
        gc.collect() # Explicitly call garbage collector


# ==============================================================================
# == Generic Plotting Functions ==
# ==============================================================================

def plot_generic_bar(data: pd.DataFrame, plot_cfg: dict, results_folder: str):
    """Generates a generic bar plot based on configuration."""
    cfg = plot_cfg.get('config', {})
    variable = plot_cfg.get('variable')
    filename = plot_cfg.get('output_filename', f"bar_{variable}.png")
    group_size = cfg.get('group_size') # Optional for grouping episodes

    if data is None or data.empty or not variable or variable not in data.columns:
        logging.warning(f"Cannot generate bar plot for '{variable}'. Data missing or invalid.")
        return

    logging.info(f"Generating bar plot for: {variable}")
    fig, ax = None, None
    try:
        fig, ax = plt.subplots(figsize=(cfg.get('figsize_w', 10), cfg.get('figsize_h', 5)))

        # --- Data Preparation ---
        df = data.copy()
        plot_data = None

        if group_size and 'episode' in df.columns:
            # Assumes categorical variable like termination_reason grouped by episode
            df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
            df = df.dropna(subset=['episode'])
            if df.empty: raise ValueError("No valid numeric episodes for grouping.")
            df['episode_group'] = (df['episode'] // group_size) * group_size
            plot_data = df.groupby(['episode_group', variable]).size().unstack(fill_value=0)
            cfg['xlabel'] = cfg.get('xlabel', 'Episode Group Start') # Auto-set label if grouped
            cfg['ylabel'] = cfg.get('ylabel', 'Count')
        else:
            # Simple counts of the variable
            plot_data = df[variable].value_counts().sort_index()
            cfg['xlabel'] = cfg.get('xlabel', variable.replace('_', ' ').title())
            cfg['ylabel'] = cfg.get('ylabel', 'Frequency')

        if plot_data is None or plot_data.empty:
            logging.warning(f"No data to plot for bar chart '{variable}' after processing.")
            if fig: plt.close(fig)
            return

        # --- Plotting ---
        cmap = cfg.get('cmap', 'hot')
        bar_width = cfg.get('bar_width', 0.8)

        if isinstance(plot_data, pd.DataFrame): # Stacked bar for grouped data
            plot_data.plot(kind='bar', stacked=True, ax=ax, colormap=cmap, width=bar_width)
            cfg['legend_title'] = variable.replace('_',' ').title() # Auto legend title
        else: # Simple bar for counts
            plot_data.plot(kind='bar', ax=ax, color=plt.get_cmap(cmap)(np.linspace(0, 1, len(plot_data))), width=bar_width)
            cfg['show_legend'] = cfg.get('show_legend', False) # Legend less useful here

        # --- Styling ---
        _apply_common_mpl_styles(ax, fig, plot_cfg)

        _save_plot(fig, results_folder, filename)

    except Exception as e:
        logging.error(f"Error generating bar plot '{filename}': {e}\n{traceback.format_exc()}")
        if fig: plt.close(fig); gc.collect()


def plot_generic_line(data: pd.DataFrame, plot_cfg: dict, results_folder: str):
    """Generates a generic line plot based on configuration."""
    cfg = plot_cfg.get('config', {})
    x_var = plot_cfg.get('x_variable')
    y_var = plot_cfg.get('y_variable')
    filename = plot_cfg.get('output_filename', f"line_{y_var}_vs_{x_var}.png")
    filter_reasons = cfg.get('filter_termination_reason') # List or None

    if data is None or data.empty or not x_var or not y_var or x_var not in data.columns or y_var not in data.columns:
        logging.warning(f"Cannot generate line plot for Y='{y_var}' vs X='{x_var}'. Data or variables missing.")
        return

    logging.info(f"Generating line plot for: {y_var} vs {x_var}")
    fig, ax = None, None
    try:
        fig, ax = plt.subplots(figsize=(cfg.get('figsize_w', 12), cfg.get('figsize_h', 6)))

        # --- Data Preparation ---
        df = data.copy()
        if filter_reasons and 'termination_reason' in df.columns:
            original_count = len(df)
            df = df[df['termination_reason'].isin(filter_reasons)]
            logging.info(f"Line plot filtering: Kept {len(df)}/{original_count} episodes with reasons {filter_reasons}.")

        df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
        df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
        df = df.dropna(subset=[x_var, y_var]).sort_values(by=x_var)

        if df.empty:
            logging.warning(f"No valid data points left for line plot '{filename}' after filtering/cleaning.")
            if fig: plt.close(fig)
            return

        x_data = df[x_var]
        y_data = df[y_var]

        # --- Plotting ---
        ax.plot(x_data, y_data,
                color=cfg.get('line_color', '#1f77b4'),
                linewidth=cfg.get('line_width', 1.5),
                linestyle=cfg.get('line_style', '-'),
                marker=cfg.get('marker_style', ''),
                markersize=cfg.get('marker_size', 0),
                markerfacecolor=cfg.get('marker_color', '#ff7f0e'),
                markeredgecolor=cfg.get('marker_color', '#ff7f0e'))

        # --- Styling ---
        # Auto-set labels if not provided
        cfg['xlabel'] = cfg.get('xlabel', x_var.replace('_', ' ').title())
        cfg['ylabel'] = cfg.get('ylabel', y_var.replace('_', ' ').title())
        _apply_common_mpl_styles(ax, fig, plot_cfg) # Apply common styles last

        _save_plot(fig, results_folder, filename)

    except Exception as e:
        logging.error(f"Error generating line plot '{filename}': {e}\n{traceback.format_exc()}")
        if fig: plt.close(fig); gc.collect()


def plot_generic_heatmap(detailed_data: list, plot_cfg: dict, results_folder: str):
    """Generates a generic heatmap plot based on configuration."""
    cfg = plot_cfg.get('config', {})
    x_var = plot_cfg.get('x_variable')
    y_var = plot_cfg.get('y_variable')
    filename = plot_cfg.get('output_filename', f"heatmap_{y_var}_vs_{x_var}.png")
    filter_reasons = cfg.get('filter_termination_reason') # List or None

    if not detailed_data or not x_var or not y_var:
        logging.warning(f"Cannot generate heatmap for Y='{y_var}' vs X='{x_var}'. Detailed data or variables missing.")
        return

    logging.info(f"Generating heatmap for: {y_var} vs {x_var}")
    fig, ax = None, None
    try:
        # --- Data Preparation ---
        x_data, y_data = _extract_heatmap_data(detailed_data, x_var, y_var, filter_reasons)

        if len(x_data) == 0 or len(y_data) == 0:
            logging.warning(f"No valid data points found for heatmap '{filename}' after extraction/filtering.")
            return

        fig, ax = plt.subplots(figsize=(cfg.get('figsize_w', 10), cfg.get('figsize_h', 7)))

        # --- Plotting ---
        bins = cfg.get('bins', 100)
        cmap = cfg.get('cmap', 'hot')
        log_scale = cfg.get('log_scale', False)
        cmin, cmax = cfg.get('cmin'), cfg.get('cmax') # For color bar limits

        # Determine range for histogram
        xmin_cfg, xmax_cfg = cfg.get('xmin'), cfg.get('xmax')
        ymin_cfg, ymax_cfg = cfg.get('ymin'), cfg.get('ymax')
        hist_range = None
        if xmin_cfg is not None and xmax_cfg is not None and ymin_cfg is not None and ymax_cfg is not None:
            hist_range = [[xmin_cfg, xmax_cfg], [ymin_cfg, ymax_cfg]]

        norm = mcolors.LogNorm(vmin=cmin, vmax=cmax) if log_scale else mcolors.Normalize(vmin=cmin, vmax=cmax)
        # cmin for hist2d is tricky with LogNorm, usually set vmin in norm instead
        hist_cmin = cmin if not log_scale and cmin is not None and cmin > 0 else None

        counts, xedges, yedges, img = ax.hist2d(
            x_data, y_data,
            bins=bins,
            cmap=cmap,
            norm=norm,
            range=hist_range,
            cmin=hist_cmin
        )

        # --- Colorbar ---
        cbar = fig.colorbar(img, ax=ax)
        cbar_label = cfg.get('clabel', 'Frequency') + (' (Log Scale)' if log_scale else '')
        cbar.set_label(cbar_label, fontsize=cfg.get('clabel_fontsize', 10))
        cbar.ax.tick_params(labelsize=cfg.get('tick_fontsize', 10))


        # --- Styling ---
        # Auto-set labels if not provided
        cfg['xlabel'] = cfg.get('xlabel', x_var.replace('_', ' ').title())
        cfg['ylabel'] = cfg.get('ylabel', y_var.replace('_', ' ').title())
        # Apply common styles AFTER hist2d and colorbar
        _apply_common_mpl_styles(ax, fig, plot_cfg)

        # PROBAR PARA PLOTS SIN GRID
        grid_visible = cfg.get('grid_on', False)
        ax.grid(visible=grid_visible, linestyle='--', alpha=0.6)

        _save_plot(fig, results_folder, filename)

    except Exception as e:
        logging.error(f"Error generating heatmap '{filename}': {e}\n{traceback.format_exc()}")
        if fig: plt.close(fig); gc.collect()


# ==============================================================================
# == Plotting Dispatcher ==
# ==============================================================================

def generate_plots(plot_configs: list, summary_df: pd.DataFrame, detailed_data: list, results_folder: str):
    """
    Generates plots based on a list of configurations.

    Args:
        plot_configs (list): A list of dictionaries, each defining a plot.
                             Expected keys: 'type', 'enabled', 'source', 'config',
                             'variable' (for bar), 'x_variable'/'y_variable' (for line/heatmap),
                             'output_filename'.
        summary_df (pd.DataFrame): The summary data across all episodes.
        detailed_data (list): A list of dictionaries, each holding detailed data for one episode.
        results_folder (str): The folder to save the plots in.
    """
    if not plot_configs:
        logging.info("No plot configurations provided. Skipping plot generation.")
        return

    plot_functions = {
        'bar': plot_generic_bar,
        'line': plot_generic_line,
        'heatmap': plot_generic_heatmap
    }

    for i, plot_cfg in enumerate(plot_configs):
        if not isinstance(plot_cfg, dict):
            logging.warning(f"Skipping plot config #{i+1}: Invalid format (not a dictionary).")
            continue

        plot_type = plot_cfg.get('type')
        is_enabled = plot_cfg.get('enabled', True) # Default to enabled if key missing

        if not is_enabled:
            logging.info(f"Skipping plot config #{i+1} ('{plot_cfg.get('output_filename', plot_type)}') - disabled.")
            continue

        if plot_type not in plot_functions:
            logging.warning(f"Skipping plot config #{i+1}: Unknown plot type '{plot_type}'. Available types: {list(plot_functions.keys())}")
            continue

        plot_func = plot_functions[plot_type]
        source = plot_cfg.get('source')
        data_to_use = None

        if source == 'summary':
            data_to_use = summary_df
            if data_to_use is None or data_to_use.empty:
                 logging.warning(f"Skipping {plot_type} plot '{plot_cfg.get('output_filename', 'N/A')}': Summary data source selected, but summary_df is empty or None.")
                 continue
        elif source == 'detailed':
            data_to_use = detailed_data
            if not data_to_use:
                 logging.warning(f"Skipping {plot_type} plot '{plot_cfg.get('output_filename', 'N/A')}': Detailed data source selected, but detailed_data list is empty.")
                 continue
        else:
            logging.warning(f"Skipping plot config #{i+1}: Invalid or missing 'source' ('summary' or 'detailed').")
            continue

        logging.info(f"--- Generating Plot {i+1}/{len(plot_configs)} (Type: {plot_type}, Source: {source}) ---")
        try:
            # Call the appropriate plotting function
            if source == 'summary':
                 plot_func(data=data_to_use, plot_cfg=plot_cfg, results_folder=results_folder)
            elif source == 'detailed':
                 plot_func(detailed_data=data_to_use, plot_cfg=plot_cfg, results_folder=results_folder)

        except Exception as e:
            # Log error but continue with other plots
            logging.error(f"Failed to generate plot from config #{i+1} ({plot_cfg.get('output_filename', plot_type)}): {e}", exc_info=True)