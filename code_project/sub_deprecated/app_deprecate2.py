# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for Dynamic System Simulation Results.
Version: 2.1.0 (Unified Matplotlib Customization, Plot Download)

Focuses on visualization and basic analysis using Matplotlib.
Includes comprehensive customization for plots and PNG download capability.
Analysis Configurator tab is currently inactive.
"""

import streamlit as st
from streamlit_option_menu import option_menu
# import plotly.graph_objects as go # Keep commented unless needed elsewhere
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For LogNorm in heatmap
import sys
import traceback
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import io # Needed for plot download
import gc
from datetime import datetime

# Importar funciones desde app_utils.py
try:
    from app_utils import (
        load_folder_structure,
        load_metadata,
        load_summary_data,
        load_selected_episodes,
        resaltar_maximo,
        PendulumAnimator
    )
    utils_imports_ok = True
except ImportError as e:
     st.error(f"FATAL ERROR: Failed to import utilities from app_utils.py: {e}.")
     logging.critical(f"app_utils import failed: {e}", exc_info=True)
     utils_imports_ok = False
     # Use st.stop() within the main execution flow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------- Constantes y Configuración Inicial ---------------------------
RESULTS_FOLDER = "results_history"
# Subfolder for saving data exported from the dashboard (kept in case needed later)
# DASHBOARD_EXPORTS_SUBFOLDER = "dashboard_exports"

# def get_dynamic_exports_folder() -> Optional[str]:
#     """Gets the path to the 'dashboard_exports' subfolder for the current run."""
#     if st.session_state.get('selected_folder_path'):
#         exports_path = os.path.join(st.session_state.selected_folder_path, DASHBOARD_EXPORTS_SUBFOLDER)
#         try:
#             os.makedirs(exports_path, exist_ok=True)
#             return exports_path
#         except OSError as e:
#             st.error(f"Could not create dashboard exports directory: {exports_path}\n{e}")
#             return None
#     return None

def initialize_session_state():
    """Inicializa las variables en st.session_state si no existen."""
    defaults = {
        'folders': [],
        'selected_folder_path': None,
        'selected_folder_name': None,
        'metadata': None,
        'config': {},
        'summary_df': None,
        'loaded_episode_data': {'simulation_data': []},
        'available_episodes_in_summary': [],
        'selected_episode_numbers_to_load': [],
        'current_episode_index': 0,
        'available_loaded_episode_numbers': [],
        'plot_selections': {'x_param': 'time', 'y_params': []},
        'heatmap_needs_update': True,
        'heatmap_data_cache': None,
        'heatmap_params_cache': None,
        # States for Analysis Configurator (unused)
        'discovery_prepared_df_preview': None,
        'discovery_selected_vars_for_prep': [],
        'discovery_prep_config_saved_path': None,
        'discovery_analysis_configs_saved': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --------------------------- Sidebar Rendering (Adopted from v1.5 / app_deprecate) ---------------------------
# Using the robust sidebar logic from previous versions
def render_sidebar():
    """Renderiza la barra lateral y maneja la selección de carpetas y filtrado de datos."""
    st.sidebar.title("Simulation Results")
    # --- Folder Selection ---
    if st.sidebar.button("🔄 Refresh Folders"):
        st.cache_data.clear()
        keys_to_reset = list(st.session_state.keys()) # Get all keys
        # Keep essential streamlit keys if needed, otherwise clear all related to the app
        essential_st_keys = [] # Potentially add keys like 'query_params' if used
        for key in keys_to_reset:
            if key not in essential_st_keys:
                del st.session_state[key]
        initialize_session_state() # Re-initialize cleared state
        st.sidebar.success("Cache cleared and state reset. Select folder again.")
        st.rerun()

    if not st.session_state.get('folders'):
        try: st.session_state.folders = load_folder_structure(RESULTS_FOLDER)
        except Exception as e: st.sidebar.error(f"Error loading folder structure: {e}"); st.session_state.folders = []
    if not st.session_state.folders: st.sidebar.warning(f"No folders in '{RESULTS_FOLDER}'."); return

    current_selection_index = 0
    current_folder_name = st.session_state.get('selected_folder_name')
    if current_folder_name and current_folder_name in st.session_state.folders:
         try: current_selection_index = st.session_state.folders.index(current_folder_name)
         except ValueError: pass

    selected_folder_name_new = st.sidebar.selectbox("Select results folder:", st.session_state.folders, index=current_selection_index, key="folder_selector")
    if not selected_folder_name_new: return
    selected_folder_path_new = os.path.join(RESULTS_FOLDER, selected_folder_name_new)

    # --- Load Metadata/Summary if folder changed ---
    if selected_folder_path_new != st.session_state.get('selected_folder_path'):
        st.session_state.selected_folder_path = selected_folder_path_new
        st.session_state.selected_folder_name = selected_folder_name_new
        st.sidebar.info(f"Selected: {selected_folder_name_new}")
        with st.spinner("Loading metadata & summary..."):
            # Clear previous folder's data thoroughly
            keys_to_reset = [k for k in st.session_state if k != 'folders' and k!= 'selected_folder_path' and k!= 'selected_folder_name'] # Keep folder info
            for key in keys_to_reset: del st.session_state[key]
            initialize_session_state() # Re-initialize relevant parts
            # Set folder info again after re-init
            st.session_state.selected_folder_path = selected_folder_path_new
            st.session_state.selected_folder_name = selected_folder_name_new

            # Load new data
            st.session_state.metadata = load_metadata(st.session_state.selected_folder_path)
            st.session_state.config = st.session_state.metadata.get('config_parameters', {}) if st.session_state.metadata else {}
            if not st.session_state.config: st.sidebar.warning("Config not in metadata.")
            st.session_state.summary_df = load_summary_data(st.session_state.selected_folder_path)
            if st.session_state.summary_df is not None and 'episode' in st.session_state.summary_df.columns:
                 try:
                     unique_eps = pd.to_numeric(st.session_state.summary_df['episode'], errors='coerce').dropna().unique()
                     st.session_state.available_episodes_in_summary = sorted(unique_eps.astype(int)) if len(unique_eps) > 0 else []
                     if not st.session_state.available_episodes_in_summary: st.sidebar.warning("No valid episode numbers in summary.")
                 except Exception as e: st.sidebar.error(f"Err processing episodes: {e}"); st.session_state.available_episodes_in_summary = []
            else: st.session_state.available_episodes_in_summary = []
            st.rerun() # Rerun to reflect changes

    # --- Filtering Options ---
    st.sidebar.subheader("Filter Episodes Before Loading")
    if st.session_state.summary_df is not None and not st.session_state.summary_df.empty and st.session_state.available_episodes_in_summary:
        df_summary = st.session_state.summary_df
        available_episodes_in_summary_nums = st.session_state.available_episodes_in_summary
        load_all = st.sidebar.checkbox(f"Load all {len(available_episodes_in_summary_nums)} summarized episodes", value=False, key="load_all_check")
        filtered_summary_df = df_summary[df_summary['episode'].isin(available_episodes_in_summary_nums)].copy()

        if not load_all:
            st.sidebar.write("Apply filters:")
            min_ep, max_ep = min(available_episodes_in_summary_nums), max(available_episodes_in_summary_nums)
            if min_ep <= max_ep: # Episode range slider
                 selected_ep_range = st.sidebar.select_slider("Episode range:", options=list(range(int(min_ep), int(max_ep) + 1)), value=(int(min_ep), int(max_ep)), key="filter_ep_range")
                 filtered_summary_df = filtered_summary_df[(filtered_summary_df['episode'] >= selected_ep_range[0]) & (filtered_summary_df['episode'] <= selected_ep_range[1])]
            # Variable value filter (simplified)
            numeric_cols = filtered_summary_df.select_dtypes(include=np.number).columns.tolist()
            essential = ['total_reward', 'episode_time', 'final_epsilon', 'final_learning_rate']
            options = sorted([c for c in numeric_cols if c == 'episode' or c in essential or not c.endswith(('_std', '_min', '_max'))], reverse=True)
            if 'episode' in options: options.remove('episode')
            if options:
                default_var = 'total_reward' if 'total_reward' in options else options[0]
                idx = options.index(default_var) if default_var in options else 0
                filter_var = st.sidebar.selectbox("Filter by summary variable:", options, index=idx, key="filter_var_select")
                if filter_var and filter_var in filtered_summary_df.columns and not filtered_summary_df[filter_var].isnull().all():
                     try: # Slider for variable range
                         min_v, max_v = float(filtered_summary_df[filter_var].min()), float(filtered_summary_df[filter_var].max())
                         if pd.notna(min_v) and pd.notna(max_v) and min_v < max_v:
                             step_v = max((max_v - min_v) / 100, 1e-6); fmt = "%.3g"
                             sel_range = st.sidebar.slider(f"Range for {filter_var}:", min_v, max_v, (min_v, max_v), step=step_v, format=fmt, key=f"val_range_{filter_var}")
                             filtered_summary_df = filtered_summary_df[(filtered_summary_df[filter_var] >= sel_range[0]) & (filtered_summary_df[filter_var] <= sel_range[1])]
                         elif min_v == max_v: st.sidebar.text(f"{filter_var}: {min_v:.4g}")
                     except Exception as e: st.sidebar.error(f"Slider err for '{filter_var}': {e}")
            # Termination reason filter
            if 'termination_reason' in filtered_summary_df.columns:
                 reasons = sorted(filtered_summary_df['termination_reason'].dropna().unique().tolist())
                 if reasons:
                     sel_reasons = st.sidebar.multiselect("Termination reason:", options=reasons, default=reasons, key="filter_term")
                     if set(sel_reasons) != set(reasons): filtered_summary_df = filtered_summary_df[filtered_summary_df['termination_reason'].isin(sel_reasons)]
            st.session_state.selected_episode_numbers_to_load = sorted(filtered_summary_df['episode'].astype(int).tolist()) if not filtered_summary_df.empty else []
        else: st.session_state.selected_episode_numbers_to_load = st.session_state.available_episodes_in_summary

        num_selected = len(st.session_state.selected_episode_numbers_to_load)
        if num_selected > 0:
            st.sidebar.info(f"{num_selected} episodes match criteria.")
            with st.sidebar.expander(f"Show {num_selected} selected episode numbers", expanded=False): st.write(st.session_state.selected_episode_numbers_to_load)
        else: st.sidebar.warning("No episodes match the current filter criteria.")
    elif st.session_state.selected_folder_path: st.sidebar.warning("Summary data unavailable. Cannot filter.")
    st.session_state.selected_episode_numbers_to_load = st.session_state.get('selected_episode_numbers_to_load', [])


    # --- Load Data Button ---
    load_button_disabled = not st.session_state.selected_episode_numbers_to_load
    if st.sidebar.button("Load Selected Episode Data", key="load_data_button", disabled=load_button_disabled, type="primary"):
        if st.session_state.selected_folder_path and st.session_state.selected_episode_numbers_to_load:
            episodes_tuple = tuple(st.session_state.selected_episode_numbers_to_load)
            with st.spinner(f"Loading data for {len(episodes_tuple)} episodes..."):
                # Reset relevant state before loading
                st.session_state.loaded_episode_data = {'simulation_data': []}
                st.session_state.available_loaded_episode_numbers = []
                st.session_state.current_episode_index = 0
                st.session_state.heatmap_needs_update = True
                st.session_state.heatmap_data_cache = None
                st.session_state.heatmap_params_cache = None
                gc.collect()
                # Load data
                loaded_data = load_selected_episodes(st.session_state.selected_folder_path, episodes_tuple)
                st.session_state.loaded_episode_data = loaded_data or {'simulation_data': []}
                # Process loaded data
                if st.session_state.loaded_episode_data.get('simulation_data'):
                     sim_data = st.session_state.loaded_episode_data['simulation_data']
                     valid_nums = []
                     for ep_data in sim_data:
                         ep_num = ep_data.get('episode')
                         try: valid_nums.append(int(ep_num))
                         except (ValueError, TypeError): logging.warning(f"Non-numeric ep num: {ep_num}")
                     st.session_state.available_loaded_episode_numbers = sorted(list(set(valid_nums)))
                     st.sidebar.success(f"Loaded {len(st.session_state.available_loaded_episode_numbers)} episodes.")
                else: st.sidebar.error("Failed to load or parse episode data.")
                st.rerun() # Rerun after loading
    st.sidebar.markdown("---")

# --------------------------- Plot Customization Renderer ---------------------------
def render_mpl_plot_customization(plot_key_base: str, defaults: Dict, plot_type: str = 'line') -> Dict:
    """
    Renders standard Matplotlib plot customization widgets within an expander.
    Args:
        plot_key_base: A unique string prefix for widget keys.
        defaults: A dictionary containing default values for settings.
        plot_type: Type of plot ('line', 'bar', 'heatmap', 'boxplot') to enable/disable relevant options.
    Returns:
        A dictionary containing the current values of the customization settings.
    """
    current_settings = defaults.copy() # Start with defaults

    with st.expander("Plot Settings"):
        cols1 = st.columns(3)
        # --- General ---
        with cols1[0]:
            current_settings['title'] = st.text_input("Title", value=defaults.get('title', ''), key=f"title_{plot_key_base}")
            current_settings['figsize_w'] = st.slider("Fig Width", 4.0, 15.0, defaults.get('figsize_w', 10.0), 0.5, key=f"fsw_{plot_key_base}")
            current_settings['figsize_h'] = st.slider("Fig Height", 3.0, 12.0, defaults.get('figsize_h', 4.0), 0.5, key=f"fsh_{plot_key_base}")
            current_settings['title_fontsize'] = st.slider("Title Fontsize", 8, 24, defaults.get('title_fontsize', 14), key=f"fs_title_{plot_key_base}")

        with cols1[1]:
            current_settings['xlabel'] = st.text_input("X-Label", value=defaults.get('xlabel', ''), key=f"xlabel_{plot_key_base}")
            current_settings['ylabel'] = st.text_input("Y-Label", value=defaults.get('ylabel', ''), key=f"ylabel_{plot_key_base}")
            current_settings['xlabel_fontsize'] = st.slider("X-Label Fontsize", 8, 18, defaults.get('xlabel_fontsize', 12), key=f"fs_xlabel_{plot_key_base}")
            current_settings['ylabel_fontsize'] = st.slider("Y-Label Fontsize", 8, 18, defaults.get('ylabel_fontsize', 12), key=f"fs_ylabel_{plot_key_base}")

        with cols1[2]:
            current_settings['tick_fontsize'] = st.slider("Tick Fontsize", 6, 16, defaults.get('tick_fontsize', 10), key=f"fs_tick_{plot_key_base}")
            current_settings['grid_on'] = st.checkbox("Show Grid", value=defaults.get('grid_on', True), key=f"grid_{plot_key_base}")
            current_settings['num_xticks'] = st.slider("Approx. X-Ticks", 5, 50, defaults.get('num_xticks', 10), key=f"xticks_{plot_key_base}", disabled=(plot_type not in ['line', 'bar']))
            current_settings['show_legend'] = st.checkbox("Show Legend", value=defaults.get('show_legend', True), key=f"legend_{plot_key_base}", disabled=(plot_type not in ['bar'])) # Only for termination plot for now
            current_settings['legend_pos'] = st.selectbox("Legend Pos", ['best', 'upper left', 'upper right', 'lower left', 'lower right', 'center left', 'center right', 'outside'], index=0, key=f"legendpos_{plot_key_base}", disabled=(not current_settings['show_legend'] or plot_type not in ['bar']))

        st.divider()
        cols2 = st.columns(3)

        # --- Line/Marker Specific ---
        line_marker_enabled = plot_type in ['line']
        with cols2[0]:
            st.markdown("**Line Settings**")
            current_settings['line_color'] = st.color_picker("Line Color", value=defaults.get('line_color', "#1f77b4"), key=f"lcolor_{plot_key_base}", disabled=not line_marker_enabled)
            current_settings['line_width'] = st.slider("Line Width", 0.5, 5.0, defaults.get('line_width', 1.5), 0.5, key=f"lw_{plot_key_base}", disabled=not line_marker_enabled)

        with cols2[1]:
            st.markdown("**Marker Settings**")
            current_settings['marker_size'] = st.slider("Marker Size (0=None)", 0, 10, defaults.get('marker_size', 3), key=f"msize_{plot_key_base}", disabled=not line_marker_enabled)
            marker_styles_options = ['.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
            default_mstyle = defaults.get('marker_style', 'o')
            mstyle_idx = marker_styles_options.index(default_mstyle) if default_mstyle in marker_styles_options else 0
            current_settings['marker_style'] = st.selectbox("Marker Style", marker_styles_options, index=mstyle_idx, key=f"mstyle_{plot_key_base}", disabled=(not line_marker_enabled or current_settings['marker_size'] == 0))
            current_settings['marker_color'] = st.color_picker("Marker Color", value=defaults.get('marker_color', "#ff7f0e"), key=f"mcolor_{plot_key_base}", disabled=(not line_marker_enabled or current_settings['marker_size'] == 0))

        # --- Bar/Heatmap/Boxplot Specific ---
        bar_heat_box_enabled = plot_type in ['bar', 'heatmap', 'boxplot']
        with cols2[2]:
            st.markdown("**Color & Style**")
            # Colormap (Bar, Heatmap)
            cmap_enabled = plot_type in ['bar', 'heatmap']
            try: cmaps = plt.colormaps()
            except: cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm', 'tab10', 'tab20'] # Fallback
            default_cmap = defaults.get('cmap', 'tab10' if plot_type == 'bar' else 'viridis')
            cmap_idx = cmaps.index(default_cmap) if default_cmap in cmaps else 0
            current_settings['cmap'] = st.selectbox("Colormap", cmaps, index=cmap_idx, key=f"cmap_{plot_key_base}", disabled=not cmap_enabled)

            # Bar Width (Bar)
            bar_width_enabled = plot_type == 'bar'
            current_settings['bar_width'] = st.slider("Bar Width", 0.1, 1.0, defaults.get('bar_width', 0.8), 0.1, key=f"width_{plot_key_base}", disabled=not bar_width_enabled)

            # Heatmap Bins & Log Scale
            heatmap_enabled = plot_type == 'heatmap'
            current_settings['heatmap_bins'] = st.slider("Heatmap Bins", 20, 300, defaults.get('heatmap_bins', 100), key=f"heatmap_bins_{plot_key_base}", disabled=not heatmap_enabled)
            current_settings['heatmap_log_scale'] = st.checkbox("Log Color Scale", value=defaults.get('heatmap_log_scale', False), key=f"heatmap_log_{plot_key_base}", disabled=not heatmap_enabled)

            # Boxplot Outliers & Layout
            boxplot_enabled = plot_type == 'boxplot'
            current_settings['boxplot_showfliers'] = st.checkbox("Show Outliers", value=defaults.get('boxplot_showfliers', True), key=f"fliers_{plot_key_base}", disabled=not boxplot_enabled)
            current_settings['boxplot_vert'] = st.checkbox("Vertical Layout", value=defaults.get('boxplot_vert', True), key=f"vert_{plot_key_base}", disabled=not boxplot_enabled)
            current_settings['boxplot_sharey'] = st.checkbox("Share Y-Axis (Vertical)", value=defaults.get('boxplot_sharey', True), key=f"sharey_{plot_key_base}", disabled=(not boxplot_enabled or not current_settings['boxplot_vert']))

    # Combine figsize
    current_settings['figsize'] = (current_settings['figsize_w'], current_settings['figsize_h'])
    # Use empty marker style if size is 0
    if current_settings.get('marker_size', 0) == 0:
        current_settings['marker_style'] = ''

    return current_settings

# --------------------------- Helper to Download Plot ---------------------------
def download_plot_button(fig: plt.Figure, filename_base: str, key_suffix: str):
    """Renders a download button for the given Matplotlib figure."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="Download Plot as PNG",
            data=buf,
            file_name=f"{filename_base}.png",
            mime="image/png",
            key=f"download_{key_suffix}"
        )
    except Exception as e:
        st.error(f"Failed to prepare plot for download: {e}")
    finally:
        buf.close() # Close the buffer

# --------------------------- Introduction / Config Tab ---------------------------
def display_simulation_config():
    # ... (keep existing implementation) ...
    st.subheader("Simulation Configuration")
    config_data = st.session_state.get('config', {})
    if config_data:
        try: st.json(config_data, expanded=False)
        except Exception as e: st.error(f"Error displaying config as JSON: {e}")
    else: st.warning("Config unavailable.")

def introduction_page():
    # ... (keep existing implementation) ...
    col1, col2= st.columns([1, 2])
    with col1:
        logo_path = 'logo_final.png'
        if os.path.exists(logo_path): st.image(logo_path, use_column_width='auto')
        else: logging.warning("Logo image not found"); st.caption("Logo image not found")
        st.markdown("<h3>Simulation Results Dashboard</h3>", unsafe_allow_html=True)
        st.divider(); st.subheader("Contact:")
        st.write("Davor Matías Samuel Ibarra Pérez\nPh.D (s) - UPV / USACH\n📧 dibasam@doctor.upv.es / davor.ibarra@usach.cl")
        st.write("Ph.D Javier Sanchis\nUniversitat Politècnica de València\n📧 jsanchis@isa.upv.es"); st.divider()
    with col2:
        if st.session_state.selected_folder_path:
             display_simulation_config(); st.divider()
             if st.session_state.summary_df is not None:
                  st.subheader("Overall Summary Stats"); st.caption("(`summary.xlsx`)")
                  try: st.dataframe(st.session_state.summary_df.describe().style.format(precision=4))
                  except Exception as e: st.error(f"Err summary stats: {e}")
             else: st.warning("Summary data not loaded.")
        else: st.info("Select a folder.")

# --------------------------- Plotting Functions (Matplotlib, Customized, Downloadable) ---------------------------

def plot_termination_reason_distribution_mpl(summary_df: pd.DataFrame):
    """Genera gráfico de barras apiladas de razones de terminación usando Matplotlib."""
    st.write("### Termination Reason Distribution (Aggregated)")
    if summary_df is None or summary_df.empty or 'termination_reason' not in summary_df.columns or 'episode' not in summary_df.columns:
        st.warning("Insufficient summary data for termination plot.")
        return

    df = summary_df.copy()
    reason_counts_df = None
    fig, ax = None, None # Initialize fig, ax

    try:
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        df = df.dropna(subset=['episode'])
        if df.empty: st.warning("No valid numeric episodes found."); return
        df['episode'] = df['episode'].astype(int)
        df['episode_group'] = (df['episode'] // 1000) * 1000
        reason_counts_df = df.groupby(['episode_group', 'termination_reason']).size().unstack(fill_value=0)
        if reason_counts_df.empty: st.warning("No termination reasons found."); return

        # --- Plot Customization ---
        plot_key_base = "term_reason_dist"
        default_settings = {
            'title': "Termination Reason Distribution per 1000 Episodes", 'xlabel': "Episode Group Start", 'ylabel': "Count",
            'figsize_w': 10.0, 'figsize_h': 5.0, 'title_fontsize': 14, 'xlabel_fontsize': 12, 'ylabel_fontsize': 12,
            'tick_fontsize': 10, 'grid_on': True, 'num_xticks': 10, # num_xticks might not be directly used for categorical x-axis
            'cmap': 'tab10', 'bar_width': 0.8, 'show_legend': True, 'legend_pos': 'outside'
        }
        customize = st.checkbox(f"Customize Plot", key=f"customize_{plot_key_base}", value=False)
        current_settings = render_mpl_plot_customization(plot_key_base, default_settings, plot_type='bar') if customize else default_settings

        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=current_settings['figsize'])
        reason_counts_df.plot(kind='bar', stacked=True, ax=ax, colormap=current_settings['cmap'], width=current_settings['bar_width'])

        ax.set_title(current_settings['title'], fontsize=current_settings['title_fontsize'])
        ax.set_xlabel(current_settings['xlabel'], fontsize=current_settings['xlabel_fontsize'])
        ax.set_ylabel(current_settings['ylabel'], fontsize=current_settings['ylabel_fontsize'])
        ax.tick_params(axis='x', rotation=45, labelsize=current_settings['tick_fontsize'])
        ax.tick_params(axis='y', labelsize=current_settings['tick_fontsize'])
        ax.grid(visible=current_settings['grid_on'], linestyle='--', alpha=0.6, axis='y')

        if current_settings['show_legend']:
            if current_settings['legend_pos'] == 'outside':
                ax.legend(title='Reason', bbox_to_anchor=(1.04, 1), loc='upper left', fontsize='small')
                fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
            else:
                ax.legend(title='Reason', loc=current_settings['legend_pos'], fontsize='small')
                fig.tight_layout()
        else:
             if ax.get_legend() is not None: ax.get_legend().remove()
             fig.tight_layout()

        st.pyplot(fig)

        # --- Download Button ---
        download_plot_button(fig, "termination_reason_distribution", plot_key_base)

        with st.expander("Show Raw Counts"): st.dataframe(reason_counts_df)

    except Exception as e:
        st.error(f"Error generating termination reason plot: {e}")
        logging.error(f"Termination plot error: {traceback.format_exc()}")
    finally:
        if fig is not None: plt.close(fig) # Close the figure


def plot_overview_metrics_mpl(data: list):
    """Genera gráficos de rendimiento general usando Matplotlib con personalización y descarga."""
    st.write("### Performance Metrics Across Loaded Episodes")
    if not data: st.warning("No episode data for overview plots."); return

    # --- Data Preparation ---
    episode_metrics_list = []
    for episode in data:
        ep_num = episode.get('episode'); reward_list = episode.get('cumulative_reward'); time_list = episode.get('time')
        if ep_num is not None and isinstance(reward_list, list) and reward_list and isinstance(time_list, list) and time_list:
            try:
                final_reward = float(reward_list[-1]); final_time = float(time_list[-1])
                if not np.isfinite(final_reward) or not np.isfinite(final_time): raise ValueError("Non-finite")
                performance = final_reward / final_time if final_time != 0 else 0
                episode_metrics_list.append({'episode': int(ep_num), 'final_reward': final_reward, 'final_time': final_time, 'performance': performance})
            except (ValueError, TypeError, IndexError) as e: logging.warning(f"Skipping ep {ep_num} overview: {e}")
        else: logging.warning(f"Skipping ep {ep_num} overview: missing data.")

    if not episode_metrics_list: st.warning("No valid episodes for overview plots."); return
    metrics_df = pd.DataFrame(episode_metrics_list).sort_values(by='episode').reset_index(drop=True)
    episode_numbers = metrics_df['episode'].tolist()

    plots_config = [
        {"title": "Cumulative Reward", "y_col": "final_reward", "ylabel": "Cumulative Reward"},
        {"title": "Episode Duration", "y_col": "final_time", "ylabel": "Duration (s)"},
        {"title": "Performance (Reward per Time)", "y_col": "performance", "ylabel": "Performance"},
    ]

    for config in plots_config:
        plot_title = config["title"]; y_col = config["y_col"]; y_data = metrics_df[y_col].tolist(); default_ylabel = config["ylabel"]
        st.write(f"#### {plot_title}")
        plot_key_base = f"overview_{y_col}"
        fig, ax = None, None # Initialize

        try:
            # --- Customization ---
            default_settings = {
                'title': plot_title, 'xlabel': "Episode", 'ylabel': default_ylabel, 'figsize_w': 10.0, 'figsize_h': 4.0,
                'title_fontsize': 14, 'xlabel_fontsize': 12, 'ylabel_fontsize': 12, 'tick_fontsize': 10,
                'line_color': "#1f77b4", 'line_width': 1.5, 'marker_color': "#ff7f0e", 'marker_size': 3, 'marker_style': 'o',
                'grid_on': True, 'num_xticks': 10
            }
            customize = st.checkbox(f"Customize Plot", key=f"customize_{plot_key_base}", value=False)
            current_settings = render_mpl_plot_customization(plot_key_base, default_settings, plot_type='line') if customize else default_settings

            # --- Create Plot ---
            fig, ax = plt.subplots(figsize=current_settings['figsize'])
            ax.plot(episode_numbers, y_data,
                    marker=current_settings['marker_style'], linestyle='-', color=current_settings['line_color'],
                    linewidth=current_settings['line_width'], markersize=current_settings['marker_size'],
                    markerfacecolor=current_settings['marker_color'], markeredgecolor=current_settings['marker_color'])
            ax.set_title(current_settings['title'], fontsize=current_settings['title_fontsize'])
            ax.set_xlabel(current_settings['xlabel'], fontsize=current_settings['xlabel_fontsize'])
            ax.set_ylabel(current_settings['ylabel'], fontsize=current_settings['ylabel_fontsize'])
            ax.grid(visible=current_settings['grid_on'], linestyle='--', alpha=0.6)

            # Dynamic Ticks
            if len(episode_numbers) > 1:
                 min_ep, max_ep = min(episode_numbers), max(episode_numbers); tick_range = max_ep - min_ep
                 if tick_range > 0:
                     num_bins_calc = max(1, current_settings['num_xticks'])
                     tick_step = max(1, int(np.ceil(tick_range / num_bins_calc)))
                     if (tick_range / tick_step) > 50: tick_step = max(1, int(np.ceil(tick_range / 15)))
                     ticks = np.arange(min_ep, max_ep + tick_step, tick_step, dtype=int)
                     if len(ticks) > 0: ax.set_xticks(ticks); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                     else: ax.set_xticks(episode_numbers) # Fallback
                 else: ticks = [min_ep]; ax.set_xticks(ticks)
            elif len(episode_numbers) == 1: ax.set_xticks(episode_numbers)
            ax.tick_params(axis='both', which='major', labelsize=current_settings['tick_fontsize'])

            fig.tight_layout()
            st.pyplot(fig)

            # --- Download Button ---
            download_plot_button(fig, f"overview_{y_col}", plot_key_base)

        except Exception as e:
            st.error(f"Error generating plot '{plot_title}': {e}")
            logging.error(f"Overview plot error: {traceback.format_exc()}")
        finally:
            if fig is not None: plt.close(fig)

        st.divider() # Divider between overview metric plots


# Prepare heatmap data - adopted from app_deprecate (no changes needed here)
def prepare_heatmap_data(data: list) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[List[str]]]:
    # ... (keep existing implementation from previous good version) ...
    if not data: return None, None
    all_metrics = {}; available_params_set = set(); valid_episode_found = False; max_len_overall = 0
    logging.info(f"Preparing heatmap data from {len(data)} episodes...")
    for ep_idx, episode in enumerate(data):
        is_ep_valid = False; temp_metrics_ep = {}; max_len_ep = 0
        ep_num_for_log = episode.get('episode', f'Index {ep_idx}')
        items_to_process = list(episode.items())
        for key, values in items_to_process:
            if isinstance(values, list) and values:
                try:
                    first_val = values[0]
                    if isinstance(first_val, (int, float, np.number)):
                        numeric_values = pd.to_numeric(values, errors='coerce')
                        if not np.isnan(numeric_values).all():
                            temp_metrics_ep[key] = numeric_values
                            available_params_set.add(key)
                            max_len_ep = max(max_len_ep, len(numeric_values))
                            is_ep_valid = True
                except (TypeError, IndexError): pass
                except Exception as conv_e: logging.warning(f"Error converting {key} in ep {ep_num_for_log}: {conv_e}")
        if not is_ep_valid: continue
        valid_episode_found = True; max_len_overall = max(max_len_overall, max_len_ep)
        for key in available_params_set:
            if key not in all_metrics: all_metrics[key] = []
            values_to_add = temp_metrics_ep.get(key)
            if values_to_add is not None: all_metrics[key].append(values_to_add)
            else: all_metrics[key].append(np.full(max_len_ep, np.nan))
    if not valid_episode_found: logging.warning("No valid episodes for heatmap prep."); return None, None
    final_params_list = sorted(list(available_params_set)); concatenated_metrics = {}; final_params_available = []
    for key in final_params_list:
        list_of_arrays = all_metrics.get(key, [])
        if list_of_arrays:
            try:
                concatenated_array = np.concatenate(list_of_arrays)
                if not np.isnan(concatenated_array).all():
                    concatenated_metrics[key] = concatenated_array
                    final_params_available.append(key)
                else: logging.warning(f"'{key}' is all NaN. Excluding.")
            except ValueError as concat_ve: logging.error(f"Shape mismatch '{key}': {concat_ve}.")
            except Exception as concat_e: logging.error(f"Concat error '{key}': {concat_e}")
        else: logging.warning(f"No data for '{key}'.")
    if len(final_params_available) < 2: logging.warning("Need >= 2 params for heatmap."); return None, None
    logging.info(f"Heatmap data OK. Params: {final_params_available}")
    return concatenated_metrics, final_params_available

# Plot heatmap - adopted from app_deprecate, add customization/download
def plot_heatmap(data: list, needs_update: bool):
    """Genera un heatmap interactivo usando Matplotlib, con personalización y descarga."""
    st.write("### Heatmaps of Trajectories (Aggregated)")
    st.caption("Density plot showing frequency of state pairs across all loaded episodes.")

    # --- Data Preparation ---
    if needs_update or st.session_state.heatmap_data_cache is None:
        with st.spinner("Aggregating data for heatmaps..."):
            st.session_state.heatmap_data_cache, st.session_state.heatmap_params_cache = prepare_heatmap_data(data)
        if st.session_state.heatmap_data_cache is None: st.warning("Could not prepare data for heatmaps."); return

    heatmap_data = st.session_state.heatmap_data_cache
    available_params = st.session_state.heatmap_params_cache
    if not heatmap_data or not available_params or len(available_params) < 2: st.warning("Insufficient data/params for heatmap."); return

    # --- Parameter Selection ---
    col1, col2 = st.columns(2)
    with col1:
        x_idx = available_params.index('time') if 'time' in available_params else 0
        x_param = st.selectbox("X-axis parameter", available_params, index=x_idx, key="heatmap_x")
    with col2:
        default_y = 'pendulum_angle'; y_idx = 0
        if default_y in available_params and default_y != x_param: y_idx = available_params.index(default_y)
        elif len(available_params) > 1: y_idx = next((i for i, p in enumerate(available_params) if p != x_param), 1 if len(available_params)>1 else 0)
        y_param = st.selectbox("Y-axis parameter", available_params, index=y_idx, key="heatmap_y")

    if x_param == y_param: st.warning("X and Y parameters must be different."); return

    # --- Customization ---
    plot_key_base = f"heatmap_{x_param}_{y_param}"
    default_title = f"Heatmap: {y_param.replace('_',' ').capitalize()} vs {x_param.replace('_',' ').capitalize()} (Frequency)"
    default_settings = {
        'title': default_title, 'xlabel': x_param.replace('_', ' ').capitalize(), 'ylabel': y_param.replace('_', ' ').capitalize(),
        'figsize_w': 10.0, 'figsize_h': 6.0, 'title_fontsize': 14, 'xlabel_fontsize': 12, 'ylabel_fontsize': 12,
        'tick_fontsize': 10, 'grid_on': False, # Grid usually off for heatmaps
        'cmap': 'viridis', 'heatmap_bins': 100, 'heatmap_log_scale': False
        # Add ranges if needed, but hist2d handles auto range well
    }
    customize = st.checkbox("Customize Heatmap", key=f"customize_{plot_key_base}", value = False)
    # Get settings (either default or from renderer)
    current_settings = render_mpl_plot_customization(plot_key_base, default_settings, plot_type='heatmap') if customize else default_settings

    # --- Generate Button and Plotting ---
    if st.button("Generate Heatmap", key="gen_heatmap_btn"):
        x_data = heatmap_data.get(x_param); y_data = heatmap_data.get(y_param)
        if x_data is None or y_data is None: st.error(f"Data missing for '{x_param}' or '{y_param}'."); return

        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_plot, y_plot = x_data[mask], y_data[mask]
        if len(x_plot) == 0: st.warning("No valid finite data points."); return

        fig, ax = None, None # Initialize
        with st.spinner("Generating heatmap..."):
            try:
                fig, ax = plt.subplots(figsize=current_settings['figsize'])
                norm = mcolors.LogNorm() if current_settings['heatmap_log_scale'] else None
                # Note: vmin/vmax can be added if range customization is added to render_mpl_plot_customization
                counts, xedges, yedges, img = ax.hist2d(x_plot, y_plot, bins=current_settings['heatmap_bins'], cmap=current_settings['cmap'], norm=norm)

                cbar = fig.colorbar(img, ax=ax)
                cbar_label = 'Frequency' + (' (Log Scale)' if current_settings['heatmap_log_scale'] else '')
                cbar.set_label(cbar_label)
                ax.set_title(current_settings['title'], fontsize=current_settings['title_fontsize'])
                ax.set_xlabel(current_settings['xlabel'], fontsize=current_settings['xlabel_fontsize'])
                ax.set_ylabel(current_settings['ylabel'], fontsize=current_settings['ylabel_fontsize'])
                ax.tick_params(axis='both', labelsize=current_settings['tick_fontsize'])
                # Grid is usually not helpful for heatmap, controlled by grid_on setting if needed
                ax.grid(visible=current_settings['grid_on'], linestyle='--', alpha=0.6)

                fig.tight_layout()
                st.pyplot(fig)

                # --- Download Button ---
                download_plot_button(fig, f"heatmap_{x_param}_vs_{y_param}", plot_key_base)

            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
                logging.error(f"Heatmap generation failed: {traceback.format_exc()}")
            finally:
                 if fig is not None: plt.close(fig)


# Q-Table Comparison - Adopted from app_deprecate (No plotting, no download needed)
def display_qtable_comparison(simulation_data, config):
    # ... (keep existing implementation - it displays DataFrames, no plots) ...
    st.write("### Q-Table Evolution")
    episodes_with_qtables_info = {}
    qtable1_df, qtable2_df = None, None
    for i, ep in enumerate(simulation_data):
        ep_num = ep.get('episode'); qtables_data = ep.get('qtables')
        if ep_num is not None and isinstance(qtables_data, dict) and qtables_data:
            try: episodes_with_qtables_info[int(ep_num)] = i
            except (ValueError, TypeError): logging.warning(f"Non-int ep num {ep_num} with Q-tables, skipping.")
    if not episodes_with_qtables_info: st.info("No Q-table data in loaded episodes."); return
    available_ep_numbers = sorted(episodes_with_qtables_info.keys())
    col1, col2 = st.columns(2)
    with col1: ep1_num = st.selectbox("Select first episode:", available_ep_numbers, index=0, key="q_ep1")
    with col2:
        valid_ep2_options = [ep for ep in available_ep_numbers if ep > ep1_num]
        if not valid_ep2_options: st.warning("No Q-table episodes after first selection."); return
        ep2_num = st.selectbox("Select second episode:", valid_ep2_options, index=0, key="q_ep2")
    idx1 = episodes_with_qtables_info[ep1_num]; idx2 = episodes_with_qtables_info[ep2_num]
    try: gain_types = list(simulation_data[idx1].get('qtables', {}).keys())
    except Exception as e: st.error(f"Err accessing Q-keys ep {ep1_num}: {e}"); return
    if not gain_types: st.error(f"Q-table dict empty/invalid ep {ep1_num}."); return
    selected_gain = st.selectbox("Select gain type (K_p, K_i, K_d):", gain_types, key="q_gain")
    try:
        qtable1_raw = simulation_data[idx1]['qtables'][selected_gain]; qtable2_raw = simulation_data[idx2]['qtables'][selected_gain]
        qtable1_df = pd.DataFrame(qtable1_raw); qtable2_df = pd.DataFrame(qtable2_raw)
    except KeyError: st.error(f"Gain '{selected_gain}' not found ep {ep1_num} or {ep2_num}."); return
    except ValueError as ve: st.error(f"Err creating DF '{selected_gain}': {ve}"); st.write("Raw Q1:", qtable1_raw); return
    except Exception as e: st.error(f"Err loading/converting Q '{selected_gain}': {e}"); return
    # Interpret State Config for Indexing (Keep existing logic)
    state_names = []; state_bins_info_str = []; index_levels = []; num_state_vars = 0; expected_rows = 1; state_config_valid = False; config_error_msg = ""
    try:
        state_cfg = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})
        if not state_cfg: config_error_msg = "State config not found."
        else:
            state_config_valid = True; ordered_state_vars = ['angle', 'angular_velocity', selected_gain]
            for var in ordered_state_vars:
                var_config = state_cfg.get(var)
                if isinstance(var_config, dict) and var_config.get('enabled', False):
                    bins = var_config.get('bins')
                    if bins and isinstance(bins, int) and bins > 0: state_names.append(var); state_bins_info_str.append(f"{var}({bins})"); index_levels.append(range(bins)); expected_rows *= bins; num_state_vars += 1
                    else: config_error_msg = f"Config '{var}' invalid."; state_config_valid = False; break
            if not state_names and state_config_valid: config_error_msg = "No enabled state vars."; state_config_valid = False
    except Exception as cfg_e: config_error_msg = f"Error parsing state config: {cfg_e}"; state_config_valid = False
    num_actions = qtable1_df.shape[1]; action_names = [f"Action_{i}" for i in range(num_actions)]
    if num_actions == 3: action_names = ['Decrease', 'Keep', 'Increase']
    qtable1_df.columns = action_names; qtable2_df.columns = action_names
    index_applied = False; actual_rows = qtable1_df.shape[0]
    if state_config_valid and expected_rows == actual_rows and num_state_vars > 0:
        try:
            if num_state_vars > 1: multi_index = pd.MultiIndex.from_product(index_levels, names=state_names)
            else: multi_index = pd.Index(index_levels[0], name=state_names[0])
            qtable1_df.index = multi_index; qtable2_df.index = multi_index; index_applied = True
            st.caption(f"State Index applied: {', '.join(state_bins_info_str)}")
        except Exception as idx_e: st.warning(f"Err applying index names: {idx_e}.")
    elif state_config_valid and expected_rows != actual_rows: st.error(f"Q Dim mismatch '{selected_gain}'! Expected: {expected_rows}, Actual: {actual_rows}")
    elif config_error_msg: st.warning(f"Cannot determine state index: {config_error_msg}.")
    else: st.warning("State config invalid. Using default index.")
    # Display DataFrames
    st.write(f"Comparing Q-Table for gain: **{selected_gain.upper()}**")
    col11, col22 = st.columns(2)
    with col11:
        st.write(f"**Episode {ep1_num}:**")
        try: st.dataframe(qtable1_df.style.apply(resaltar_maximo, axis=1).format(precision=4))
        except Exception as df_disp_e: st.error(f"Err display Q1: {df_disp_e}"); st.dataframe(qtable1_df)
    with col22:
        st.write(f"**Episode {ep2_num}:**")
        try: st.dataframe(qtable2_df.style.apply(resaltar_maximo, axis=1).format(precision=4))
        except Exception as df_disp_e: st.error(f"Err display Q2: {df_disp_e}"); st.dataframe(qtable2_df)
    st.info("Green cells highlight the action with the highest Q-value for that state.")
    if num_actions == 3: st.caption("Assuming Actions are: 'Decrease' Gain, 'Keep' Gain, 'Increase' Gain.")
    # No download button needed as these are DataFrames, not plots.


# --- Boxplots (Matplotlib Implementation) ---
@st.cache_data
def prepare_boxplot_data_mpl(episode_data_tuple: Tuple[Dict, ...], vars_to_plot: Tuple[str, ...]) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    """Prepares data for Matplotlib boxplot, returns dict and labels."""
    # ... (keep existing implementation) ...
    boxplot_data_dict = {var: [] for var in vars_to_plot}; episode_labels = []
    logging.info(f"Prep boxplot data {len(vars_to_plot)} vars, {len(episode_data_tuple)} eps...")
    episode_data_list = list(episode_data_tuple)
    try: episode_data_list.sort(key=lambda ep: int(ep.get('episode', -1)))
    except (ValueError, TypeError): logging.warning("Cannot sort eps numerically.")
    for episode in episode_data_list:
        ep_num = episode.get('episode', 'N/A'); ep_label = str(ep_num); episode_has_data = False
        for var in vars_to_plot:
            values = episode.get(var)
            if isinstance(values, list):
                numeric_vals = pd.to_numeric(values, errors='coerce'); valid_vals = numeric_vals[np.isfinite(numeric_vals)]
                if len(valid_vals) > 0: boxplot_data_dict[var].append(valid_vals); episode_has_data = True
                else: boxplot_data_dict[var].append(np.array([]))
            else: boxplot_data_dict[var].append(np.array([]))
        if episode_has_data: episode_labels.append(ep_label)
    final_boxplot_data = {}; final_vars = []
    for var, data_arrays in boxplot_data_dict.items():
        if any(len(arr) > 0 for arr in data_arrays):
             final_boxplot_data[var] = [data_arrays[i] for i, label in enumerate(episode_labels) if i < len(data_arrays)] # Ensure index bounds
             final_vars.append(var)
    if not final_vars: logging.warning("No valid data points for boxplot prep."); return {}, []
    logging.info(f"Finished boxplot data prep. {len(final_vars)} vars, {len(episode_labels)} eps.")
    return final_boxplot_data, episode_labels


def plot_episode_boxplots_mpl():
    """Genera boxplots para variables específicas usando Matplotlib con personalización y descarga."""
    st.subheader("Variable Distribution Across Loaded Episodes (Box Plots)")

    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    data = loaded_data.get('simulation_data', [])
    if not data: st.warning("No episode data loaded."); return

    # --- Variable Selection ---
    numeric_vars_options = []
    if data:
        first_ep = data[0]
        for key, value in first_ep.items():
            if isinstance(value, list) and value and key != 'qtables':
                 try:
                    if isinstance(value[0], (int, float, np.number)): numeric_vars_options.append(key)
                 except IndexError: pass
        vars_to_exclude = {'time', 'episode', 'cumulative_reward', 'epsilon', 'learning_rate', 'action_kp', 'action_ki', 'action_kd'}
        numeric_vars_options = sorted([v for v in list(set(numeric_vars_options)) if v not in vars_to_exclude])
    if not numeric_vars_options: st.warning("No suitable numeric variables for boxplots."); return
    default_selections = [v for v in ['kp', 'ki', 'kd', 'pendulum_angle', 'pendulum_velocity', 'force', 'reward', 'error'] if v in numeric_vars_options]
    selected_variables = st.multiselect("Select variables for Box Plot:", numeric_vars_options, default=default_selections, key="boxplot_vars_mpl")
    if not selected_variables: st.info("Select one or more variables."); return

    # --- Data Preparation ---
    boxplot_data_dict, episode_labels = {}, []
    try:
        data_tuple = tuple(data); vars_tuple = tuple(selected_variables)
        boxplot_data_dict, episode_labels = prepare_boxplot_data_mpl(data_tuple, vars_tuple)
    except TypeError as e:
         st.error(f"Cannot cache boxplot data: {e}")
         boxplot_data_dict, episode_labels = prepare_boxplot_data_mpl.__wrapped__(data, selected_variables)
    if not boxplot_data_dict or not episode_labels: st.warning(f"No valid numeric data found."); return

    # --- Plot Customization ---
    plot_key_base = "boxplots_mpl"
    default_settings = {
        'title': "Distribution per Episode for Selected Variables", 'xlabel': "Episode Number", 'ylabel': "Value",
        'figsize_w': 10.0, 'figsize_h': max(6.0, 3.0 * len(selected_variables)), # Dynamic height default
        'title_fontsize': 16, 'xlabel_fontsize': 12, 'ylabel_fontsize': 12, 'tick_fontsize': 8,
        'grid_on': True, 'boxplot_showfliers': True, 'boxplot_vert': True, 'boxplot_sharey': True
    }
    customize = st.checkbox(f"Customize Box Plots", key=f"customize_{plot_key_base}", value=False)
    current_settings = render_mpl_plot_customization(plot_key_base, default_settings, plot_type='boxplot') if customize else default_settings

    # --- Create Plot ---
    st.write("Generating boxplots...")
    num_vars = len(selected_variables)
    vert_layout = current_settings['boxplot_vert']
    share_y = current_settings['boxplot_sharey'] if vert_layout else False
    share_x = share_y if not vert_layout else False # Share X if horizontal and Y shared

    nrows = num_vars if vert_layout else 1
    ncols = 1 if vert_layout else num_vars
    fig, axes = None, None # Initialize

    try:
        fig, axes = plt.subplots(nrows, ncols, figsize=current_settings['figsize'], sharey=share_y, sharex=share_x, squeeze=False)
        axes_flat = axes.flatten()

        for i, var in enumerate(selected_variables):
            ax = axes_flat[i]; data_for_var = boxplot_data_dict.get(var, [])
            plot_data = [d for d in data_for_var if len(d) > 0]
            plot_labels = [episode_labels[j] for j, d in enumerate(data_for_var) if len(d) > 0]

            if not plot_data:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(var.replace('_', ' ').capitalize())
                continue

            bp = ax.boxplot(plot_data, labels=plot_labels, showfliers=current_settings['boxplot_showfliers'], vert=vert_layout, patch_artist=True)

            ax.set_title(var.replace('_', ' ').capitalize())
            ax.grid(visible=current_settings['grid_on'], linestyle='--', alpha=0.6)

            if vert_layout:
                ax.tick_params(axis='x', rotation=45, labelsize=current_settings['tick_fontsize'])
                ax.tick_params(axis='y', labelsize=current_settings['tick_fontsize'])
                if i == num_vars - 1 or not share_y: ax.set_xlabel(current_settings['xlabel'], fontsize=current_settings['xlabel_fontsize'])
                if i == 0 or not share_y: ax.set_ylabel(current_settings['ylabel'], fontsize=current_settings['ylabel_fontsize']) # Add Y label more consistently
            else: # Horizontal layout
                ax.tick_params(axis='y', rotation=0, labelsize=current_settings['tick_fontsize'])
                ax.tick_params(axis='x', labelsize=current_settings['tick_fontsize'])
                if i == 0 or not share_x: ax.set_ylabel(current_settings['xlabel'], fontsize=current_settings['xlabel_fontsize']) # Use xlabel setting for y label here
                if i == 0 or not share_x: ax.set_xlabel(current_settings['ylabel'], fontsize=current_settings['ylabel_fontsize']) # Use ylabel setting for x label

        fig.suptitle(current_settings['title'], fontsize=current_settings['title_fontsize'])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

        # --- Download Button ---
        download_plot_button(fig, "episode_boxplots", plot_key_base)

    except Exception as e:
        st.error(f"Error generating boxplots: {e}")
        logging.error(f"Boxplot generation failed: {traceback.format_exc()}")
    finally:
        if fig is not None: plt.close(fig)


# --- Episode Details Tab ---
def episode_details():
    """Muestra detalles, gráficos Matplotlib y animación de un episodio específico."""
    st.subheader("Detailed Episode Analysis")
    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    loaded_sim_data = loaded_data.get('simulation_data', [])
    available_loaded_episode_numbers = st.session_state.get('available_loaded_episode_numbers', [])
    if not loaded_sim_data or not available_loaded_episode_numbers: st.warning("No episode data loaded."); return

    # --- Episode Navigation (Keep existing robust logic) ---
    st.sidebar.subheader("Select Episode for Details")
    if not available_loaded_episode_numbers: st.sidebar.warning("No loaded eps."); return
    min_ep, max_ep = min(available_loaded_episode_numbers), max(available_loaded_episode_numbers)
    if st.session_state.current_episode_index >= len(available_loaded_episode_numbers): st.session_state.current_episode_index = 0
    current_ep_num_in_state = available_loaded_episode_numbers[st.session_state.current_episode_index]
    target_episode_num = st.sidebar.number_input(f"Episode ({min_ep}-{max_ep}):", min_ep, max_ep, current_ep_num_in_state, 1, key="detail_episode_input")
    new_selected_index = st.session_state.current_episode_index
    try: # Logic to find exact or closest episode and update state/rerun
        if target_episode_num != current_ep_num_in_state:
            if target_episode_num in available_loaded_episode_numbers: new_selected_index = available_loaded_episode_numbers.index(target_episode_num)
            else: closest_episode_num = min(available_loaded_episode_numbers, key=lambda x: abs(x - target_episode_num)); new_selected_index = available_loaded_episode_numbers.index(closest_episode_num); st.sidebar.info(f"Showing closest: {closest_episode_num}"); st.session_state.detail_episode_input = closest_episode_num
            if new_selected_index != st.session_state.current_episode_index: st.session_state.current_episode_index = new_selected_index; st.rerun()
    except Exception as nav_e: st.sidebar.error(f"Error finding episode: {nav_e}"); return
    col_prev, col_next = st.sidebar.columns(2)
    prev_disabled = st.session_state.current_episode_index == 0
    next_disabled = st.session_state.current_episode_index >= len(available_loaded_episode_numbers) - 1
    if col_prev.button("⬅️ Prev Ep", key="prev_ep_button", use_container_width=True, disabled=prev_disabled): st.session_state.current_episode_index -= 1; st.session_state.detail_episode_input = available_loaded_episode_numbers[st.session_state.current_episode_index]; st.rerun()
    if col_next.button("Next Ep ➡️", key="next_ep_button", use_container_width=True, disabled=next_disabled): st.session_state.current_episode_index += 1; st.session_state.detail_episode_input = available_loaded_episode_numbers[st.session_state.current_episode_index]; st.rerun()

    # --- Display Selected Episode Details ---
    try:
        selected_episode_data = loaded_sim_data[st.session_state.current_episode_index]
        actual_episode_number = selected_episode_data.get('episode', 'N/A')
        term_reason = selected_episode_data.get('termination_reason', 'N/A')
        time_list = selected_episode_data.get('time', []); reward_list = selected_episode_data.get('cumulative_reward', [])
        final_time = float(time_list[-1]) if time_list else np.nan; final_reward = float(reward_list[-1]) if reward_list else np.nan
        st.write(f"#### Details for Episode: {actual_episode_number}")
        st.info(f"Termination: **{term_reason}** | Final Time: **{final_time:.3f}s** | Final Reward: **{final_reward:.3f}**")

        col1, col2 = st.columns(2)
        df_episode = None
        with col1: # Data Table (Keep existing logic)
            st.write("**Episode Data Table**"); # ... (Existing DataFrame creation logic) ...
            try:
                df_data = {}; max_len = 0
                for k, v in selected_episode_data.items():
                    if isinstance(v, list) and k != 'qtables': max_len = max(max_len, len(v))
                if max_len > 0:
                    for k, v in selected_episode_data.items():
                        if k == 'qtables': continue
                        if isinstance(v, list):
                            current_len = len(v)
                            if current_len == max_len: df_data[k] = v
                            elif current_len < max_len: df_data[k] = list(v) + [np.nan] * (max_len - current_len)
                            else: df_data[k] = v[:max_len]
                        elif isinstance(v, (int, float, str, bool, np.number)): df_data[k] = [v] * max_len
                    if df_data:
                        df_episode = pd.DataFrame(df_data)
                        display_cols = ['time', 'pendulum_angle', 'pendulum_velocity', 'cart_position', 'cart_velocity', 'force', 'reward', 'cumulative_reward', 'kp', 'ki', 'kd', 'error', 'epsilon', 'learning_rate', 'action_kp', 'action_ki', 'action_kd', 'termination_reason']
                        existing_cols = [col for col in display_cols if col in df_episode.columns]
                        st.dataframe(df_episode[existing_cols], height=300)
                    else: st.warning("No suitable data for episode table.")
                else: st.warning("No list data found in episode.")
            except Exception as e: st.error(f"Error preparing table: {e}")

        with col2: # Statistics (Keep existing logic)
            st.write("**Summary Statistics**")
            if df_episode is not None and not df_episode.empty:
                 try:
                     numeric_cols_df = df_episode.select_dtypes(include=np.number)
                     if not numeric_cols_df.empty: st.dataframe(numeric_cols_df.describe().T.style.format(precision=4))
                     else: st.warning("No numeric data for stats.")
                 except Exception as e: st.error(f"Error calculating stats: {e}")
            else: st.warning("Data table unavailable.")

        # Button to download the table data (optional, kept from previous version if useful)
        # if st.button(f"Save Full Data Table Ep {actual_episode_number}", key="save_episode_table"):
        #     if df_episode is not None: save_plot_data(df_episode, f"episode_{actual_episode_number}_full_data")
        #     else: st.warning("Table not generated.")

        st.divider()
        st.write("**Interactive Plots (Matplotlib)**")
        plot_episode_details_graphs_mpl(selected_episode_data, actual_episode_number) # Pass episode number for keys

        st.divider()
        st.write("**Animation**")
        plot_episode_animation(selected_episode_data) # Keep existing animation function

    except IndexError: st.error(f"Error accessing episode index {st.session_state.current_episode_index}.")
    except Exception as detail_e: st.error(f"Unexpected error displaying details: {detail_e}"); logging.error(f"Detail display error: {traceback.format_exc()}")


def plot_episode_details_graphs_mpl(episode: dict, episode_number: Union[int, str]):
    """Genera gráficos interactivos para un episodio usando Matplotlib con personalización y descarga."""
    plottable_vars = []
    ep_num = episode_number # Use passed episode number
    if episode:
         for k, v in episode.items(): # Find plottable vars
              if isinstance(v, list) and v and k != 'qtables':
                  try:
                      if isinstance(v[0], (int, float, np.number)): plottable_vars.append(k)
                  except IndexError: pass
         plottable_vars = sorted(plottable_vars)
    if not plottable_vars: st.warning("No plottable data in episode."); return

    # Session state for selections, keyed by episode
    state_key_base = f'plot_selections_ep_{ep_num}'
    if state_key_base not in st.session_state: st.session_state[state_key_base] = {'x_param': 'time', 'y_params': []}
    plot_selections = st.session_state[state_key_base]

    col1, col2 = st.columns([1, 2])
    with col1: # X-axis selection
        x_idx = plottable_vars.index('time') if 'time' in plottable_vars else 0
        plot_selections['x_param'] = st.selectbox("X-axis:", plottable_vars, index=x_idx, key=f'detail_plot_x_{ep_num}')
    with col2: # Y-axis selection
        y_options = [v for v in plottable_vars if v != plot_selections['x_param']]
        current_y = plot_selections.get('y_params', [])
        valid_defaults = [p for p in current_y if p in y_options]
        if not valid_defaults and y_options: valid_defaults = [p for p in ['pendulum_angle', 'cart_position', 'force', 'reward'] if p in y_options][:2] # Default to max 2
        plot_selections['y_params'] = st.multiselect("Y-axis variables:", y_options, default=valid_defaults, key=f'detail_plot_y_{ep_num}')

    x_param = plot_selections['x_param']; y_params = plot_selections['y_params']
    if not x_param or not y_params: st.info("Select X and Y axis variables."); return

    x_data_raw = episode.get(x_param)
    if not isinstance(x_data_raw, list) or not x_data_raw: st.error(f"X-axis data '{x_param}' missing/invalid."); return
    x_data_numeric = pd.to_numeric(x_data_raw, errors='coerce')

    # --- Common Plot Customization ---
    st.markdown("###### Common Plot Settings for Episode Details")
    plot_key_base = f"detail_plots_{ep_num}" # Common key base for this episode's plots
    default_settings = {
        'xlabel': x_param.replace('_', ' ').capitalize(), 'figsize_w': 10.0, 'figsize_h': 4.0,
        'xlabel_fontsize': 12, 'ylabel_fontsize': 12, 'tick_fontsize': 10, 'title_fontsize': 14, # Added title fontsize
        'line_color': "#1f77b4", 'line_width': 1.5, 'marker_color': "#ff7f0e", 'marker_size': 0, 'marker_style': 'o',
        'grid_on': True, 'num_xticks': 10
        # No title/ylabel here as they vary per plot
    }
    customize = st.checkbox("Customize Detail Plots", key=f"customize_{plot_key_base}", value=False)
    current_settings = render_mpl_plot_customization(plot_key_base, default_settings, plot_type='line') if customize else default_settings

    # --- Plot each selected Y variable ---
    for y_param in y_params:
        st.write(f"---"); y_data_raw = episode.get(y_param)
        if not isinstance(y_data_raw, list) or not y_data_raw: st.warning(f"Y-axis data '{y_param}' missing/invalid. Skip."); continue
        y_data_numeric = pd.to_numeric(y_data_raw, errors='coerce')

        min_len = min(len(x_data_numeric), len(y_data_numeric))
        x_aligned = x_data_numeric[:min_len]; y_aligned = y_data_numeric[:min_len]
        valid_mask = np.isfinite(x_aligned) & np.isfinite(y_aligned)
        x_final = x_aligned[valid_mask]; y_final = y_aligned[valid_mask]
        if len(x_final) == 0: st.warning(f"No valid data points for '{y_param}' vs '{x_param}'. Skip."); continue

        fig, ax = None, None # Initialize
        try:
            # --- Create Plot ---
            fig, ax = plt.subplots(figsize=current_settings['figsize'])
            ax.plot(x_final, y_final, color=current_settings['line_color'], linewidth=current_settings['line_width'],
                    marker=current_settings['marker_style'], markersize=current_settings['marker_size'], linestyle='-',
                    markerfacecolor=current_settings['marker_color'], markeredgecolor=current_settings['marker_color'])

            y_title = y_param.replace('_', ' ').capitalize()
            plot_title = f"Episode {ep_num}: {y_title} vs {current_settings['xlabel']}"
            ax.set_title(plot_title, fontsize=current_settings['title_fontsize'])
            ax.set_xlabel(current_settings['xlabel'], fontsize=current_settings['xlabel_fontsize'])
            ax.set_ylabel(y_title, fontsize=current_settings['ylabel_fontsize'])
            ax.grid(visible=current_settings['grid_on'], linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=current_settings['tick_fontsize'])
            fig.tight_layout()
            st.pyplot(fig)

            # --- Download Button ---
            download_plot_button(fig, f"episode_{ep_num}_{y_param}_vs_{x_param}", f"detail_{ep_num}_{y_param}")

        except Exception as e:
            st.error(f"Failed to plot '{y_param}' vs '{x_param}': {e}")
            logging.error(f"Detail plot error: {traceback.format_exc()}")
        finally:
            if fig is not None: plt.close(fig)


# Plot animation - adopted from app_deprecate (no changes needed here)
def plot_episode_animation(episode: dict):
    # ... (keep existing implementation) ...
    episode_index_or_id = episode.get('episode', 'N/A')
    st.write(f"**Animation Controls (Episode {episode_index_or_id})**")
    if not st.session_state.selected_folder_path: st.warning("Results folder path not set."); return
    anim_key_base = f"anim_ep_{episode_index_or_id}"
    show_animation = st.checkbox("Show/Generate Animation", value=False, key=f"{anim_key_base}_show")
    if not show_animation: st.caption("Enable checkbox to view/generate animation."); return
    animation_filename = f"animation_episode_{episode_index_or_id}.gif"
    animation_path = os.path.join(st.session_state.selected_folder_path, animation_filename)
    regenerate_ui = False
    if os.path.exists(animation_path):
        st.success(f"Animation found: `{os.path.basename(animation_path)}`")
        try:
            st.image(animation_path, caption=f"Cached Animation Ep {episode_index_or_id}")
            if st.button("Regenerate Animation", key=f"{anim_key_base}_regen_found"):
                regenerate_ui = True
                try: 
                    os.remove(animation_path)
                    st.info("Removed existing. Configure & Generate.") 
                except OSError as e: 
                    st.error(f"Err removing anim: {e}")
        except Exception as e: st.error(f"Err display anim: {e}. Regenerate?"); regenerate_ui = True
    else: regenerate_ui = True; st.info(f"Animation `{os.path.basename(animation_path)}` not found.")

    if regenerate_ui: # UI for Generation
        st.write("Configure & Generate Animation:")
        cols_anim = st.columns(3)
        with cols_anim[0]: fps = st.slider("FPS", 10, 60, 30, key=f"{anim_key_base}_fps"); speed = st.slider("Speed", 0.1, 5.0, 1.0, 0.1, format="%.1fx", key=f"{anim_key_base}_speed")
        with cols_anim[1]:
             cart_pos = episode.get('cart_position', []); valid_cart = [x for x in cart_pos if isinstance(x, (int, float)) and np.isfinite(x)]; max_abs = max(abs(x) for x in valid_cart) if valid_cart else 3.0
             def_xlim = (-max(3.0, abs(max_abs) * 1.1), max(3.0, abs(max_abs) * 1.1)); def_xlim_f = (float(def_xlim[0]), float(def_xlim[1]))
             x_lim = st.slider("X Limits", -15.0, 15.0, def_xlim_f, 0.5, key=f"{anim_key_base}_xlim")
        with cols_anim[2]: def_ylim = (-3.0, 3.0); y_lim = st.slider("Y Limits", -5.0, 5.0, def_ylim, 0.5, key=f"{anim_key_base}_ylim")

        if st.button("Generate Animation", key=f"{anim_key_base}_gen_button", type="primary"):
            req_keys = ['time', 'cart_position', 'pendulum_angle']; valid = True; data_lens = {}; min_len = float('inf')
            for key in req_keys: # Validation
                 data_list = episode.get(key)
                 if not isinstance(data_list, list) or not data_list: st.error(f"Anim fail: Missing '{key}'."); valid = False; break
                 data_lens[key] = len(data_list); min_len = min(min_len, len(data_list))
            if valid and len(set(data_lens.values())) > 1: st.warning(f"Inconsistent lengths: {data_lens}. Using min: {min_len}"); anim_ep_data = {k: episode[k][:min_len] for k in req_keys if k in episode}; [anim_ep_data.update({k:v}) for k,v in episode.items() if k not in anim_ep_data and not isinstance(v, list)]
            elif valid: anim_ep_data = episode
            else: return

            cfg_anim = {"fps": fps, "speed": speed, "x_lim": x_lim, "y_lim": y_lim, "dpi": 100}
            prog = st.progress(0.0); stat = st.empty(); stat.text("Init anim..."); animator = None; fig_anim = None
            try: # Generation and Saving
                animator = PendulumAnimator(anim_ep_data, cfg_anim); fig_anim = animator.fig
                stat.text("Creating frames..."); anim = animator.create_animation()
                if anim:
                    stat.text(f"Saving {os.path.basename(animation_path)}...")
                    total_fr = len(anim_ep_data['time'])
                    def prog_cb(curr, total): p = min(1.0, (curr + 1) / total) if total > 0 else 0; prog.progress(p)
                    anim.save(animation_path, writer='pillow', fps=cfg_anim['fps'], progress_callback=prog_cb)
                    prog.progress(1.0); stat.success(f"Animation saved!"); st.image(animation_path, caption=f"Generated Anim Ep {episode_index_or_id}")
                else: stat.error("Anim creation failed."); prog.empty()
            except Exception as e: stat.error(f"Failed anim gen/save: {e}"); prog.empty(); logging.error(f"Anim error: {traceback.format_exc()}")
            finally: # Cleanup
                 if fig_anim and plt.fignum_exists(fig_anim.number): plt.close(fig_anim)
                 gc.collect()


# --------------------------- Performance Overview Tab ---------------------------
def performance_overview():
    """Displays performance plots (Matplotlib) and Q-table comparison."""
    st.subheader("Performance Overview")
    # --- Termination Reason Plot ---
    plot_termination_reason_distribution_mpl(st.session_state.get('summary_df'))
    st.divider()
    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    sim_data = loaded_data.get('simulation_data', [])
    if sim_data:
        # --- Overview Metrics ---
        plot_overview_metrics_mpl(sim_data) # Includes internal dividers now
        # --- Heatmap ---
        plot_heatmap(sim_data, st.session_state.heatmap_needs_update)
        st.session_state.heatmap_needs_update = False
        st.divider()
        # --- Q-Table comparison ---
        if st.session_state.get('config'): display_qtable_comparison(sim_data, st.session_state.config)
        else: st.warning("Config missing for Q-Table.")
    elif not st.session_state.selected_folder_path: st.info("Select a results folder.")
    elif not st.session_state.selected_episode_numbers_to_load: st.info("Select & load episode data.")
    else: st.warning("No simulation data found for loaded episodes."); st.info("Check files or filters.")


# --------------------------- Variables Discovery Tab (Stubbed Out) ---------------------------
def variables_discovery():
    """Pestaña para generación de configuraciones de análisis (INACTIVE)."""
    st.subheader("Analysis Configurator")
    st.info("This section is under development and currently inactive.")
    st.caption("Functionality for defining data preparation and analysis configurations for external scripts will be added here.")
    # loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    # sim_data = loaded_data.get('simulation_data', [])
    # if not sim_data: st.warning("Load episode data first (when implemented).")


# --------------------------- Función Principal (main) ---------------------------
def main():
    if not utils_imports_ok: st.error("Dashboard cannot start: Import errors."); st.stop()
    st.set_page_config(page_title="RL Control Dashboard", page_icon="🤖", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #333;'><b>Dynamic System Simulation Dashboard</b></h1>", unsafe_allow_html=True)
    initialize_session_state()
    render_sidebar()
    # Removed "Episode Boxplots" tab, integrate elsewhere if needed
    tabs = ["Introduction & Config", "Performance Overview", "Episode Details", "Analysis Configurator"]
    icons = ['house-door', 'bar-chart-line', 'search', 'tools']
    selected_page = option_menu(
        menu_title=None, options=tabs, icons=icons, menu_icon='cast', default_index=0, orientation='horizontal',
        styles={ "container": {"padding": "5px !important", "background-color": "#f0f2f6", "border-radius": "5px"}, "icon": {"color": "#023047", "font-size": "18px"}, "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px 5px", "--hover-color": "#dfe6e9", "color": "#023047", "border-radius": "5px", "padding": "10px 15px"}, "nav-link-selected": {"background-color": "#219ebc", "color": "white", "font-weight": "bold"}, }
    )
    if st.session_state.selected_folder_path:
        if selected_page == "Introduction & Config": introduction_page()
        elif selected_page == "Performance Overview": performance_overview()
        elif selected_page == "Episode Details": episode_details()
        elif selected_page == "Analysis Configurator": variables_discovery() # Show stub
    else: st.info("👋 Welcome! Select a simulation results folder.")

# Helper class for JSON saving if needed (not primary use now)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()