# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for visualizing Dynamic System Simulation Results.
Version: 1.5.0 (Performance, Usability, Q-Table Fixes)
"""

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import traceback
import logging
from typing import Any, Dict, List, Optional, Tuple, Union # Added Union
import gc # Import garbage collector

# Importar funciones desde app_utils.py
try:
    from code_project.deprecated.app_utils import (
        load_folder_structure,
        load_metadata,
        load_summary_data,
        load_selected_episodes,
        resaltar_maximo, # Necesaria para Q-tables
        PendulumAnimator # Necesaria para animación
    )
    utils_imports_ok = True
except ImportError as e:
     # Error crítico si no se pueden importar utils
     st.error(f"FATAL ERROR: Failed to import utilities from app_utils.py: {e}. Dashboard cannot function. Check file existence and dependencies.")
     logging.critical(f"app_utils import failed: {e}", exc_info=True)
     utils_imports_ok = False
     # Use st.stop() within the main execution flow, not globally here
     # st.stop() # Stop execution if utils are missing


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------- Configuración Inicial ---------------------------
RESULTS_FOLDER = "results_history" # Asegúrate que coincida con tu config

def initialize_session_state():
    """Inicializa las variables en st.session_state si no existen."""
    defaults = {
        'folders': [],
        'selected_folder_path': None,
        'metadata': None,
        'config': {},
        'summary_df': None,
        'loaded_episode_data': {'simulation_data': []}, # Initialize with empty list
        'available_episodes_in_summary': [],
        'selected_episode_numbers_to_load': [],
        'current_episode_index': 0, # Index within the loaded_episode_data['simulation_data'] list
        'available_loaded_episode_numbers': [], # List of numeric episode numbers actually loaded
        'plot_selections': {'x_param': 'time', 'y_params': []},
        'heatmap_needs_update': True, # Flag to trigger heatmap data prep
        'heatmap_data_cache': None, # Cache for prepared heatmap data
        'heatmap_params_cache': None # Cache for available params in heatmap data
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --------------------------- Funciones de Interfaz ---------------------------

def render_sidebar():
    """Renderiza la barra lateral y maneja la selección de carpetas y filtrado de datos."""
    st.sidebar.title("Simulation Results")

    # --- Folder Selection ---
    if st.sidebar.button("🔄 Refresh Folders"):
        st.cache_data.clear() # Clear streamlit's data cache
        # Clear relevant session state parts forcefully
        keys_to_reset = [
            'folders', 'selected_folder_path', 'metadata', 'config',
            'summary_df', 'loaded_episode_data', 'available_episodes_in_summary',
            'selected_episode_numbers_to_load', 'current_episode_index',
            'available_loaded_episode_numbers', 'heatmap_needs_update',
            'heatmap_data_cache', 'heatmap_params_cache'
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        initialize_session_state() # Re-initialize cleared state
        # No explicit rerun needed here, subsequent widgets will trigger it
        st.sidebar.success("Cache cleared and state reset. Select folder again.")


    # Load initial folders if list is empty
    if not st.session_state.get('folders'):
        try:
            st.session_state.folders = load_folder_structure(RESULTS_FOLDER)
        except Exception as e:
             st.sidebar.error(f"Error loading folder structure: {e}")
             st.session_state.folders = []

    if not st.session_state.folders:
        st.sidebar.warning(f"No simulation folders found in '{RESULTS_FOLDER}'.")
        return # Stop if no folders

    # Determine current selection index safely
    current_selection_index = 0
    current_folder_name = os.path.basename(st.session_state.selected_folder_path) if st.session_state.selected_folder_path else None
    if current_folder_name and current_folder_name in st.session_state.folders:
         try:
            current_selection_index = st.session_state.folders.index(current_folder_name)
         except ValueError:
            pass # Keep index 0 if name not found

    selected_folder_name = st.sidebar.selectbox(
        "Select a results folder:",
        st.session_state.folders,
        index=current_selection_index,
        key="folder_selector" # Consistent key
    )

    if not selected_folder_name:
        return

    selected_folder_path_new = os.path.join(RESULTS_FOLDER, selected_folder_name)

    # --- Load Metadata and Summary if folder selection changed ---
    if selected_folder_path_new != st.session_state.get('selected_folder_path'):
        st.session_state.selected_folder_path = selected_folder_path_new
        st.sidebar.info(f"Selected: {selected_folder_name}")

        # Use spinner for loading feedback
        with st.spinner("Loading metadata and summary..."):
            # Clear potentially stale data related to the *previous* folder
            keys_to_reset_on_folder_change = [
                'metadata', 'config', 'summary_df', 'loaded_episode_data',
                'available_episodes_in_summary', 'selected_episode_numbers_to_load',
                'current_episode_index', 'available_loaded_episode_numbers',
                'heatmap_needs_update', 'heatmap_data_cache', 'heatmap_params_cache'
            ]
            for key in keys_to_reset_on_folder_change:
                 if key in st.session_state:
                      del st.session_state[key]
            # Re-initialize only the necessary parts after clearing
            st.session_state.loaded_episode_data = {'simulation_data': []}
            st.session_state.available_episodes_in_summary = []
            st.session_state.selected_episode_numbers_to_load = []
            st.session_state.current_episode_index = 0
            st.session_state.available_loaded_episode_numbers = []
            st.session_state.heatmap_needs_update = True
            st.session_state.heatmap_data_cache = None
            st.session_state.heatmap_params_cache = None

            # Load metadata and config for the *new* folder
            st.session_state.metadata = load_metadata(st.session_state.selected_folder_path) # Uses cache_data
            if st.session_state.metadata and 'config_parameters' in st.session_state.metadata:
                st.session_state.config = st.session_state.metadata.get('config_parameters', {})
            else:
                 st.session_state.config = {}
                 st.sidebar.warning("Config parameters not found in metadata.")

            # Load summary and extract available episodes
            st.session_state.summary_df = load_summary_data(st.session_state.selected_folder_path) # Uses cache_data, shows internal errors/warnings
            if st.session_state.summary_df is not None and 'episode' in st.session_state.summary_df.columns:
                 try:
                     # Ensure unique, integer, sorted episode numbers
                     unique_eps = pd.to_numeric(st.session_state.summary_df['episode'], errors='coerce').dropna().unique()
                     if len(unique_eps) > 0:
                         st.session_state.available_episodes_in_summary = sorted(unique_eps.astype(int))
                         logging.info(f"Found {len(st.session_state.available_episodes_in_summary)} unique episodes in summary.")
                     else:
                          st.sidebar.warning("No valid numeric episode numbers found in summary 'episode' column.")
                          st.session_state.available_episodes_in_summary = []
                 except Exception as e:
                      st.sidebar.error(f"Error processing episode numbers from summary: {e}")
                      st.session_state.available_episodes_in_summary = []
            # Trigger a rerun to ensure UI reflects the new folder's state
            st.rerun()

    # --- Filtering Options (only if summary loaded successfully) ---
    st.sidebar.subheader("Filter Episodes Before Loading")
    if st.session_state.summary_df is not None and not st.session_state.summary_df.empty and st.session_state.available_episodes_in_summary:
        df_summary = st.session_state.summary_df
        available_episodes_in_summary_nums = st.session_state.available_episodes_in_summary

        load_all = st.sidebar.checkbox(f"Load all {len(available_episodes_in_summary_nums)} summarized episodes", value=False, key="load_all_check")

        # Initialize filtered_summary for this run
        filtered_summary_df = df_summary[df_summary['episode'].isin(available_episodes_in_summary_nums)].copy()

        if not load_all:
            st.sidebar.write("Apply filters:")

            # Filter by episode range
            min_ep, max_ep = min(available_episodes_in_summary_nums), max(available_episodes_in_summary_nums)
            # Ensure options are integers
            options_list = list(range(int(min_ep), int(max_ep) + 1))

            # Check if range is valid before creating slider
            if min_ep <= max_ep:
                selected_ep_range = st.sidebar.select_slider(
                    "Select episode range:",
                    options=options_list,
                    value=(int(min_ep), int(max_ep)), # Ensure value is int tuple
                    key="filter_ep_range"
                )
                filtered_summary_df = filtered_summary_df[(filtered_summary_df['episode'] >= selected_ep_range[0]) & (filtered_summary_df['episode'] <= selected_ep_range[1])]
            else:
                st.sidebar.warning("Invalid episode range in summary.")


            # Filter by variable value (simplified: exclude _std, _min, _max, _mean)
            all_numeric_cols = filtered_summary_df.select_dtypes(include=np.number).columns.tolist()
            cols_to_exclude_suffixes = ('_std', '_min', '_max')
            # Keep 'episode' temporarily for identification, remove before showing options
            essential_cols = ['total_reward', 'episode_time', 'final_epsilon', 'final_learning_rate'] # Add other key summary cols if needed

            filter_variable_options = [
                col for col in all_numeric_cols
                if col == 'episode' or col in essential_cols or not col.endswith(cols_to_exclude_suffixes)
            ]
            # Remove 'episode' from user-selectable options
            if 'episode' in filter_variable_options:
                filter_variable_options.remove('episode')

            if filter_variable_options:
                default_filter_var = 'total_reward' if 'total_reward' in filter_variable_options else filter_variable_options[0]
                filter_var_index = filter_variable_options.index(default_filter_var)

                filter_variable = st.sidebar.selectbox(
                    "Filter by summary variable value:",
                    options=sorted(filter_variable_options, reverse=True), # Sort for user convenience
                    index=filter_var_index,
                    key="filter_var_select"
                )

                # Check if variable exists and has non-null values *in the currently filtered data*
                if filter_variable and filter_variable in filtered_summary_df.columns and not filtered_summary_df[filter_variable].isnull().all():
                    try:
                        min_val = float(filtered_summary_df[filter_variable].min())
                        max_val = float(filtered_summary_df[filter_variable].max())

                        if pd.notna(min_val) and pd.notna(max_val):
                            if min_val == max_val:
                                st.sidebar.text(f"{filter_variable}: {min_val:.4g}")
                            else:
                                # Determine a reasonable step, avoid zero or too small steps
                                val_range = max_val - min_val
                                step_val = max(val_range / 100, 1e-6) # Avoid zero step, use small minimum
                                # Ensure step isn't excessively small compared to range
                                if step_val < 1e-9 * abs(val_range): step_val = 1e-9 * abs(val_range)

                                # Format based on value magnitude and step
                                num_decimals = -int(np.floor(np.log10(step_val))) + 1 if step_val > 0 else 4
                                slider_format = f"%.{max(0, num_decimals)}f" # Dynamic precision format, at least 0 decimals
                                if abs(max_val) > 1e4 or abs(min_val) > 1e4 or step_val > 1e3: slider_format = "%.3g" # Use general format for large numbers


                                selected_val_range = st.sidebar.slider(
                                    f"Select range for {filter_variable}:",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"val_range_{filter_variable}",
                                    step=step_val,
                                    format=slider_format # Use dynamic format
                                )
                                filtered_summary_df = filtered_summary_df[
                                    (filtered_summary_df[filter_variable] >= selected_val_range[0]) &
                                    (filtered_summary_df[filter_variable] <= selected_val_range[1])
                                ]
                        else:
                            st.sidebar.warning(f"'{filter_variable}' contains NaNs or invalid values, cannot create slider.")
                    except Exception as slider_ex:
                        st.sidebar.error(f"Error creating slider for '{filter_variable}': {slider_ex}")

                elif filter_variable:
                     st.sidebar.warning(f"'{filter_variable}' has only missing values after previous filters.")
            # else: st.sidebar.info("No suitable summary variables for value filtering.")


            # Filter by termination reason
            if 'termination_reason' in filtered_summary_df.columns:
                 reasons = sorted(filtered_summary_df['termination_reason'].dropna().unique().tolist())
                 if reasons:
                     selected_reasons = st.sidebar.multiselect(
                         "Filter by termination reason:",
                         options=reasons,
                         default=reasons, # Select all by default
                         key="filter_term"
                     )
                     # Apply filter only if selection differs from all available reasons
                     if set(selected_reasons) != set(reasons):
                          filtered_summary_df = filtered_summary_df[filtered_summary_df['termination_reason'].isin(selected_reasons)]

            # REMOVED: Select Top N / Sorting section

            # Final list of episode numbers from the fully filtered summary, always sorted
            st.session_state.selected_episode_numbers_to_load = sorted(filtered_summary_df['episode'].astype(int).tolist()) if not filtered_summary_df.empty else []

        else: # Load all episodes available in the initial summary
            st.session_state.selected_episode_numbers_to_load = st.session_state.available_episodes_in_summary

        # Display count and potentially the list
        num_selected = len(st.session_state.selected_episode_numbers_to_load)
        if num_selected > 0:
            st.sidebar.info(f"{num_selected} episodes match criteria.")
            # Keep expander collapsed by default for large lists
            with st.sidebar.expander(f"Show {num_selected} selected episode numbers", expanded=False):
                 st.write(st.session_state.selected_episode_numbers_to_load)
        else:
            st.sidebar.warning("No episodes match the current filter criteria.")

    elif st.session_state.selected_folder_path: # Folder selected, but no summary
        st.sidebar.warning("Summary data not found/empty. Cannot filter episodes.")
        st.session_state.selected_episode_numbers_to_load = []


    # --- Load Data Button ---
    load_button_disabled = not st.session_state.selected_episode_numbers_to_load
    if st.sidebar.button("Load Selected Episode Data", key="load_data_button", disabled=load_button_disabled, type="primary"):
        # Check path and selection again just before loading
        if st.session_state.selected_folder_path and st.session_state.selected_episode_numbers_to_load:
            # Convert list to tuple for caching load_selected_episodes
            episodes_to_load_tuple = tuple(st.session_state.selected_episode_numbers_to_load)

            with st.spinner(f"Loading data for {len(episodes_to_load_tuple)} episodes..."):
                st.session_state.loaded_episode_data = {'simulation_data': []} # Clear previous, ensure structure
                st.session_state.available_loaded_episode_numbers = []
                st.session_state.current_episode_index = 0
                st.session_state.heatmap_needs_update = True # Mark heatmap data as stale
                st.session_state.heatmap_data_cache = None
                st.session_state.heatmap_params_cache = None
                gc.collect() # Garbage collect before large load

                # Call cached function
                loaded_data = load_selected_episodes(
                    st.session_state.selected_folder_path,
                    episodes_to_load_tuple # Pass tuple
                )
                st.session_state.loaded_episode_data = loaded_data or {'simulation_data': []} # Store result, ensure dict structure

                # Extract available loaded episode numbers AFTER loading
                if st.session_state.loaded_episode_data and st.session_state.loaded_episode_data.get('simulation_data'):
                     loaded_sim_data = st.session_state.loaded_episode_data['simulation_data']
                     # Ensure they are numeric and sorted
                     valid_nums = []
                     for ep_data in loaded_sim_data:
                         ep_num = ep_data.get('episode')
                         try:
                             valid_nums.append(int(ep_num))
                         except (ValueError, TypeError):
                             logging.warning(f"Non-numeric episode number found in loaded data: {ep_num}")
                     st.session_state.available_loaded_episode_numbers = sorted(list(set(valid_nums)))
                     st.sidebar.success(f"Loaded data for {len(st.session_state.available_loaded_episode_numbers)} episodes.")
                elif loaded_data: # Function returned but no simulation data
                     st.sidebar.warning("Loading function ran but found no simulation data for selected episodes.")
                else: # Function might have failed or returned None
                     st.sidebar.error("Failed to load episode data.")

                # Reset index after load
                st.session_state.current_episode_index = 0
                # st.rerun() # Rerun happens naturally or can be triggered if needed
    st.sidebar.markdown("---")

# --------------------------- Pestañas de la Aplicación ---------------------------

def display_simulation_config():
    """Displays the simulation configuration parameters."""
    st.subheader("Simulation Configuration")
    config_data = st.session_state.get('config', {})
    if config_data:
        try:
            st.json(config_data, expanded=False) # Show collapsed by default
        except Exception as e:
             st.error(f"Error displaying configuration as JSON: {e}")
             # Fallback to text representation
             try:
                 st.text(json.dumps(config_data, indent=2))
             except Exception as dump_e:
                 st.error(f"Could not even dump config to text: {dump_e}")
                 st.write(config_data) # Last resort
    else:
        st.warning("Configuration parameters not available (Metadata missing or empty).")

def introduction_page():
    """Displays the introduction and simulation configuration."""
    col1, col2= st.columns([1, 2]) # Adjust column widths if needed
    with col1:
        logo_path = 'logo_final.png'
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width='auto')
        else:
            logging.warning("Logo image not found at: %s", os.path.abspath(logo_path))
            st.caption("Logo image not found")

        st.markdown("<h3 style='text-align: center; margin-bottom: 0;'>Simulation Results Dashboard</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: small;'>Overview and analysis</p>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Contact:")
        st.write("Davor Matías Samuel Ibarra Pérez")
        st.write("Ph.D (s) en Automática, Robótica e Informática Industrial")
        st.write("Universitat Politècnica de València / Universidad de Santiago de Chile")
        st.write("📧 dibasam@doctor.upv.es / davor.ibarra@usach.cl")
        st.write("Ph.D Javier Sanchis")
        st.write("Universitat Politècnica de València")
        st.write("📧 jsanchis@isa.upv.es")
        st.divider()
    with col2:
        if st.session_state.selected_folder_path:
             display_simulation_config()
             st.divider()
             # Display summary statistics if available
             if st.session_state.summary_df is not None:
                  st.subheader("Overall Summary Statistics")
                  st.caption("(Based on loaded `summary.xlsx`)")
                  try:
                       # Use st.dataframe for better interactivity, apply styling
                       st.dataframe(st.session_state.summary_df.describe().style.format(precision=4))
                  except Exception as e:
                       st.error(f"Error displaying summary statistics: {e}")
                       st.dataframe(st.session_state.summary_df.describe()) # Fallback without styling
             else:
                  st.warning("Summary data (`summary.xlsx`) not loaded or file not found.")
        else:
            st.info("Select a results folder from the sidebar to view configuration and summary.")


def performance_overview():
    """Displays performance plots and Q-table comparison based on loaded data."""
    st.subheader("Performance Overview for Loaded Episodes")

    # Check if data is loaded
    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    sim_data = loaded_data.get('simulation_data', [])

    if sim_data:
        plot_performance_overview(sim_data) # Matplotlib plots for reward, time, performance
        st.divider()
        # Pass flag to heatmap to indicate if data needs reprocessing
        plot_heatmap(sim_data, st.session_state.heatmap_needs_update)
        st.session_state.heatmap_needs_update = False # Reset flag after processing
        st.divider()

        # Q-Table comparison needs config
        if st.session_state.get('config'):
            display_qtable_comparison(sim_data, st.session_state.config)
        else:
             st.warning("Cannot display Q-Table comparison: Configuration data missing (check metadata.json).")

    # Guidance messages if data not loaded
    elif not st.session_state.selected_folder_path:
         st.info("Please select a results folder from the sidebar.")
    elif not st.session_state.selected_episode_numbers_to_load:
         st.info("Select episodes using the filters in the sidebar and click 'Load Selected Episode Data'.")
    elif 'load_data_button' not in st.session_state: # Button hasn't been pressed yet maybe
         st.info("Use the sidebar to select and load episode data.")
    else: # Data loading attempted but sim_data is empty
         st.info("No simulation data found within the loaded files for the selected episodes. Check file contents or filters, or try loading again.")

# --- Plotting Functions ---

def plot_performance_overview(data: list):
    """Genera gráficos de rendimiento general para los episodios cargados utilizando Matplotlib."""
    st.write("### Performance Metrics Across Loaded Episodes")
    if not data:
        st.warning("No episode data available for overview plots.")
        return

    episode_metrics = []
    for episode in data:
        ep_num = episode.get('episode')
        reward_list = episode.get('cumulative_reward')
        time_list = episode.get('time')
        if ep_num is not None and isinstance(reward_list, list) and reward_list and isinstance(time_list, list) and time_list:
            # Ensure values are numeric before using
            try:
                final_reward = float(reward_list[-1])
                final_time = float(time_list[-1])
                performance = final_reward / final_time if final_time != 0 else 0
                episode_metrics.append((int(ep_num), final_reward, final_time, performance))
            except (ValueError, TypeError, IndexError) as e:
                 logging.warning(f"Skipping ep {ep_num or 'N/A'} in overview plot due to data issue: {e}.")
        else:
            logging.warning(f"Skipping ep {ep_num or 'N/A'} in overview plot: missing/invalid reward or time lists.")

    if not episode_metrics:
        st.warning("No valid episodes found with numeric reward/time for overview plot.")
        return

    # Ensure sorting by episode number (which should be int now)
    episode_metrics.sort(key=lambda x: x[0])
    episode_numbers = [x[0] for x in episode_metrics]
    cumulative_rewards = [x[1] for x in episode_metrics]
    final_times = [x[2] for x in episode_metrics]
    performances = [x[3] for x in episode_metrics]

    plots_config = [
        {"title": "Cumulative Reward", "data": cumulative_rewards, "ylabel": "Cumulative Reward"},
        {"title": "Episode Duration", "data": final_times, "ylabel": "Duration (s)"},
        {"title": "Performance (Reward per Time)", "data": performances, "ylabel": "Performance"},
    ]

    for config in plots_config:
        plot_title = config["title"]
        y_data = config["data"]
        default_ylabel = config["ylabel"]
        st.write(f"#### {plot_title}")
        plot_key_base = "".join(filter(str.isalnum, plot_title)).lower()
        customize = st.checkbox(f"Customize '{plot_title}' Plot", key=f"customize_{plot_key_base}", value=False)

        # Default plot settings
        title_to_use = plot_title
        xlabel_to_use = "Episode"
        ylabel_to_use = default_ylabel
        lc="#1f77b4" # line color
        mc="#ff7f0e" # marker color
        lw=1.5 # line width
        ms=3 # marker size
        grid=True
        num_bins=10 # Target number of x-ticks

        if customize:
            with st.expander("Plot Settings"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    cust_title = st.text_input("Title", value=plot_title, key=f"title_{plot_key_base}")
                    cust_xlabel = st.text_input("X-Label", value="Episode", key=f"xlabel_{plot_key_base}")
                    cust_ylabel = st.text_input("Y-Label", value=default_ylabel, key=f"ylabel_{plot_key_base}")
                with col2:
                    line_color = st.color_picker("Line Color", value=lc, key=f"lcolor_{plot_key_base}")
                    marker_color = st.color_picker("Marker Color", value=mc, key=f"mcolor_{plot_key_base}")
                    line_width = st.slider("Line Width", 0.5, 5.0, lw, 0.5, key=f"lw_{plot_key_base}")
                with col3:
                    marker_size = st.slider("Marker Size", 0, 10, ms, key=f"msize_{plot_key_base}")
                    grid_on = st.checkbox("Show Grid", value=grid, key=f"grid_{plot_key_base}")
                    num_xticks = st.slider("Approx. X-Ticks", 5, 50, num_bins, key=f"xticks_{plot_key_base}")
            # Update settings if customized
            title_to_use=cust_title
            xlabel_to_use=cust_xlabel
            ylabel_to_use=cust_ylabel
            lc=line_color
            mc=marker_color
            lw=line_width
            ms=marker_size
            grid=grid_on
            num_bins=num_xticks

        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(episode_numbers, y_data, marker='o', linestyle='-', color=lc, linewidth=lw, markersize=ms, markerfacecolor=mc, markeredgecolor=mc)
        ax.set_title(title_to_use, fontsize=14)
        ax.set_xlabel(xlabel_to_use, fontsize=12)
        ax.set_ylabel(ylabel_to_use, fontsize=12)
        ax.grid(visible=grid, linestyle='--', alpha=0.6)

        # Dynamic Ticks based on actual episode numbers
        if len(episode_numbers) > 1:
             min_ep, max_ep = min(episode_numbers), max(episode_numbers)
             tick_range = max_ep - min_ep
             if tick_range == 0:
                 ticks = [min_ep]
             else:
                 # Calculate a reasonable step avoiding too many ticks
                 tick_step = max(1, int(np.ceil(tick_range / max(1, num_bins))))
                 # Adjust step if it still results in too many ticks (e.g., > 50)
                 if tick_range / tick_step > 50:
                     tick_step = max(1, int(np.ceil(tick_range / 15))) # Reduce target bins if needed
                 # Generate ticks making sure they are integers
                 ticks = np.arange(min_ep, max_ep + tick_step, tick_step, dtype=int)
             ax.set_xticks(ticks)
             plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        elif len(episode_numbers) == 1:
            ax.set_xticks(episode_numbers) # Single tick if only one episode

        ax.tick_params(axis='both', which='major', labelsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to release memory


def prepare_heatmap_data(data: list) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[List[str]]]:
    """Prepares concatenated data for heatmaps, returning data dict and available params."""
    if not data:
        return None, None

    all_metrics = {}
    available_params_set = set()
    valid_episode_found = False
    max_len_overall = 0 # Track max length across all episodes if needed (not strictly necessary for concat)

    logging.info(f"Preparing heatmap data from {len(data)} episodes...")
    with st.spinner("Aggregating data for heatmaps..."): # Add spinner here
        for ep_idx, episode in enumerate(data):
            is_ep_valid = False
            temp_metrics_ep = {}
            max_len_ep = 0
            ep_num_for_log = episode.get('episode', f'Index {ep_idx}')

            # Iterate through items safely
            items_to_process = list(episode.items())
            for key, values in items_to_process:
                # Check if it's a list containing numeric data
                if isinstance(values, list) and values:
                    # Attempt conversion early to check type, handle potential errors
                    try:
                        # Check first element's type robustly
                        if isinstance(values[0], (int, float, np.number)):
                            numeric_values = pd.to_numeric(values, errors='coerce')
                            # Only add if it contains *some* valid numbers
                            if not np.isnan(numeric_values).all():
                                temp_metrics_ep[key] = numeric_values # Store as numpy array now
                                available_params_set.add(key)
                                max_len_ep = max(max_len_ep, len(numeric_values))
                                is_ep_valid = True
                    except (TypeError, IndexError):
                        # Ignore non-numeric lists or errors accessing first element
                        pass
                    except Exception as conv_e:
                        logging.warning(f"Error converting {key} in ep {ep_num_for_log}: {conv_e}")

            if not is_ep_valid:
                # logging.debug(f"No valid list data found in ep {ep_num_for_log}.")
                continue # Skip episode if no valid lists found

            valid_episode_found = True
            max_len_overall = max(max_len_overall, max_len_ep)

            # Add data from this episode to the main dictionary, padding/truncating implicitly during concat
            for key in available_params_set: # Use the growing set of all params found so far
                if key not in all_metrics:
                    all_metrics[key] = [] # Initialize list for this param if new

                values_to_add = temp_metrics_ep.get(key) # Get the numpy array prepared earlier
                if values_to_add is not None:
                    all_metrics[key].append(values_to_add) # Append the whole array (or list)
                else:
                    # If this episode didn't have this param, append NaNs of appropriate length?
                    # Concatenation below handles this better. Append None or empty array placeholder.
                     all_metrics[key].append(np.array([np.nan] * max_len_ep)) # Pad with NaNs for this episode's max len


    if not valid_episode_found:
        logging.warning("No valid episodes with numeric list data found for heatmap prep.")
        return None, None

    # Concatenate arrays for each parameter
    final_params_list = sorted(list(available_params_set))
    concatenated_metrics = {}
    final_params_available = []

    for key in final_params_list:
        list_of_arrays = all_metrics.get(key, [])
        if list_of_arrays:
            try:
                # Pad arrays within the list before concatenating if lengths differ significantly?
                # Or rely on pandas/numpy concat behavior (might be inefficient).
                # Let's try direct concatenation first. It should handle NaNs.
                concatenated_array = np.concatenate(list_of_arrays)
                # Final check: ensure it's not all NaNs after concat
                if not np.isnan(concatenated_array).all():
                    concatenated_metrics[key] = concatenated_array
                    final_params_available.append(key)
                else:
                    logging.warning(f"Parameter '{key}' is all NaN after concatenation. Excluding.")
            except Exception as concat_e:
                logging.error(f"Error concatenating data for '{key}': {concat_e}")
        else:
             logging.warning(f"No data collected for parameter '{key}'.")


    if len(final_params_available) < 2:
        logging.warning("Less than 2 valid numeric parameters available after aggregation for heatmap.")
        return None, None

    logging.info(f"Heatmap data prepared. Available params: {final_params_available}")
    return concatenated_metrics, final_params_available


def plot_heatmap(data: list, needs_update: bool):
    """Genera un heatmap interactivo usando Matplotlib, optimizado para calcular solo al hacer clic."""
    st.write("### Heatmaps of Trajectories")

    # --- Data Preparation (run only if needed or data changed) ---
    if needs_update or st.session_state.heatmap_data_cache is None:
        st.session_state.heatmap_data_cache, st.session_state.heatmap_params_cache = prepare_heatmap_data(data)
        if st.session_state.heatmap_data_cache is None:
            st.warning("Could not prepare data for heatmaps.")
            return

    # Use cached data and params
    heatmap_data = st.session_state.heatmap_data_cache
    available_params = st.session_state.heatmap_params_cache

    if not heatmap_data or not available_params or len(available_params) < 2:
        st.warning("Insufficient data or parameters available for heatmap.")
        return
    # --- End Data Prep ---

    st.write("Select parameters for heatmap:")
    col1, col2 = st.columns(2)
    with col1: # X Param
        # Try to find 'time' or default to first param
        x_idx = available_params.index('time') if 'time' in available_params else 0
        x_param = st.selectbox("X-axis parameter", available_params, index=x_idx, key="heatmap_x")
    with col2: # Y Param
        # Try to find 'pendulum_angle', ensure it's not same as x, or default
        default_y = 'pendulum_angle'
        y_idx = 0
        if default_y in available_params and default_y != x_param:
            y_idx = available_params.index(default_y)
        elif len(available_params) > 1:
            # Find first param that is not x_param
            y_idx = next((i for i, p in enumerate(available_params) if p != x_param), 0)

        y_param = st.selectbox("Y-axis parameter", available_params, index=y_idx, key="heatmap_y")

    if x_param == y_param:
        st.warning("X and Y parameters must be different.")
        return

    # --- Customization (Widgets always visible, logic applied on button click) ---
    customize = st.checkbox("Customize Heatmap", key="customize_heatmap", value = False)
    default_title = f"Heatmap: {y_param.capitalize()} vs {x_param.capitalize()} (Frequency)"
    # Default values for customization widgets
    bin_count_default = 150
    cmap_choice_default = 'hot'
    x_min_def, x_max_def, y_min_def, y_max_def = None, None, None, None # Default to auto range
    cbar_min_def, cbar_max_def = None, None # Default colorbar range to auto

    # Pre-calculate default percentile ranges if needed for placeholders
    x_data_all = heatmap_data.get(x_param)
    y_data_all = heatmap_data.get(y_param)
    if x_data_all is not None:
        x_data_clean = x_data_all[~np.isnan(x_data_all)]
        if len(x_data_clean) > 1:
            x_min_def, x_max_def = tuple(np.percentile(x_data_clean, [0, 100]))
    if y_data_all is not None:
        y_data_clean = y_data_all[~np.isnan(y_data_all)]
        if len(y_data_clean) > 1:
             y_min_def, y_max_def = tuple(np.percentile(y_data_clean, [0.5, 99.5]))

    # Widget values will be read inside the button click event
    if customize:
        with st.expander("Heatmap Settings"):
            cust_title_widget = st.text_input("Title", value=default_title, key="heatmap_title_widget")
            bin_count_widget = st.slider("Number of Bins", 50, 500, bin_count_default, key="heatmap_bins_widget")
            try:
                cmaps = plt.colormaps()
                cmap_idx_default = cmaps.index(cmap_choice_default) if cmap_choice_default in cmaps else 0
            except: # Handle case where plt.colormaps() might fail or list is different
                cmaps = ['hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
                cmap_idx_default = 0
            cmap_choice_widget = st.selectbox("Color Map", cmaps, index=cmap_idx_default, key="heatmap_cmap_widget")

            st.write("Axis & Colorbar Ranges (blank/None for auto):")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                 # Use number_input with None default for auto-range
                 x_min_widget = st.number_input(f"X-Min ({x_param})", value=x_min_def, format="%g", key="heatmap_xmin_widget")
                 x_max_widget = st.number_input(f"X-Max ({x_param})", value=x_max_def, format="%g", key="heatmap_xmax_widget")
                 cbar_min_widget = st.number_input(f"Colorbar Min", value=cbar_min_def, format="%g", key="heatmap_cmin_widget")
            with rcol2:
                 y_min_widget = st.number_input(f"Y-Min ({y_param})", value=y_min_def, format="%g", key="heatmap_ymin_widget")
                 y_max_widget = st.number_input(f"Y-Max ({y_param})", value=y_max_def, format="%g", key="heatmap_ymax_widget")
                 cbar_max_widget = st.number_input(f"Colorbar Max", value=cbar_max_def, format="%g", key="heatmap_cmax_widget")

    # --- Generate Button ---
    if st.button("Generate Heatmap", key="gen_heatmap_btn"):
        # --- Get data and settings INSIDE button click ---
        x_data = heatmap_data.get(x_param)
        y_data = heatmap_data.get(y_param)

        if x_data is None or y_data is None:
            st.error(f"Data for selected parameters ('{x_param}', '{y_param}') not found.")
            return

        # Retrieve settings from widgets if customize is enabled
        title_to_use = default_title
        bin_count_to_use = bin_count_default
        cmap_to_use = cmap_choice_default
        x_range_to_use = None # Auto range default
        y_range_to_use = None
        c_range_to_use = (None, None) # vmin, vmax

        if customize:
            # Read values from the customization widgets stored in session state via their keys
            title_to_use = st.session_state.get("heatmap_title_widget", default_title)
            bin_count_to_use = st.session_state.get("heatmap_bins_widget", bin_count_default)
            cmap_to_use = st.session_state.get("heatmap_cmap_widget", cmap_choice_default)

            x_min_val = st.session_state.get("heatmap_xmin_widget")
            x_max_val = st.session_state.get("heatmap_xmax_widget")
            y_min_val = st.session_state.get("heatmap_ymin_widget")
            y_max_val = st.session_state.get("heatmap_ymax_widget")
            cbar_min_val = st.session_state.get("heatmap_cmin_widget")
            cbar_max_val = st.session_state.get("heatmap_cmax_widget")

            # Only set range if both min and max are valid numbers and min < max
            if x_min_val is not None and x_max_val is not None and isinstance(x_min_val, (int, float)) and isinstance(x_max_val, (int, float)) and x_min_val < x_max_val:
                x_range_to_use = (x_min_val, x_max_val)
            if y_min_val is not None and y_max_val is not None and isinstance(y_min_val, (int, float)) and isinstance(y_max_val, (int, float)) and y_min_val < y_max_val:
                y_range_to_use = (y_min_val, y_max_val)

            # Set colorbar range (vmin/vmax handle None internally)
            c_range_to_use = (cbar_min_val, cbar_max_val)


        # Filter NaNs from selected parameter data
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_plot, y_plot = x_data[mask], y_data[mask]

        if len(x_plot) == 0:
            st.warning("No valid, non-NaN data points for the selected parameter combination.")
            return

        # --- Perform Plotting ---
        with st.spinner("Generating heatmap..."):
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                # hist2d arguments: x, y, bins, cmap, range (optional list [x_range, y_range]), vmin, vmax
                counts, xedges, yedges, img = ax.hist2d(
                    x_plot, y_plot,
                    bins=bin_count_to_use,
                    cmap=cmap_to_use,
                    range=[x_range_to_use, y_range_to_use] if x_range_to_use and y_range_to_use else None,
                    vmin=c_range_to_use[0], # Pass vmin
                    vmax=c_range_to_use[1]  # Pass vmax
                )

                cbar = fig.colorbar(img, ax=ax)
                cbar.set_label('Frequency')
                ax.set_title(title_to_use)
                ax.set_xlabel(x_param.replace('_', ' ').capitalize())
                ax.set_ylabel(y_param.replace('_', ' ').capitalize())

                # Explicitly set limits if a range was provided (hist2d might handle auto)
                if x_range_to_use:
                    ax.set_xlim(x_range_to_use)
                if y_range_to_use:
                    ax.set_ylim(y_range_to_use)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Release memory

            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
                logging.error(f"Heatmap generation failed: {traceback.format_exc()}")


def display_qtable_comparison(simulation_data, config):
    """Muestra la comparación de Q-Tables con lógica de indexación mejorada."""
    st.write("### Q-Table Evolution")

    # Find episodes that *claim* to have Q-tables
    episodes_with_qtables_info = {} # Store index and actual ep number
    for i, ep in enumerate(simulation_data):
        ep_num = ep.get('episode')
        qtables_data = ep.get('qtables')
        # Check if qtables exist, is a dict, and is not empty
        if ep_num is not None and isinstance(qtables_data, dict) and qtables_data:
            try:
                episodes_with_qtables_info[int(ep_num)] = i # Map ep_num (int) to list index
            except (ValueError, TypeError):
                 logging.warning(f"Non-integer episode number {ep_num} with Q-tables found, skipping.")

    if not episodes_with_qtables_info:
        st.info("No Q-table data found in the currently loaded episodes.")
        st.caption("Ensure `extract_qtables` was enabled during simulation and relevant episodes are loaded.")
        return

    available_ep_numbers = sorted(episodes_with_qtables_info.keys())

    col1, col2 = st.columns(2)
    with col1:
        ep1_num = st.selectbox("Select first episode:", available_ep_numbers, index=0, key="q_ep1")
    with col2:
        # Options for second episode must be strictly greater than the first
        valid_ep2_options = [ep for ep in available_ep_numbers if ep > ep1_num]
        if not valid_ep2_options:
            st.warning("No Q-table episodes available after the first selection.")
            return
        # Select the first valid option by default
        ep2_num = st.selectbox("Select second episode:", valid_ep2_options, index=0, key="q_ep2")

    # Get the list index corresponding to the selected episode numbers
    idx1 = episodes_with_qtables_info[ep1_num]
    idx2 = episodes_with_qtables_info[ep2_num]

    # Determine available gain types from the first selected episode's Q-table dict
    try:
        gain_types = list(simulation_data[idx1].get('qtables', {}).keys())
    except Exception as e:
         st.error(f"Error accessing Q-table keys for episode {ep1_num}: {e}")
         return

    if not gain_types:
        st.error(f"Q-table dictionary for episode {ep1_num} is empty or invalid.")
        return
    selected_gain = st.selectbox("Select gain type (K_p, K_i, K_d):", gain_types, key="q_gain")

    # --- Attempt to Load and Convert Q-Tables ---
    try:
        qtable1_raw = simulation_data[idx1]['qtables'][selected_gain]
        qtable2_raw = simulation_data[idx2]['qtables'][selected_gain]

        # Convert to DataFrame (handle potential nested lists/arrays)
        qtable1_df = pd.DataFrame(qtable1_raw)
        qtable2_df = pd.DataFrame(qtable2_raw)

    except KeyError:
        st.error(f"Selected gain '{selected_gain}' not found in Q-tables for episode {ep1_num} or {ep2_num}.")
        return
    except ValueError as ve:
         st.error(f"Error creating DataFrame from Q-table data for '{selected_gain}': {ve}")
         st.caption("Data might be ragged or have inconsistent dimensions.")
         st.write("Raw Q-table data (Ep 1):", qtable1_raw) # Show raw data for debugging
         return
    except Exception as e:
        st.error(f"Unexpected error loading/converting Q-tables for '{selected_gain}': {e}")
        return

    # --- Interpret State Configuration for Indexing ---
    state_names = []
    state_bins_info_str = [] # For display string
    index_levels = [] # List of ranges for MultiIndex
    num_state_vars = 0
    expected_rows = 1
    state_config_valid = False
    config_error_msg = ""

    try:
        # Navigate safely through potentially missing config keys
        state_cfg = config.get('environment', {}).get('agent', {}).get('params', {}).get('state_config', {})

        if not state_cfg:
            config_error_msg = "State configuration ('state_config') not found in loaded metadata."
        else:
            state_config_valid = True
            # Define the order of state variables as expected by the agent's q-table structure
            # IMPORTANT: This order MUST match the order used in the agent's `discretize` function
            ordered_state_vars = ['angle', 'angular_velocity', selected_gain]

            for var in ordered_state_vars:
                var_config = state_cfg.get(var)
                if isinstance(var_config, dict) and var_config.get('enabled', False):
                    bins = var_config.get('bins')
                    if bins and isinstance(bins, int) and bins > 0:
                        state_names.append(var)
                        state_bins_info_str.append(f"{var}({bins})")
                        index_levels.append(range(bins)) # Add range(0, bins)
                        expected_rows *= bins
                        num_state_vars += 1
                    else:
                        # Invalid config for this variable
                        config_error_msg = f"Config for state variable '{var}' is enabled but 'bins' is missing, invalid, or zero."
                        state_config_valid = False
                        break
                # If var is not in config or not enabled, it's skipped (correctly)

            if not state_names and state_config_valid:
                 config_error_msg = "State configuration found, but no state variables seem to be enabled."
                 state_config_valid = False

    except Exception as cfg_e:
        config_error_msg = f"Error processing state configuration from metadata: {cfg_e}"
        state_config_valid = False

    # --- Name Actions ---
    num_actions = qtable1_df.shape[1]
    action_names = [f"Action_{i}" for i in range(num_actions)]
    if num_actions == 3:
        action_names = ['Decrease', 'Keep', 'Increase'] # Specific names for 3 actions
    qtable1_df.columns = action_names
    qtable2_df.columns = action_names

    # --- Apply Index if Possible ---
    index_applied = False
    actual_rows = qtable1_df.shape[0]

    if state_config_valid:
        if expected_rows == actual_rows:
            if num_state_vars > 0:
                try:
                    if num_state_vars > 1:
                        multi_index = pd.MultiIndex.from_product(index_levels, names=state_names)
                        qtable1_df.index = multi_index
                        qtable2_df.index = multi_index
                    elif num_state_vars == 1: # Single index
                        single_index = pd.Index(index_levels[0], name=state_names[0])
                        qtable1_df.index = single_index
                        qtable2_df.index = single_index
                    index_applied = True
                    st.caption(f"State Index applied successfully: {', '.join(state_bins_info_str)}")
                except Exception as idx_e:
                    st.warning(f"Error applying calculated index names: {idx_e}. Using default integer index.")
            else:
                # Should not happen if state_config_valid is true and rows match > 0
                 st.warning("State config seems valid but no state variables found for indexing. Using default integer index.")
        else:
            # Row count mismatch is a critical sign of inconsistency
            st.error(f"Q-Table dimension mismatch for '{selected_gain}'!")
            st.warning(f"Expected rows based on config ({', '.join(state_bins_info_str)}): **{expected_rows}**")
            st.warning(f"Actual rows found in loaded Q-Table: **{actual_rows}**")
            st.caption("This indicates an inconsistency between the simulation configuration saved in metadata and the actual structure of the saved Q-table. Using default integer index.")
    elif config_error_msg:
        # Config was invalid or missing
        st.warning(f"Cannot determine state index names: {config_error_msg}. Using default integer index.")
    else:
         # Fallback if config wasn't processed correctly for other reasons
         st.warning("State configuration not available or invalid. Using default integer index.")


    # --- Display DataFrames ---
    st.write(f"Comparing Q-Table for gain: **{selected_gain.upper()}**")
    col11, col22 = st.columns(2)

    with col11:
        st.write(f"**Episode {ep1_num}:**")
        try:
            st.dataframe(qtable1_df.style.apply(resaltar_maximo, axis=1).format(precision=4))
        except Exception as df_disp_e:
            st.error(f"Error displaying Q-Table for Ep {ep1_num}: {df_disp_e}")
            st.dataframe(qtable1_df) # Fallback without style

    with col22:
        st.write(f"**Episode {ep2_num}:**")
        try:
            st.dataframe(qtable2_df.style.apply(resaltar_maximo, axis=1).format(precision=4))
        except Exception as df_disp_e:
            st.error(f"Error displaying Q-Table for Ep {ep2_num}: {df_disp_e}")
            st.dataframe(qtable2_df) # Fallback without style

    st.info("Green cells highlight the action with the highest Q-value for that state.")
    if num_actions == 3:
        st.caption("Assuming Actions are: 'Decrease' Gain, 'Keep' Gain, 'Increase' Gain.")


def episode_boxplots():
    """Genera boxplots para variables específicas a través de los episodios cargados."""
    st.subheader("Variable Distribution Across Loaded Episodes")

    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    data = loaded_data.get('simulation_data', [])

    if not data:
        st.warning("No episode data loaded. Please load data from the sidebar.")
        return

    # Identify numeric variables suitable for boxplots from the first episode
    numeric_vars_options = []
    if data:
        first_ep = data[0]
        for key, value in first_ep.items():
            # Check if it's a list, not empty, and contains numbers
            if isinstance(value, list) and value and isinstance(value[0], (int, float, np.number)):
                numeric_vars_options.append(key)
        # Exclude variables that are usually less interesting for distribution plots
        vars_to_exclude = {'time', 'episode', 'cumulative_reward', 'epsilon', 'learning_rate',
                           'action_kp', 'action_ki', 'action_kd'} # Keep kp, ki, kd themselves
        numeric_vars_options = sorted([v for v in list(set(numeric_vars_options)) if v not in vars_to_exclude])

    if not numeric_vars_options:
        st.warning("No suitable numeric variables found in loaded episodes for boxplots.")
        return

    # Sensible defaults - check if they exist in the options
    default_selections = [v for v in ['kp', 'ki', 'kd', 'pendulum_angle', 'pendulum_velocity', 'force', 'reward', 'error'] if v in numeric_vars_options]

    selected_variables = st.multiselect(
        "Select variables for Box Plot:",
        numeric_vars_options,
        default=default_selections,
        key="boxplot_vars"
    )

    if not selected_variables:
        st.info("Select one or more variables to generate boxplots.")
        return

    # Use @st.cache_data for the potentially expensive DataFrame preparation
    # The cache key depends on the tuple representation of the episode data and selected variables
    @st.cache_data
    def prepare_boxplot_data(episode_data_tuple: Tuple[Dict, ...], vars_to_plot: Tuple[str, ...]) -> pd.DataFrame:
        """Converts episode data list into a long-format DataFrame for box plotting."""
        plot_data_list = []
        logging.info(f"Preparing boxplot data for {len(vars_to_plot)} variables across {len(episode_data_tuple)} episodes...")
        episode_data_list = list(episode_data_tuple) # Convert back to list for iteration

        for episode in episode_data_list:
            ep_num = episode.get('episode', 'N/A')
            # Ensure episode number is treated as a string category for plotting if non-numeric
            ep_label = str(ep_num)

            for var in vars_to_plot:
                values = episode.get(var)
                if isinstance(values, list):
                    # Convert to numeric, coerce errors, drop NaNs
                    numeric_vals = pd.to_numeric(values, errors='coerce')
                    valid_vals = numeric_vals[~np.isnan(numeric_vals)]
                    if len(valid_vals) > 0:
                         # Append dictionary for each valid value
                         plot_data_list.extend([{'Episode': ep_label, 'Variable': var, 'Value': val} for val in valid_vals])

        if not plot_data_list:
            logging.warning("No valid data points found for selected variables in boxplot preparation.")
            return pd.DataFrame()

        logging.info(f"Finished preparing boxplot data. {len(plot_data_list)} data points.")
        return pd.DataFrame(plot_data_list)

    # Convert data and selected_variables to tuples for caching
    # Make sure internal dicts/lists within episode data are hashable if caching deeply.
    # Simple conversion to tuple of dicts should work if structure is standard JSON types.
    try:
        data_tuple = tuple(data) # Convert list of dicts to tuple of dicts
        vars_tuple = tuple(selected_variables)
        df_plot = prepare_boxplot_data(data_tuple, vars_tuple)
    except TypeError as e:
         st.error(f"Could not prepare data for caching (contains unhashable types?): {e}")
         # Fallback to non-cached version if caching fails
         df_plot = prepare_boxplot_data.__wrapped__(data, selected_variables) # Call original function

    if df_plot.empty:
        st.warning(f"No valid numeric data found for the selected variable(s) across the loaded episodes.")
        return

    # Sort Episodes numerically for consistent plotting order
    try:
        # Extract unique episode labels, convert to numeric, sort, convert back to string for category order
        unique_ep_labels = df_plot['Episode'].unique()
        sorted_ep_numeric = sorted([ep for ep in pd.to_numeric(unique_ep_labels, errors='coerce').dropna()])
        sorted_ep_str_labels = [str(int(ep)) if ep == int(ep) else str(ep) for ep in sorted_ep_numeric] # Handle potential floats if needed
        # Apply categorical type with the sorted order
        df_plot['Episode'] = pd.Categorical(df_plot['Episode'], categories=sorted_ep_str_labels, ordered=True)
    except Exception as e:
        st.warning(f"Could not sort episodes numerically for plot ordering: {e}. Using default order.")

    st.write("Generating boxplots...")
    try:
        # Create Plotly figure using facet_row for multiple variables
        fig = px.box(df_plot,
                     x='Episode',
                     y='Value',
                     color='Episode', # Color by episode
                     facet_row='Variable', # Create a subplot row for each variable
                     labels={'Value': '', 'Episode': 'Episode Number'}, # Clear Y label, set X
                     title="Distribution per Episode for Selected Variables")

        # Update layout for better readability
        fig.update_layout(
            showlegend=False, # Legend is redundant with x-axis labels and colors
            height=max(400, 200 * len(selected_variables)), # Adjust height based on number of variables
            margin=dict(l=60, r=30, t=60, b=50) # Adjust margins
        )
        # Ensure each facet Y-axis has a label and is independent
        fig.update_yaxes(matches=None, title='Value', showticklabels=True)
        # Clean up facet labels (remove 'Variable=')
        fig.for_each_annotation(lambda anno: anno.update(text=anno.text.split('=')[-1].replace('_',' ').capitalize()))
        # Rotate x-axis labels if there are many episodes
        if len(df_plot['Episode'].unique()) > 15:
            fig.update_xaxes(tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating boxplots: {e}")
        logging.error(f"Boxplot generation failed: {traceback.format_exc()}")


def episode_details():
    """Muestra detalles, gráficos y animación de un episodio específico con navegación mejorada."""
    st.subheader("Detailed Episode Analysis")

    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    loaded_sim_data = loaded_data.get('simulation_data', [])
    available_loaded_episode_numbers = st.session_state.get('available_loaded_episode_numbers', [])

    if not loaded_sim_data or not available_loaded_episode_numbers:
        st.warning("No episode data loaded or available. Please load data from the sidebar.")
        return

    # --- Episode Navigation in Sidebar ---
    st.sidebar.subheader("Select Episode for Details")

    min_ep, max_ep = min(available_loaded_episode_numbers), max(available_loaded_episode_numbers)
    current_ep_num_in_state = available_loaded_episode_numbers[st.session_state.current_episode_index]

    # Use number_input for direct entry and finding closest match
    target_episode_num = st.sidebar.number_input(
        f"Enter Episode Number ({min_ep}-{max_ep}):",
        min_value=min_ep,
        max_value=max_ep,
        value=current_ep_num_in_state, # Default to current selection
        step=1,
        key="detail_episode_input"
    )

    # Find the index corresponding to the target_episode_num
    new_selected_index = st.session_state.current_episode_index # Default to current
    try:
        if target_episode_num in available_loaded_episode_numbers:
            new_selected_index = available_loaded_episode_numbers.index(target_episode_num)
            if new_selected_index != st.session_state.current_episode_index:
                 st.session_state.current_episode_index = new_selected_index
                 st.rerun() # Rerun if exact match found and index changed
        else:
            # Find closest episode if exact number not available
            closest_episode_num = min(available_loaded_episode_numbers, key=lambda x: abs(x - target_episode_num))
            new_selected_index = available_loaded_episode_numbers.index(closest_episode_num)
            if new_selected_index != st.session_state.current_episode_index:
                 st.sidebar.info(f"Episode {target_episode_num} not loaded. Showing closest: {closest_episode_num}")
                 st.session_state.current_episode_index = new_selected_index
                 # Update the number input widget value to reflect the actual selection
                 st.session_state.detail_episode_input = closest_episode_num
                 st.rerun() # Rerun to reflect the change

    except ValueError: # Should not happen with number input range, but handle defensively
         st.sidebar.error("Invalid episode number.")
         return
    except Exception as nav_e:
         st.sidebar.error(f"Error finding episode: {nav_e}")
         return


    # Previous/Next Buttons
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("⬅️ Previous Episode", key="prev_ep_button", use_container_width=True):
        if st.session_state.current_episode_index > 0:
            st.session_state.current_episode_index -= 1
            # Update number input to match new selection
            st.session_state.detail_episode_input = available_loaded_episode_numbers[st.session_state.current_episode_index]
            st.rerun()
        else:
            st.sidebar.caption("Already at the first loaded episode.")

    if col_next.button("Next Episode ➡️", key="next_ep_button", use_container_width=True):
        if st.session_state.current_episode_index < len(available_loaded_episode_numbers) - 1:
            st.session_state.current_episode_index += 1
            # Update number input to match new selection
            st.session_state.detail_episode_input = available_loaded_episode_numbers[st.session_state.current_episode_index]
            st.rerun()
        else:
            st.sidebar.caption("Already at the last loaded episode.")

    # --- Display Selected Episode Details ---
    try:
        # Get data using the reliably updated current_episode_index
        selected_episode_data = loaded_sim_data[st.session_state.current_episode_index]
        actual_episode_number = selected_episode_data.get('episode', 'N/A') # Should match target or closest
        term_reason = selected_episode_data.get('termination_reason', 'N/A')
        time_list = selected_episode_data.get('time', [])
        reward_list = selected_episode_data.get('cumulative_reward', [])
        final_time = float(time_list[-1]) if time_list else np.nan
        final_reward = float(reward_list[-1]) if reward_list else np.nan

        st.write(f"#### Details for Episode: {actual_episode_number}")
        st.info(f"Termination: **{term_reason}** | Final Time: **{final_time:.3f}s** | Final Reward: **{final_reward:.3f}**")

        col1, col2 = st.columns(2)
        df_episode = None # Initialize df_episode

        with col1: # Data Table
            st.write("**Episode Data Table**")
            try:
                # Prepare data for DataFrame, handling potential list length inconsistencies
                df_data = {}
                max_len = 0
                # Find max length only among list-type values
                for k, v in selected_episode_data.items():
                    if isinstance(v, list):
                        max_len = max(max_len, len(v))

                if max_len > 0:
                    for k, v in selected_episode_data.items():
                        if k == 'qtables': continue # Skip Q-tables in this view

                        if isinstance(v, list):
                            current_len = len(v)
                            if current_len == max_len:
                                df_data[k] = v
                            elif current_len < max_len:
                                # Pad with NaN
                                df_data[k] = list(v) + [np.nan] * (max_len - current_len)
                            else: # current_len > max_len
                                # Truncate
                                df_data[k] = v[:max_len]
                        elif isinstance(v, (int, float, str, bool, np.number)):
                            # Repeat scalar values for max_len
                            df_data[k] = [v] * max_len
                        # Ignore other types (like dicts for Q-tables)

                    if df_data:
                        df_episode = pd.DataFrame(df_data)
                        # Define desired columns and order, checking existence
                        display_cols = ['time', 'pendulum_angle', 'pendulum_velocity', 'cart_position', 'cart_velocity', 'force', 'reward', 'cumulative_reward', 'kp', 'ki', 'kd', 'error', 'epsilon', 'learning_rate', 'action_kp', 'action_ki', 'action_kd', 'termination_reason']
                        existing_cols = [col for col in display_cols if col in df_episode.columns]
                        # Display only existing columns in the desired order
                        st.dataframe(df_episode[existing_cols], height=300)
                    else:
                        st.warning("No suitable data found to create episode table.")
                else:
                    st.warning("No time-series (list) data found in this episode.")
            except Exception as e:
                st.error(f"Error preparing or displaying episode data table: {e}")
                logging.error(f"Table display error: {traceback.format_exc()}")

        with col2: # Statistics
            st.write("**Summary Statistics**")
            if df_episode is not None and not df_episode.empty:
                 try:
                     # Select only numeric columns for describe()
                     numeric_cols_df = df_episode.select_dtypes(include=np.number)
                     if not numeric_cols_df.empty:
                         # Transpose for better readability and apply formatting
                         st.dataframe(numeric_cols_df.describe().T.style.format(precision=4))
                     else:
                         st.warning("No numeric data available in the table for statistics.")
                 except Exception as e:
                     st.error(f"Error calculating or displaying statistics: {e}")
                     logging.error(f"Stats display error: {traceback.format_exc()}")
            else:
                 st.warning("Data table not available or empty, cannot calculate statistics.")

        st.divider()
        st.write("**Interactive Plots**")
        plot_episode_details_graphs(selected_episode_data) # Renamed function

        st.divider()
        st.write("**Animation**")
        plot_episode_animation(selected_episode_data)

    except IndexError:
         st.error(f"Error accessing selected episode at index {st.session_state.current_episode_index}. Data might be inconsistent.")
    except Exception as detail_e:
         st.error(f"An unexpected error occurred displaying episode details: {detail_e}")
         logging.error(f"Episode detail display error: {traceback.format_exc()}")


def plot_episode_details_graphs(episode: dict): # Renamed function
    """Genera gráficos interactivos para un episodio usando Plotly."""
    plottable_vars = []
    ep_num = episode.get('episode', 'N/A')
    if episode:
         # Find keys with list values containing numbers
         for k, v in episode.items():
              if isinstance(v, list) and v and isinstance(v[0], (int, float, np.number)):
                   plottable_vars.append(k)
         plottable_vars = sorted(plottable_vars)

    if not plottable_vars:
        st.warning("No plottable time-series data found in this episode.")
        return

    # Use session state to remember selections for this tab
    if 'plot_selections' not in st.session_state:
        st.session_state.plot_selections = {'x_param': 'time', 'y_params': []}

    col1, col2 = st.columns([1, 2])
    with col1:
        # Default X to 'time' if available
        x_idx = plottable_vars.index('time') if 'time' in plottable_vars else 0
        # Ensure key is unique to avoid conflicts
        st.session_state.plot_selections['x_param'] = st.selectbox(
            "X-axis:",
            plottable_vars,
            index=x_idx,
            key=f'detail_plot_x_{ep_num}' # Key includes episode number
        )
    with col2:
        # Options for Y axis exclude the selected X axis
        y_options = [v for v in plottable_vars if v != st.session_state.plot_selections['x_param']]
        # Ensure default selections are valid and exist in current options
        valid_defaults = [p for p in st.session_state.plot_selections.get('y_params', []) if p in y_options]
        # Sensible defaults if none selected or invalid
        if not valid_defaults and y_options:
            valid_defaults = [p for p in ['pendulum_angle', 'cart_position', 'force', 'reward'] if p in y_options]

        st.session_state.plot_selections['y_params'] = st.multiselect(
            "Y-axis variables:",
            y_options,
            default=valid_defaults,
            key=f'detail_plot_y_{ep_num}' # Key includes episode number
        )

    x_param = st.session_state.plot_selections['x_param']
    y_params = st.session_state.plot_selections['y_params']

    if not x_param or not y_params:
        st.info("Select X and at least one Y axis variable to plot.")
        return

    x_data_raw = episode.get(x_param)

    # Validate X data
    if not isinstance(x_data_raw, list) or not x_data_raw:
        st.error(f"Selected X-axis data ('{x_param}') is missing or invalid in episode {ep_num}.")
        return
    # Convert X data to numeric, handle errors
    x_data_numeric = pd.to_numeric(x_data_raw, errors='coerce')


    # Plot each selected Y variable against X
    for y_param in y_params:
        y_data_raw = episode.get(y_param)

        # Validate Y data
        if not isinstance(y_data_raw, list) or not y_data_raw:
            st.warning(f"Selected Y-axis data ('{y_param}') is missing or invalid in episode {ep_num}. Skipping plot.")
            continue
        # Convert Y data to numeric, handle errors
        y_data_numeric = pd.to_numeric(y_data_raw, errors='coerce')

        # Align data: Find shortest length and handle NaNs introduced by conversion
        min_len = min(len(x_data_numeric), len(y_data_numeric))
        x_aligned = x_data_numeric[:min_len]
        y_aligned = y_data_numeric[:min_len]

        # Create mask for valid (non-NaN) points in both X and Y
        valid_mask = ~np.isnan(x_aligned) & ~np.isnan(y_aligned)
        x_final = x_aligned[valid_mask]
        y_final = y_aligned[valid_mask]

        if len(x_final) == 0:
            st.warning(f"No valid overlapping numeric data points found for '{y_param}' vs '{x_param}'. Skipping plot.")
            continue

        # Create Plotly figure
        try:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=x_final, y=y_final, mode='lines', name=f"{y_param} vs {x_param}"))

            # Improve titles and labels
            x_title = x_param.replace('_', ' ').capitalize()
            y_title = y_param.replace('_', ' ').capitalize()
            plot_title = f"Episode {ep_num}: {y_title} vs {x_title}"

            fig.update_layout(
                title=plot_title,
                xaxis_title=x_title,
                yaxis_title=y_title,
                margin=dict(l=50, r=20, t=50, b=40), # Adjust margins
                template="plotly_white" # Use a clean template
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to generate plot for '{y_param}' vs '{x_param}': {e}")
            logging.error(f"Plotly graph error: {traceback.format_exc()}")


def plot_episode_animation(episode: dict):
    """Genera o muestra la animación de un episodio usando PendulumAnimator."""
    episode_index_or_id = episode.get('episode', 'N/A') # Use actual episode number if available
    st.write(f"**Animation Controls (Episode {episode_index_or_id})**")

    if not st.session_state.selected_folder_path:
        st.warning("Cannot generate/find animation: Results folder path is not set.")
        return

    # Use a consistent key base for widgets related to this episode's animation
    anim_key_base = f"anim_ep_{episode_index_or_id}"

    show_animation = st.checkbox("Show/Generate Animation", value=False, key=f"{anim_key_base}_show")

    if not show_animation:
        st.caption("Enable the checkbox above to view or generate the animation.")
        return

    # Define animation filename and path
    animation_filename = f"animation_episode_{episode_index_or_id}.gif"
    animation_path = os.path.join(st.session_state.selected_folder_path, animation_filename)

    regenerate_ui = False # Flag to show generation UI

    # Check if animation file exists
    if os.path.exists(animation_path):
        st.success(f"Animation file found: `{animation_filename}`")
        try:
            st.image(animation_path, caption=f"Cached Animation for Episode {episode_index_or_id}")
            # Add button to allow regeneration even if found
            if st.button("Regenerate Animation", key=f"{anim_key_base}_regen_found"):
                regenerate_ui = True
                try:
                    os.remove(animation_path) # Remove old file before regenerating
                    st.info("Removed existing animation file. Click 'Generate' below.")
                except OSError as e:
                    st.error(f"Error removing existing animation file: {e}")
                    # Proceed to show generation UI anyway
        except Exception as e:
             st.error(f"Error displaying cached animation file: {e}. You might need to regenerate it.")
             regenerate_ui = True # Show generation UI if display fails
             if st.button("Attempt Regeneration", key=f"{anim_key_base}_regen_error"):
                 try:
                    if os.path.exists(animation_path): os.remove(animation_path)
                 except OSError as e_rem: st.error(f"Error removing corrupt animation file: {e_rem}")
                 # Continue to show generation UI
    else:
        # File not found, show generation UI
        regenerate_ui = True
        st.info(f"Animation file `{animation_filename}` not found.")

    # --- UI for Generation ---
    if regenerate_ui:
        st.write("Configure and Generate Animation:")
        cols_anim = st.columns(3)
        with cols_anim[0]:
            fps = st.slider("FPS (Frames Per Second)", 10, 60, 30, key=f"{anim_key_base}_fps")
            speed = st.slider("Playback Speed", 0.1, 5.0, 1.0, 0.1, format="%.1fx", key=f"{anim_key_base}_speed")
        with cols_anim[1]:
             # Calculate reasonable default X limits based on data
             cart_pos_data = episode.get('cart_position', [])
             valid_cart_pos = [x for x in cart_pos_data if isinstance(x, (int, float)) and np.isfinite(x)]
             max_abs_cart = max(abs(x) for x in valid_cart_pos) if valid_cart_pos else 3.0
             default_xlim = [-max(3.0, max_abs_cart * 1.1), max(3.0, max_abs_cart * 1.1)] # Add some margin
             x_lim_anim = st.slider("X-Axis Limits", -15.0, 15.0, (default_xlim[0], default_xlim[1]), 0.5, key=f"{anim_key_base}_xlim")
        with cols_anim[2]:
             # Default Y limits usually fine for pendulum
             default_ylim = [-3.0, 3.0]
             y_lim_anim = st.slider("Y-Axis Limits", -5.0, 5.0, (default_ylim[0], default_ylim[1]), 0.5, key=f"{anim_key_base}_ylim")

        if st.button("Generate Animation", key=f"{anim_key_base}_gen_button", type="primary"):
            # --- Validation before generation ---
            required_keys = ['time', 'cart_position', 'pendulum_angle']
            valid_data = True
            data_lengths = {}
            for key in required_keys:
                 data_list = episode.get(key)
                 if not isinstance(data_list, list) or not data_list:
                     st.error(f"Animation failed: Missing or empty required data for '{key}'.")
                     valid_data = False
                     break
                 data_lengths[key] = len(data_list)

            if valid_data and len(set(data_lengths.values())) > 1:
                 st.error(f"Animation failed: Inconsistent data lengths found: {data_lengths}")
                 valid_data = False

            if not valid_data:
                return # Stop if validation failed

            # --- Prepare config and Animator ---
            config_animation = {
                "fps": fps,
                "speed": speed,
                "x_lim": x_lim_anim,
                "y_lim": y_lim_anim,
                "dpi": 100 # Resolution for saving GIF
            }

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            status_text.text("Initializing animation...")
            animator = None # Initialize animator outside try block for finally clause

            try:
                # Initialize the animator (creates figure)
                animator = PendulumAnimator(episode, config_animation)
                status_text.text("Creating animation frames...")
                # Create the animation object
                anim = animator.create_animation()

                if anim:
                    status_text.text(f"Saving animation to {animation_filename}...")
                    # Define a callback for progress update
                    def progress_callback(current_frame, total_frames):
                        progress = (current_frame + 1) / total_frames
                        progress_bar.progress(progress)
                        # Avoid overwhelming Streamlit with text updates
                        # if current_frame % 10 == 0 or current_frame == total_frames - 1:
                        #    status_text.text(f"Saving frame {current_frame + 1}/{total_frames}...")

                    # Save the animation using pillow writer for GIF
                    anim.save(
                        animation_path,
                        writer='pillow',
                        fps=config_animation['fps'],
                        progress_callback=progress_callback # Pass the callback
                    )

                    progress_bar.progress(1.0) # Ensure it reaches 100%
                    status_text.success(f"Animation saved successfully!")
                    st.image(animation_path, caption=f"Generated Animation for Episode {episode_index_or_id}") # Display after saving
                else:
                    status_text.error("Animation creation failed (returned None).")
                    progress_bar.empty()

            except Exception as e:
                status_text.error(f"Failed to generate or save animation: {e}")
                progress_bar.empty()
                logging.error(f"Animation generation/saving error: {traceback.format_exc()}")
            finally:
                 # IMPORTANT: Close the matplotlib figure to release memory
                 if animator and animator.fig:
                     plt.close(animator.fig)
                 gc.collect() # Explicit garbage collection after potentially large operation


# --------------------------- Función Principal ---------------------------
def main():
    # Check util imports first
    if not utils_imports_ok:
         st.error("Dashboard cannot start due to import errors from app_utils.py. Check logs.")
         st.stop() # Stop execution if utils are missing

    st.set_page_config(page_title="RL Control Dashboard", page_icon="🤖", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #999;'><b>Dynamic System Simulation Dashboard</b></h1>", unsafe_allow_html=True) # Adjusted color

    initialize_session_state() # Ensure state is initialized on first run/refresh
    render_sidebar() # Render sidebar and handle folder/data loading logic

    # Define tabs/pages for the main area
    tabs = ["Introduction & Config", "Performance Overview", "Episode Boxplots", "Episode Details"]
    icons = ['house-door', 'bar-chart-line', 'box-seam', 'search'] # Changed last icon

    # Create horizontal menu using streamlit-option-menu
    selected_page = option_menu(
        menu_title=None, # No main title for the menu itself
        options=tabs,
        icons=icons,
        menu_icon='cast', # A different icon for the menu
        default_index=0,
        orientation='horizontal',
        styles={ # Optional styling customization
            "container": {"padding": "5px !important", "background-color": "#f0f2f6", "border-radius": "5px"},
            "icon": {"color": "#023047", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin":"0px 5px", # Add some spacing between items
                "--hover-color": "#dfe6e9", # Lighter hover color
                "color": "#023047", # Darker text color
                "border-radius": "5px",
                "padding": "10px 15px" # Adjust padding
            },
            "nav-link-selected": {
                "background-color": "#219ebc", # Selected background color
                "color": "white", # Selected text color
                "font-weight": "bold",
            },
        }
    )

    # Display content based on selected page, only if a folder is selected
    if st.session_state.selected_folder_path:
        if selected_page == "Introduction & Config":
            introduction_page()
        elif selected_page == "Performance Overview":
            performance_overview()
        elif selected_page == "Episode Boxplots":
            episode_boxplots()
        elif selected_page == "Episode Details":
            episode_details()
    else:
         # Show a welcome message if no folder is selected yet
         st.info("👋 Welcome! Please select a simulation results folder from the sidebar to begin exploring.")

if __name__ == "__main__":
    main()