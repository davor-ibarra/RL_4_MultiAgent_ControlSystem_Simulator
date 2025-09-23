"""
Streamlit Dashboard for Dynamic System Simulation Results.
Version: 2.3.3 (Refactored for Performance and Maintainability)

Focuses on visualization and analysis using Matplotlib.
Implements on-demand plotting and data processing triggered by user actions.
Correctly loads and displays agent state from the new JSON format.
Utilizes app_utils.py for data loading and preparation helpers.
"""

import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker # For MaxNLocator
import matplotlib.animation as animation
import traceback
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import io # Needed for plot download
import gc # Garbage collector
from datetime import datetime
import glob # For finding agent state files

# Import utility functions
try:
    from app_utils import (
        load_folder_structure,
        load_metadata,
        load_summary_data,
        load_selected_episodes,
        load_agent_state_file,
        extract_trajectory_data, # Helper for single trajectory
        align_trajectory_data,   # Helper for alignment
        extract_heatmap_xy_data, # Specific for heatmap data prep
        resaltar_maximo,
        PendulumAnimator
    )
    utils_imports_ok = True
except ImportError as e:
    # Display error prominently in Streamlit if imports fail
    st.error(f"FATAL ERROR: Failed to import utilities from app_utils.py: {e}.\nPlease ensure app_utils.py is in the correct location and has no errors.")
    logging.critical(f"app_utils import failed: {e}", exc_info=True)
    utils_imports_ok = False
    # We might want to stop the app here if utils are critical
    # st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# == Constants and Session State Initialization ==
# =============================================================================
RESULTS_FOLDER = "results_history" # Base folder for simulation results

# --- Key Prefixes for Session State Caching ---
# Caching processed data to avoid recalculation unless necessary
STATE_PREFIX_PLOT_DATA = "plot_data_cache_"
# Caching plot settings to retain user customization
STATE_PREFIX_PLOT_SETTINGS = "plot_settings_cache_"

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    # Store essential state variables
    defaults = {
        'folders': [],                      # List of available result folders
        'selected_folder_path': None,       # Full path to the currently selected folder
        'selected_folder_name': None,       # Name of the currently selected folder
        'metadata': None,                   # Loaded metadata dictionary
        'config': {},                       # Config parameters extracted from metadata
        'summary_df': None,                 # Loaded summary data as DataFrame
        'available_episodes_in_summary': [],# List of episode numbers found in summary_df
        'selected_episode_numbers_to_load': [],# Episodes selected by user filters
        'loaded_episode_data': {'simulation_data': []}, # Raw data for loaded trajectories
        'available_loaded_episode_numbers': [],# Episode numbers actually present in loaded_episode_data
        'current_episode_index': 0,         # Index for navigating loaded episodes
        'available_agent_state_episodes': [],# List of episode numbers for which agent state files exist
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            logging.debug(f"Initialized session state key: {key}")

def clear_plot_cache_state():
    """Removes plot-specific data and settings caches from session state."""
    keys_to_delete = [
        k for k in st.session_state
        if k.startswith(STATE_PREFIX_PLOT_DATA) or k.startswith(STATE_PREFIX_PLOT_SETTINGS)
    ]
    deleted_count = len(keys_to_delete)
    for key in keys_to_delete:
        del st.session_state[key]
    if deleted_count > 0:
        logging.info(f"Cleared {deleted_count} plot cache entries from session state.")
    gc.collect() # Trigger garbage collection

def reset_all_state():
    """Resets the entire session state."""
    logging.info("Resetting all session state.")
    st.cache_data.clear() # Clear Streamlit's data cache
    keys_to_reset = list(st.session_state.keys())
    for key in keys_to_reset:
        del st.session_state[key]
    initialize_session_state() # Re-initialize with defaults
    gc.collect()
    st.sidebar.success("Cache & state reset.")
    st.rerun() # Rerun the app immediately


# =============================================================================
# == Sidebar Rendering and Data Loading Triggers ==
# =============================================================================

@st.cache_data # Cache finding agent states per folder
def find_available_agent_state_episodes(folder_path: str) -> List[int]:
    """Scans the folder for agent_state_episode_*.json files and returns sorted episode numbers."""
    if not folder_path or not os.path.isdir(folder_path):
        logging.debug(f"Cannot find agent states: Invalid folder path '{folder_path}'")
        return []
    pattern = os.path.join(folder_path, 'agent_state_episode_*.json')
    files = glob.glob(pattern)
    episode_numbers = []
    for f in files:
        try:
            # Extract number between last '_' and '.json'
            base = os.path.basename(f)
            num_str = base.split('_')[-1].replace('.json', '')
            episode_numbers.append(int(num_str))
        except (ValueError, IndexError, TypeError):
            logging.warning(f"Could not parse episode number from agent state filename: {f}")
    if episode_numbers:
        logging.info(f"Found {len(episode_numbers)} agent state files in {os.path.basename(folder_path)}.")
    return sorted(episode_numbers)

def render_sidebar():
    """Renders the sidebar and handles folder selection, filtering, and data loading triggers."""
    st.sidebar.title("Simulation Results")

    # --- Refresh Button ---
    if st.sidebar.button("🔄 Refresh Folders & Clear Cache", key="refresh_folders"):
        reset_all_state() # Resets everything and reruns

    # --- Folder Selection ---
    if not st.session_state.get('folders'):
        st.session_state.folders = load_folder_structure(RESULTS_FOLDER)
        # If folders still empty after loading, show warning
        if not st.session_state.folders:
            st.sidebar.warning(f"No simulation folders found in '{RESULTS_FOLDER}'.")
            return # Stop sidebar rendering if no folders

    current_folder_name = st.session_state.get('selected_folder_name')
    current_selection_index = 0
    if current_folder_name and current_folder_name in st.session_state.folders:
        try:
            current_selection_index = st.session_state.folders.index(current_folder_name)
        except ValueError: # Handle case where selected folder disappeared
            st.session_state.selected_folder_name = None
            st.session_state.selected_folder_path = None

    selected_folder_name_new = st.sidebar.selectbox(
        "Select results folder:",
        st.session_state.folders,
        index=current_selection_index,
        key="folder_selector"
    )

    if not selected_folder_name_new:
        st.sidebar.warning("Please select a results folder.")
        return # Stop if no folder selected

    selected_folder_path_new = os.path.join(RESULTS_FOLDER, selected_folder_name_new)

    # --- Load Data on Folder Change ---
    if selected_folder_path_new != st.session_state.get('selected_folder_path'):
        st.sidebar.info(f"Loading folder: {selected_folder_name_new}")
        # Reset state related to the *previous* folder
        keys_to_reset_on_folder_change = [
            'metadata', 'config', 'summary_df', 'loaded_episode_data',
            'available_episodes_in_summary', 'selected_episode_numbers_to_load',
            'available_loaded_episode_numbers', 'current_episode_index',
            'available_agent_state_episodes'
        ]
        for key in keys_to_reset_on_folder_change:
            if key in st.session_state:
                # Re-initialize to default value from initialize_session_state logic
                initialize_session_state() # Ensure defaults are available
                default_val = st.session_state.get(key) # Get the default value again
                st.session_state[key] = default_val

        st.session_state.selected_folder_path = selected_folder_path_new
        st.session_state.selected_folder_name = selected_folder_name_new
        clear_plot_cache_state() # Clear plot-specific caches

        # Load essential data for the new folder
        with st.spinner("Loading metadata, summary & agent states..."):
            st.session_state.metadata = load_metadata(st.session_state.selected_folder_path)
            st.session_state.config = st.session_state.metadata.get('config_parameters', {}) if st.session_state.metadata else {}
            if not st.session_state.metadata: st.sidebar.warning("Metadata file missing or invalid.")
            if not st.session_state.config and st.session_state.metadata: st.sidebar.warning("Config parameters missing in metadata.")

            st.session_state.summary_df = load_summary_data(st.session_state.selected_folder_path)
            if st.session_state.summary_df is not None and 'episode' in st.session_state.summary_df.columns:
                # Extract available episodes from the summary
                unique_eps = st.session_state.summary_df['episode'].dropna().unique()
                st.session_state.available_episodes_in_summary = sorted([int(e) for e in unique_eps if pd.notna(e)])
                if not st.session_state.available_episodes_in_summary:
                    st.sidebar.warning("No valid episode numbers found in the summary file.")
            elif st.session_state.selected_folder_path: # Only warn if folder is selected but summary failed
                st.sidebar.warning("Summary file missing or invalid.")
                st.session_state.available_episodes_in_summary = []

            # Find available agent state files
            st.session_state.available_agent_state_episodes = find_available_agent_state_episodes(st.session_state.selected_folder_path)
            st.sidebar.caption(f"Found {len(st.session_state.available_agent_state_episodes)} agent state file(s).")

        gc.collect()
        st.rerun() # Rerun to reflect the new folder's data

    # --- Episode Filtering (Based on Summary) ---
    st.sidebar.subheader("Filter Episodes Before Loading Trajectories")
    summary_df = st.session_state.get('summary_df')
    available_summary_eps = st.session_state.get('available_episodes_in_summary', [])

    if summary_df is not None and not summary_df.empty and available_summary_eps:
        # Make a copy for filtering to avoid modifying the cached original
        filtered_summary_df = summary_df[summary_df['episode'].isin(available_summary_eps)].copy()
        num_avail = len(available_summary_eps)

        # Option to load all or filter
        load_all = st.sidebar.checkbox(f"Load all {num_avail} summarized episodes", value=False, key="load_all_check")

        if not load_all:
            st.sidebar.write("Apply filters (based on summary data):")
            min_ep, max_ep = min(available_summary_eps), max(available_summary_eps)

            # 1. Episode Range Filter
            if min_ep < max_ep:
                options_list = list(range(min_ep, max_ep + 1))
                default_range_val = (min_ep, max_ep)
                selected_ep_range = st.sidebar.select_slider(
                    "Episode range:",
                    options=options_list,
                    value=default_range_val,
                    key="filter_ep_range"
                )
                filtered_summary_df = filtered_summary_df[
                    (filtered_summary_df['episode'] >= selected_ep_range[0]) &
                    (filtered_summary_df['episode'] <= selected_ep_range[1])
                ]
            elif min_ep == max_ep:
                 st.sidebar.text(f"Episode: {min_ep}") # Only one episode

            # 2. Numeric Variable Filter
            numeric_cols = filtered_summary_df.select_dtypes(include=np.number).columns.tolist()
            # Exclude 'episode' and std/min/max columns from filter options
            essential_metrics = ['total_reward', 'episode_time', 'performance', 'final_epsilon', 'final_learning_rate']
            filter_options = sorted(
                [c for c in numeric_cols if c != 'episode' and not any(c.endswith(suffix) for suffix in ['_std', '_min', '_max'])],
                key=lambda x: (x not in essential_metrics, x) # Prioritize essential metrics
            )

            if filter_options:
                default_var = 'total_reward' if 'total_reward' in filter_options else filter_options[0]
                filter_var = st.sidebar.selectbox("Filter by variable:", filter_options, index=filter_options.index(default_var), key="filter_var_select")

                if filter_var and filter_var in filtered_summary_df.columns and not filtered_summary_df[filter_var].isnull().all():
                    try:
                        # Convert to numeric explicitly for min/max
                        numeric_series = pd.to_numeric(filtered_summary_df[filter_var], errors='coerce')
                        min_v, max_v = numeric_series.min(), numeric_series.max()

                        if pd.notna(min_v) and pd.notna(max_v):
                            if min_v < max_v:
                                range_v = max_v - min_v
                                # Dynamic step and format based on range
                                step_v = max(range_v / 100, 1e-6) if range_v > 0 else 1e-6
                                fmt = "%.3g" if range_v > 0.01 or range_v == 0 else "%.5f"
                                sel_range = st.sidebar.slider(
                                    f"Range for {filter_var}:",
                                    min_value=float(min_v), max_value=float(max_v),
                                    value=(float(min_v), float(max_v)),
                                    step=float(step_v), format=fmt,
                                    key=f"val_range_{filter_var}"
                                )
                                # Filter based on the slider range
                                filtered_summary_df = filtered_summary_df[
                                    (pd.to_numeric(filtered_summary_df[filter_var], errors='coerce') >= sel_range[0]) &
                                    (pd.to_numeric(filtered_summary_df[filter_var], errors='coerce') <= sel_range[1])
                                ]
                            elif min_v == max_v:
                                st.sidebar.text(f"{filter_var}: {min_v:.4g} (Constant value)")
                        else:
                            st.sidebar.warning(f"Variable '{filter_var}' contains only NaNs or non-numeric data, cannot filter.")
                    except Exception as e:
                        st.sidebar.error(f"Error creating filter slider for '{filter_var}': {e}")
                        logging.error(f"Filter slider error: {e}", exc_info=True)

            # 3. Termination Reason Filter
            if 'termination_reason' in filtered_summary_df.columns:
                reasons = sorted(filtered_summary_df['termination_reason'].dropna().unique().tolist())
                if reasons:
                    sel_reasons = st.sidebar.multiselect(
                        "Termination reason:",
                        options=reasons,
                        default=reasons, # Select all by default
                        key="filter_term"
                    )
                    # Filter only if the selection differs from the default (all)
                    if set(sel_reasons) != set(reasons):
                        filtered_summary_df = filtered_summary_df[filtered_summary_df['termination_reason'].isin(sel_reasons)]

            # Store the final list of episodes to load
            st.session_state.selected_episode_numbers_to_load = sorted(filtered_summary_df['episode'].astype(int).tolist()) if not filtered_summary_df.empty else []

        else: # Load all is checked
            st.session_state.selected_episode_numbers_to_load = available_summary_eps

        # Display number of episodes matching filters
        num_selected = len(st.session_state.selected_episode_numbers_to_load)
        if num_selected > 0:
            st.sidebar.info(f"{num_selected} trajectories match filters.")
            # Expander to show selected episodes (limited display)
            show_limit = 50
            with st.sidebar.expander(f"Show first {min(num_selected, show_limit)} selected episodes", expanded=False):
                st.write(st.session_state.selected_episode_numbers_to_load[:show_limit])
                if num_selected > show_limit: st.caption(f"... and {num_selected - show_limit} more.")
        else:
            st.sidebar.warning("No trajectories match the current filters.")

    elif st.session_state.selected_folder_path: # Folder selected, but summary is bad/missing
        st.sidebar.warning("Summary data unavailable for filtering.")
        st.session_state.selected_episode_numbers_to_load = []

    # --- Load Trajectory Data Button ---
    selected_eps_list = st.session_state.get('selected_episode_numbers_to_load', [])
    load_button_disabled = not isinstance(selected_eps_list, list) or not selected_eps_list

    if st.sidebar.button("Load Selected Trajectory Data", key="load_data_button", disabled=load_button_disabled, type="primary"):
        if st.session_state.selected_folder_path and selected_eps_list:
            # Convert list to tuple for caching in load_selected_episodes
            episodes_tuple_to_load = tuple(sorted(selected_eps_list))
            with st.spinner(f"Loading trajectory data for {len(episodes_tuple_to_load)} episode(s)..."):
                # Clear previous loaded data and plot caches before loading new data
                clear_plot_cache_state()
                st.session_state.loaded_episode_data = {'simulation_data': []}
                st.session_state.available_loaded_episode_numbers = []
                st.session_state.current_episode_index = 0
                gc.collect()

                # Call the cached loading function
                loaded_data = load_selected_episodes(st.session_state.selected_folder_path, episodes_tuple_to_load)

                st.session_state.loaded_episode_data = loaded_data if loaded_data else {'simulation_data': []}

                # Update the list of actually available loaded episodes
                if st.session_state.loaded_episode_data.get('simulation_data'):
                    sim_data = st.session_state.loaded_episode_data['simulation_data']
                    valid_nums = []
                    for ep in sim_data:
                        try: valid_nums.append(int(ep.get('episode', -1)))
                        except (ValueError, TypeError): pass
                    st.session_state.available_loaded_episode_numbers = sorted([num for num in list(set(valid_nums)) if num >= 0])
                    logging.info(f"Successfully loaded {len(st.session_state.available_loaded_episode_numbers)} episodes into state.")
                else:
                    st.session_state.available_loaded_episode_numbers = []
                    logging.warning("No simulation data loaded into state after call.")

                st.rerun() # Rerun to update the UI based on newly loaded data
        elif not st.session_state.selected_folder_path:
            st.sidebar.error("Select a results folder first.")
        else:
            st.sidebar.error("No episodes selected to load.") # Should be prevented by disabled state, but good failsafe

    st.sidebar.markdown("---") # Visual separator

# =============================================================================
# == Plotting and UI Component Rendering ==
# =============================================================================

def render_mpl_plot_customization(plot_key_base: str, defaults: Dict, plot_type: str = 'line') -> Dict:
    """
    Renders Matplotlib plot customization widgets within an expander.

    Args:
        plot_key_base: A unique string prefix for widget keys for this plot.
        defaults: A dictionary containing default values for the settings.
        plot_type: Type of plot ('line', 'bar', 'heatmap') to enable/disable relevant options.

    Returns:
        A dictionary containing the current values of the customization settings.
    """
    # Get current settings from defaults (will be updated by widgets)
    current_settings = defaults.copy()
    is_heatmap = plot_type == 'heatmap'
    is_bar = plot_type == 'bar'
    is_line = plot_type == 'line' # Add other types like boxplot if needed

    with st.expander("⚙️ Plot Settings"):
        # Use columns for better layout
        cols_layout = [1, 1, 1] # General | Axes | Ticks/Grid/Legend
        if is_heatmap: cols_layout = [1, 1, 1] # General | Axes | Ticks/Grid / Color
        elif is_bar: cols_layout = [1, 1, 1] # General | Axes | Ticks/Grid/Legend / Style
        else: cols_layout = [1, 1, 1] # Line uses this layout

        cols1 = st.columns(cols_layout)

        # --- Column 1: General & Figure ---
        with cols1[0]:
            st.markdown("**General**")
            current_settings['title'] = st.text_input(
                "Title", value=defaults.get('title', ''), key=f"title_{plot_key_base}"
            )
            current_settings['title_fontsize'] = st.slider(
                "Title Fontsize", 8, 24, defaults.get('title_fontsize', 14), key=f"fs_title_{plot_key_base}"
            )
            st.markdown("**Figure Size**")
            col_w, col_h = st.columns(2)
            def_fw = defaults.get('figsize_w', 10.0)
            def_fh = defaults.get('figsize_h', 6.0)
            current_settings['figsize_w'] = col_w.slider(
                "Width", 4.0, 20.0, float(def_fw), 0.5, key=f"fsw_{plot_key_base}", help="Figure width in inches"
            )
            current_settings['figsize_h'] = col_h.slider(
                "Height", 3.0, 15.0, float(def_fh), 0.5, key=f"fsh_{plot_key_base}", help="Figure height in inches"
            )
            # Combine into figsize tuple later

        # --- Column 2: Axes & Labels ---
        with cols1[1]:
            st.markdown("**Axes**")
            current_settings['xlabel'] = st.text_input(
                "X-Label", value=defaults.get('xlabel', ''), key=f"xlabel_{plot_key_base}"
            )
            current_settings['ylabel'] = st.text_input(
                "Y-Label", value=defaults.get('ylabel', ''), key=f"ylabel_{plot_key_base}"
            )
            current_settings['xlabel_fontsize'] = st.slider(
                "X-Label Fontsize", 8, 18, defaults.get('xlabel_fontsize', 12), key=f"fs_xlabel_{plot_key_base}"
            )
            current_settings['ylabel_fontsize'] = st.slider(
                "Y-Label Fontsize", 8, 18, defaults.get('ylabel_fontsize', 12), key=f"fs_ylabel_{plot_key_base}"
            )

        # --- Column 3: Ticks, Grid, Legend / Colorbar ---
        with cols1[2]:
            st.markdown("**Ticks & Grid**")
            current_settings['tick_fontsize'] = st.slider(
                "Tick Fontsize", 6, 16, defaults.get('tick_fontsize', 10), key=f"fs_tick_{plot_key_base}"
            )
            # X-tick rotation (useful for episode numbers)
            current_settings['xtick_rotation'] = st.slider(
                "X-Tick Rotation", 0, 90, defaults.get('xtick_rotation', 0), 15, key=f"xtick_rot_{plot_key_base}",
                disabled=is_heatmap # Rotation less common for heatmap axes
            )
            # Grid visibility
            current_settings['grid_on'] = st.checkbox(
                "Show Grid", value=defaults.get('grid_on', True), key=f"grid_{plot_key_base}"
            )
            # Approx number of ticks (using MaxNLocator)
            current_settings['num_xticks'] = st.slider(
                 "Approx. X-Ticks", 3, 30, defaults.get('num_xticks', 8), key=f"numxticks_{plot_key_base}",
                 disabled=(is_heatmap or is_bar) # Primarily for line plots
            )
            current_settings['num_yticks'] = st.slider(
                 "Approx. Y-Ticks", 3, 30, defaults.get('num_yticks', 6), key=f"numyticks_{plot_key_base}",
                 disabled=is_bar # Less useful for bar charts
            )

            if not is_heatmap:
                st.markdown("**Legend**")
                legend_enabled = is_bar or is_line # Enable legend for bar and line plots
                current_settings['show_legend'] = st.checkbox(
                    "Show Legend", value=defaults.get('show_legend', False), key=f"legend_{plot_key_base}",
                    disabled=not legend_enabled
                )
                current_settings['legend_pos'] = st.selectbox(
                    "Legend Position",
                    ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'outside'],
                    index=['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center', 'outside'].index(defaults.get('legend_pos','best')),
                    key=f"legendpos_{plot_key_base}",
                    disabled=(not current_settings['show_legend'] or not legend_enabled)
                )
                current_settings['legend_fontsize'] = st.select_slider(
                     "Legend Fontsize", options=['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'],
                     value=defaults.get('legend_fontsize', 'small'), key=f"legendfs_{plot_key_base}",
                     disabled=(not current_settings['show_legend'] or not legend_enabled)
                )
            else: # Heatmap Colorbar settings
                st.markdown("**Colorbar**")
                current_settings['clabel'] = st.text_input("Colorbar Label", value=defaults.get('clabel', 'Frequency'), key=f"clabel_{plot_key_base}")
                current_settings['clabel_fontsize'] = st.slider("Cbar Label FS", 8, 18, defaults.get('clabel_fontsize', 10), key=f"fs_clabel_{plot_key_base}")


        st.divider() # Separator before plot-type specific settings

        # --- Plot-Type Specific Settings ---
        cols2 = st.columns(3)

        # --- Column 1: Line/Marker (for Line plots) ---
        with cols2[0]:
            st.markdown("**Line Style**")
            current_settings['line_color'] = st.color_picker(
                "Line Color", value=defaults.get('line_color', "#1f77b4"), key=f"lcolor_{plot_key_base}",
                disabled=not is_line
            )
            current_settings['line_width'] = st.slider(
                "Line Width", 0.0, 5.0, defaults.get('line_width', 1.5), 0.1, key=f"lw_{plot_key_base}",
                disabled=not is_line
            )
            line_styles = ['-', '--', '-.', ':', 'None', ' ', ''] # Matplotlib styles
            default_ls = defaults.get('line_style', '-')
            ls_idx = line_styles.index(default_ls) if default_ls in line_styles else 0
            current_settings['line_style'] = st.selectbox("Line Style", line_styles, index=ls_idx, key=f"lstyle_{plot_key_base}", disabled=not is_line)


            st.markdown("**Marker Style**")
            current_settings['marker_size'] = st.slider(
                "Marker Size", 0, 15, defaults.get('marker_size', 0), 1, key=f"msize_{plot_key_base}",
                disabled=not is_line
            )
            # Common marker styles
            marker_styles_options = ['', '.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
            default_mstyle = defaults.get('marker_style', '')
            mstyle_idx = marker_styles_options.index(default_mstyle) if default_mstyle in marker_styles_options else 0
            current_settings['marker_style'] = st.selectbox(
                "Marker Style", marker_styles_options, index=mstyle_idx, key=f"mstyle_{plot_key_base}",
                disabled=(not is_line or current_settings['marker_size'] == 0)
            )
            current_settings['marker_color'] = st.color_picker(
                "Marker Color", value=defaults.get('marker_color', "#ff7f0e"), key=f"mcolor_{plot_key_base}",
                disabled=(not is_line or current_settings['marker_size'] == 0)
            )

        # --- Column 2: Bar/Heatmap Style ---
        with cols2[1]:
            st.markdown("**Bar Style**")
            current_settings['bar_width'] = st.slider(
                "Bar Width", 0.1, 1.0, defaults.get('bar_width', 0.8), 0.1, key=f"width_{plot_key_base}",
                disabled=not is_bar
            )
            # Colormap selection (useful for multi-bar or heatmaps)
            try: cmaps = plt.colormaps()
            except AttributeError: cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm', 'tab10', 'tab20', 'Set3', 'Pastel1'] # Fallback list
            if is_bar: default_cmap_val = defaults.get('cmap', 'tab20')
            elif is_heatmap: default_cmap_val = defaults.get('cmap', 'viridis')
            else: default_cmap_val = 'viridis' # Default fallback
            cmap_idx = cmaps.index(default_cmap_val) if default_cmap_val in cmaps else 0
            current_settings['cmap'] = st.selectbox(
                "Colormap (Bar/Heatmap)", cmaps, index=cmap_idx, key=f"cmap_{plot_key_base}",
                disabled=not (is_bar or is_heatmap)
            )

            st.markdown("**Heatmap Style**")
            current_settings['heatmap_log_scale'] = st.checkbox(
                "Log Color Scale", value=defaults.get('heatmap_log_scale', False), key=f"heatmap_log_{plot_key_base}",
                disabled=not is_heatmap
            )
            current_settings['heatmap_bins'] = st.slider(
                "Heatmap Bins", 10, 500, defaults.get('heatmap_bins', 100), 10, key=f"heatmap_bins_{plot_key_base}",
                disabled=not is_heatmap
            )

        # --- Column 3: Heatmap Ranges ---
        with cols2[2]:
            st.markdown("**Heatmap Ranges (Optional)**")
            help_txt = "Leave blank for auto-range"
            current_settings['xmin'] = st.number_input(f"X-Min", value=defaults.get('xmin'), format="%g", key=f"heatmap_xmin_{plot_key_base}", help=help_txt, disabled=not is_heatmap)
            current_settings['xmax'] = st.number_input(f"X-Max", value=defaults.get('xmax'), format="%g", key=f"heatmap_xmax_{plot_key_base}", help=help_txt, disabled=not is_heatmap)
            current_settings['ymin'] = st.number_input(f"Y-Min", value=defaults.get('ymin'), format="%g", key=f"heatmap_ymin_{plot_key_base}", help=help_txt, disabled=not is_heatmap)
            current_settings['ymax'] = st.number_input(f"Y-Max", value=defaults.get('ymax'), format="%g", key=f"heatmap_ymax_{plot_key_base}", help=help_txt, disabled=not is_heatmap)
            current_settings['cmin'] = st.number_input(f"Colorbar Min", value=defaults.get('cmin'), format="%g", key=f"heatmap_cmin_{plot_key_base}", help=help_txt, disabled=not is_heatmap)
            current_settings['cmax'] = st.number_input(f"Colorbar Max", value=defaults.get('cmax'), format="%g", key=f"heatmap_cmax_{plot_key_base}", help=help_txt, disabled=not is_heatmap)

    # --- Post-processing Settings ---
    # Combine width/height into figsize tuple
    try: current_settings['figsize'] = (float(current_settings['figsize_w']), float(current_settings['figsize_h']))
    except Exception: current_settings['figsize'] = (10.0, 6.0) # Fallback

    # Ensure marker style is off if size is 0
    if current_settings.get('marker_size', 0) == 0: current_settings['marker_style'] = ''
    # Ensure line is minimally visible if both width and marker size are 0
    if current_settings.get('line_width', 1.5) == 0 and current_settings.get('marker_size', 0) == 0: current_settings['line_width'] = 0.1

    return current_settings


def _apply_common_mpl_styles(ax: plt.Axes, fig: plt.Figure, settings: Dict):
    """Applies common Matplotlib styling options from a settings dictionary."""
    ax.set_title(settings.get('title', ''), fontsize=settings.get('title_fontsize', 14))
    ax.set_xlabel(settings.get('xlabel', ''), fontsize=settings.get('xlabel_fontsize', 12))
    ax.set_ylabel(settings.get('ylabel', ''), fontsize=settings.get('ylabel_fontsize', 12))

    # Tick parameters
    tick_fs = settings.get('tick_fontsize', 10)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.tick_params(axis='x', which='major', rotation=settings.get('xtick_rotation', 0))

    # Grid
    ax.grid(visible=settings.get('grid_on', True), linestyle='--', alpha=0.6, axis='both') # Default grid on

    # Legend (only if explicitly requested and possible)
    legend_visible = settings.get('show_legend', False)
    if legend_visible and ax.get_legend_handles_labels()[0]: # Check if there are handles to show
        pos = settings.get('legend_pos', 'best')
        legend_fs = settings.get('legend_fontsize', 'small')
        legend_title = settings.get('legend_title', None)
        try:
            if pos == 'outside':
                # Place legend outside the top right corner
                lgd = ax.legend(title=legend_title, bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=legend_fs)
            else:
                 lgd = ax.legend(title=legend_title, loc=pos, fontsize=legend_fs)
            # Optional: adjust layout if legend is outside
            # fig.tight_layout(rect=[0, 0, 0.85 if pos == 'outside' else 1, 1]) # Adjust rect might be needed
        except Exception as e_legend:
            logging.warning(f"Could not render legend with settings {pos}, {legend_fs}: {e_legend}")
            try: ax.legend(title=legend_title, loc='best', fontsize='small') # Fallback legend
            except Exception: pass # Give up if fallback fails
    elif ax.get_legend():
        ax.get_legend().remove() # Remove legend if not requested

    # Axis Limits (Apply only if values are provided)
    xmin, xmax = settings.get('xmin'), settings.get('xmax')
    ymin, ymax = settings.get('ymin'), settings.get('ymax')
    if xmin is not None or xmax is not None: ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None: ax.set_ylim(bottom=ymin, top=ymax)

    # Tick Locators (MaxNLocator)
    num_xticks = settings.get('num_xticks')
    num_yticks = settings.get('num_yticks')
    # Apply locators only if a positive integer is provided
    if isinstance(num_xticks, int) and num_xticks > 0:
        is_integer = 'episode' in settings.get('xlabel','').lower() # Try integer ticks for episode axis
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=num_xticks, prune='both', integer=is_integer))
    if isinstance(num_yticks, int) and num_yticks > 0:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=num_yticks, prune='both'))

    # Final layout adjustment
    try:
        # Adjust right boundary slightly if legend is outside
        rect_right = 0.85 if settings.get('show_legend', False) and settings.get('legend_pos') == 'outside' else 1.0
        fig.tight_layout(rect=[0, 0, rect_right, 1])
    except ValueError as e_tl:
        logging.warning(f"Could not apply tight_layout automatically: {e_tl}. Manual adjustment might be needed.")
        try: fig.tight_layout() # Try default tight_layout as fallback
        except Exception: pass # Ignore if fallback also fails

def download_plot_button(fig: plt.Figure, filename_base: str, key_suffix: str):
    """Renders a download button for the given Matplotlib figure."""
    if fig is None:
        st.warning("Plot figure is not available for download.")
        return
    buf = io.BytesIO()
    try:
        # Save figure to buffer
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        # Generate download button
        st.download_button(
            label="💾 Download Plot (PNG)",
            data=buf,
            file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d%H%M')}.png",
            mime="image/png",
            key=f"download_{key_suffix}"
        )
    except Exception as e:
        st.error(f"Failed to prepare plot for download: {e}")
        logging.error(f"Plot download preparation failed for {filename_base}: {e}", exc_info=True)
    finally:
        buf.close() # Ensure buffer is closed


# =============================================================================
# == Specific Plotting Functions (with On-Demand Logic) ==
# =============================================================================

def plot_termination_reason_distribution_mpl(summary_df: Optional[pd.DataFrame]):
    """Generates stacked bar chart of termination reasons (Data processing on demand)."""
    st.write("### Termination Reason Distribution (Aggregated from Summary)")
    if summary_df is None or summary_df.empty or 'termination_reason' not in summary_df.columns or 'episode' not in summary_df.columns:
        st.warning("Summary data insufficient for termination plot (requires 'episode' and 'termination_reason').")
        return

    plot_key_base = "term_reason_dist"
    settings_cache_key = f"{STATE_PREFIX_PLOT_SETTINGS}{plot_key_base}"
    data_cache_key = f"{STATE_PREFIX_PLOT_DATA}{plot_key_base}"

    # --- Customization ---
    default_settings = {
        'title': f"Termination Reason Distribution", 'xlabel': "Episode Group Start", 'ylabel': "Count",
        'figsize_w': 10.0, 'figsize_h': 5.0, 'title_fontsize': 14, 'xlabel_fontsize': 12,
        'ylabel_fontsize': 12, 'tick_fontsize': 10, 'xtick_rotation': 45, 'grid_on': True,
        'cmap': 'tab20', 'bar_width': 0.8, 'show_legend': True, 'legend_pos': 'outside',
        'legend_fontsize': 'small', 'num_xticks': None, 'num_yticks': 6, # Auto x ticks based on groups
        'group_size': None # Will be determined dynamically
    }
    cached_settings = st.session_state.get(settings_cache_key, default_settings)
    customize = st.checkbox(f"Customize Termination Plot", key=f"customize_{plot_key_base}", value=False)
    current_settings = render_mpl_plot_customization(plot_key_base, cached_settings, plot_type='bar') if customize else cached_settings
    if customize and current_settings != cached_settings: st.session_state[settings_cache_key] = current_settings

    # --- Process Data Button ---
    if st.button("📊 Generate/Update Termination Plot", key=f"generate_{plot_key_base}"):
        with st.spinner("Processing termination data..."):
            try:
                df = summary_df.copy()
                df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
                df = df.dropna(subset=['episode', 'termination_reason'])
                if df.empty:
                    st.warning("No valid episodes with termination reasons found in summary.")
                    st.session_state[data_cache_key] = None # Clear cache
                    return

                df['episode'] = df['episode'].astype(int)

                # Dynamic grouping based on episode range
                ep_range = df['episode'].max() - df['episode'].min() if len(df['episode']) > 1 else 0
                group_size = 1 if ep_range < 50 else 10 if ep_range < 500 else 50 if ep_range < 2000 else 100 if ep_range < 10000 else 500
                df['episode_group'] = (df['episode'] // group_size) * group_size
                reason_counts_df = df.groupby(['episode_group', 'termination_reason']).size().unstack(fill_value=0)

                if reason_counts_df.empty:
                    st.warning("No termination reasons found after grouping.")
                    st.session_state[data_cache_key] = None
                else:
                    # Store processed data and group size in cache
                    st.session_state[data_cache_key] = {'data': reason_counts_df, 'group_size': group_size}
                    logging.info(f"Cached termination plot data (key: {data_cache_key}) using group size {group_size}")

            except Exception as e:
                st.error(f"Error processing termination data: {e}")
                logging.error(f"Termination data processing failed: {e}", exc_info=True)
                st.session_state[data_cache_key] = None # Clear cache on error
                return

    # --- Display Plot (if data is cached) ---
    cached_plot_data = st.session_state.get(data_cache_key)
    if cached_plot_data and isinstance(cached_plot_data.get('data'), pd.DataFrame):
        reason_counts_df = cached_plot_data['data']
        group_size = cached_plot_data['group_size']
        fig, ax = None, None
        try:
            # Update titles/labels based on dynamic group size
            current_settings['title'] = f"Termination Reason Dist. per ~{group_size} Episodes"
            current_settings['xlabel'] = f"Episode Group Start (Group Size ≈ {group_size})"

            fig, ax = plt.subplots(figsize=current_settings['figsize'])
            reason_counts_df.plot(kind='bar', stacked=True, ax=ax, colormap=current_settings['cmap'], width=current_settings['bar_width'])

            # Apply common styles AFTER plotting
            _apply_common_mpl_styles(ax, fig, current_settings)

            st.pyplot(fig)
            download_plot_button(fig, "termination_reason_distribution", plot_key_base)
            with st.expander("Show Raw Grouped Counts"): st.dataframe(reason_counts_df)

        except Exception as e:
            st.error(f"Error generating termination plot: {e}")
            logging.error(f"Termination plot generation failed: {e}", exc_info=True)
        finally:
            if fig: plt.close(fig); gc.collect()
    elif st.session_state.get(f"generate_{plot_key_base}_clicked", False): # Check if button was clicked but failed
         st.warning("Data processing failed or yielded no results. Click generate again if needed.")

def plot_overview_metrics_mpl(loaded_sim_data: list):
    """Generates overview performance plots (Data processing on demand)."""
    st.write("### Performance Metrics Across Loaded Episodes")
    if not loaded_sim_data:
        st.warning("Load trajectory data first to see overview metrics.")
        return

    plot_key_base_prefix = "overview_metrics"
    data_cache_key = f"{STATE_PREFIX_PLOT_DATA}{plot_key_base_prefix}"

    # --- Process Data Button ---
    if st.button("📊 Generate/Update Overview Metrics Plot", key=f"generate_{plot_key_base_prefix}"):
        with st.spinner("Processing overview metrics from loaded trajectories..."):
            episode_metrics_list = []
            for episode in loaded_sim_data:
                if not isinstance(episode, dict): continue
                ep_num = episode.get('episode')
                # Use extract_trajectory_data for safe access
                reward_list = extract_trajectory_data(episode, 'cumulative_reward')
                time_list = extract_trajectory_data(episode, 'time')

                if ep_num is None or reward_list is None or time_list is None or len(reward_list) == 0 or len(time_list) == 0:
                    # logging.debug(f"Skipping episode {ep_num}: missing essential overview data.")
                    continue

                try:
                    # Get last finite value
                    final_reward = reward_list[np.isfinite(reward_list)][-1] if np.any(np.isfinite(reward_list)) else np.nan
                    final_time = time_list[np.isfinite(time_list)][-1] if np.any(np.isfinite(time_list)) else np.nan
                    ep_num_int = int(ep_num)

                    if not np.isfinite(final_reward) or not np.isfinite(final_time):
                        performance = np.nan
                    else:
                        performance = final_reward / final_time if final_time > 1e-6 else 0.0 # Avoid division by zero

                    episode_metrics_list.append({
                        'episode': ep_num_int,
                        'final_reward': final_reward,
                        'final_time': final_time,
                        'performance': performance
                    })
                except (ValueError, TypeError, IndexError) as e:
                    logging.warning(f"Skipping ep {ep_num} overview calculation: {e}")

            if not episode_metrics_list:
                st.warning("No valid episodes found in loaded data for overview plots.")
                st.session_state[data_cache_key] = None # Clear cache
                return

            # Create DataFrame and store in cache
            metrics_df = pd.DataFrame(episode_metrics_list).sort_values(by='episode').reset_index(drop=True)
            st.session_state[data_cache_key] = metrics_df
            logging.info(f"Cached overview plot data ({len(metrics_df)} eps) for key {data_cache_key}")

    # --- Display Plots (if data is cached) ---
    metrics_df = st.session_state.get(data_cache_key)
    if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        plots_config = [
            {"title": "Final Cumulative Reward vs Episode", "y_col": "final_reward", "ylabel": "Total Reward"},
            {"title": "Episode Duration vs Episode", "y_col": "final_time", "ylabel": "Duration (s)"},
            {"title": "Performance (Reward/Time) vs Episode", "y_col": "performance", "ylabel": "Performance (Reward/s)"}
        ]
        episode_numbers = metrics_df['episode'].tolist()

        for config in plots_config:
            plot_title = config["title"]; y_col = config["y_col"]; default_ylabel = config["ylabel"]
            st.write(f"#### {plot_title}")

            if y_col not in metrics_df.columns:
                st.warning(f"Column '{y_col}' not found in processed overview data."); continue
            y_data = metrics_df[y_col].tolist()
            if len(y_data) != len(episode_numbers):
                st.error(f"Data length mismatch for overview plot '{y_col}'. Processed data might be corrupted."); continue

            plot_key_base = f"{plot_key_base_prefix}_{y_col}"
            settings_cache_key = f"{STATE_PREFIX_PLOT_SETTINGS}{plot_key_base}"
            fig, ax = None, None

            # --- Customization ---
            default_settings = {
                'title': plot_title, 'xlabel': "Episode", 'ylabel': default_ylabel,
                'figsize_w': 10.0, 'figsize_h': 4.0, 'title_fontsize': 14, 'xlabel_fontsize': 12,
                'ylabel_fontsize': 12, 'tick_fontsize': 10, 'xtick_rotation': 30, 'line_color': "#1f77b4",
                'line_width': 1.5, 'line_style': '-', 'marker_color': "#ff7f0e", 'marker_size': 2,
                'marker_style': 'o', 'grid_on': True, 'num_xticks': 10, 'num_yticks': 6, 'show_legend': False
            }
            cached_settings = st.session_state.get(settings_cache_key, default_settings)
            # Update dynamic defaults
            cached_settings['xlabel'] = "Episode"
            cached_settings['ylabel'] = default_ylabel
            cached_settings['title'] = plot_title

            customize = st.checkbox(f"Customize Plot", key=f"customize_{plot_key_base}", value=False)
            current_settings = render_mpl_plot_customization(plot_key_base, cached_settings, plot_type='line') if customize else cached_settings
            if customize and current_settings != cached_settings: st.session_state[settings_cache_key] = current_settings

            # --- Plotting ---
            try:
                fig, ax = plt.subplots(figsize=current_settings['figsize'])
                ax.plot(episode_numbers, y_data,
                        marker=current_settings['marker_style'],
                        linestyle=current_settings['line_style'],
                        color=current_settings['line_color'],
                        linewidth=current_settings['line_width'],
                        markersize=current_settings['marker_size'],
                        markerfacecolor=current_settings['marker_color'],
                        markeredgecolor=current_settings['marker_color'])

                # Apply common styles after plotting
                _apply_common_mpl_styles(ax, fig, current_settings)

                st.pyplot(fig)
                download_plot_button(fig, f"overview_{y_col}", plot_key_base)
            except Exception as e:
                st.error(f"Error plotting '{plot_title}': {e}")
                logging.error(f"Overview plot generation failed for {y_col}: {e}", exc_info=True)
            finally:
                if fig: plt.close(fig); gc.collect()
            st.divider()
    elif st.session_state.get(f"generate_{plot_key_base_prefix}_clicked", False): # Check if button was clicked
        st.warning("Data processing failed or yielded no results. Click generate again if needed.")


def plot_heatmap(sim_data: list):
    """Generates a heatmap using Matplotlib, with on-demand data extraction and customization."""
    st.write("### Heatmaps of Trajectories (Aggregated)")
    st.caption("Density plot showing frequency of state pairs across selected & loaded episodes.")

    if not sim_data:
        st.warning("Load trajectory data to generate heatmaps.")
        return

    # --- Determine Available Parameters ---
    available_params = set()
    if sim_data:
        # Check keys in the first few episodes that are lists/arrays of numbers
        for episode in sim_data[:min(5, len(sim_data))]: # Check first 5 episodes
            if isinstance(episode, dict):
                for key, value in episode.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0 and key not in ['episode']:
                        # Quick check if the first element looks numeric
                        try:
                           if isinstance(value[0], (int, float, np.number)):
                               available_params.add(key)
                        except (IndexError, TypeError): continue
    sorted_params = sorted(list(available_params))

    if len(sorted_params) < 2:
        st.warning("Insufficient plottable trajectory variables found (need at least 2 numeric lists/arrays).")
        return

    # --- Parameter Selection ---
    col1, col2 = st.columns(2)
    with col1:
        # Try common defaults first
        default_x = next((p for p in ['time', 'cart_position', 'pendulum_angle'] if p in sorted_params), sorted_params[0])
        x_idx = sorted_params.index(default_x)
        x_param = st.selectbox("X-axis parameter", sorted_params, index=x_idx, key="heatmap_x")
    with col2:
        # Try common defaults, ensure different from x
        default_y_prefs = ['pendulum_angle', 'pendulum_velocity', 'cumulative_reward', 'error', 'cart_velocity']
        default_y = next((p for p in default_y_prefs if p in sorted_params and p != x_param), None)
        if default_y is None: # Find first param different from x
            default_y = next((p for p in sorted_params if p != x_param), sorted_params[1] if len(sorted_params)>1 and sorted_params[1]!=x_param else sorted_params[0])
        y_idx = sorted_params.index(default_y)
        y_param = st.selectbox("Y-axis parameter", sorted_params, index=y_idx, key="heatmap_y")

    if not x_param or not y_param: st.info("Select X and Y parameters."); return
    if x_param == y_param: st.warning("X and Y parameters must be different."); return

    # --- Customization ---
    plot_key_base = f"heatmap_{x_param}_{y_param}"
    settings_cache_key = f"{STATE_PREFIX_PLOT_SETTINGS}{plot_key_base}"
    data_cache_key = f"{STATE_PREFIX_PLOT_DATA}{plot_key_base}" # Cache key for extracted X/Y data

    default_title = f"Heatmap: {y_param.replace('_',' ').title()} vs {x_param.replace('_',' ').title()}"
    default_settings = {
        'title': default_title, 'xlabel': x_param.replace('_', ' ').title(), 'ylabel': y_param.replace('_', ' ').title(),
        'figsize_w': 10.0, 'figsize_h': 7.0, 'title_fontsize': 14, 'xlabel_fontsize': 12,
        'ylabel_fontsize': 12, 'tick_fontsize': 10, 'xtick_rotation': 0, 'grid_on': False, # Grid usually off
        'cmap': 'viridis', 'heatmap_bins': 100, 'heatmap_log_scale': False,
        'clabel': 'Frequency', 'clabel_fontsize': 10, 'num_xticks': None, 'num_yticks': 6,
        'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'cmin': None, 'cmax': None # Ranges default to auto
    }
    cached_settings = st.session_state.get(settings_cache_key, default_settings)
    # Update dynamic defaults
    cached_settings['xlabel'] = x_param.replace('_', ' ').title()
    cached_settings['ylabel'] = y_param.replace('_', ' ').title()
    cached_settings['title'] = default_title
    cached_settings['clabel'] = 'Frequency' + (' (Log Scale)' if cached_settings.get('heatmap_log_scale') else '')


    customize = st.checkbox("Customize Heatmap", key=f"customize_{plot_key_base}", value = False)
    current_settings = render_mpl_plot_customization(plot_key_base, cached_settings, plot_type='heatmap') if customize else cached_settings
    if customize and current_settings != cached_settings: st.session_state[settings_cache_key] = current_settings

    # --- Generate Button and Plotting ---
    if st.button("📊 Generate/Update Heatmap", key=f"generate_{plot_key_base}"):
        # --- Data Extraction (On Demand) ---
        with st.spinner(f"Extracting data for {y_param} vs {x_param}..."):
            extracted_data = extract_heatmap_xy_data(sim_data, x_param, y_param)

        if extracted_data is None:
            st.error(f"Failed to extract valid data for heatmap ('{x_param}' vs '{y_param}'). Check logs for details.")
            st.session_state[data_cache_key] = None # Clear cache
            return
        # Store extracted data in cache
        st.session_state[data_cache_key] = {'x': extracted_data[0], 'y': extracted_data[1]}
        logging.info(f"Cached heatmap data for key: {data_cache_key}")

    # --- Display Plot (if data is cached) ---
    cached_plot_data = st.session_state.get(data_cache_key)
    if cached_plot_data and 'x' in cached_plot_data and 'y' in cached_plot_data:
        x_data = cached_plot_data['x']
        y_data = cached_plot_data['y']

        if len(x_data) == 0:
             st.warning("No valid data points found for the selected heatmap parameters after filtering."); return

        logging.info(f"Heatmap: Plotting {len(x_data)} points for {y_param} vs {x_param}")
        fig, ax = None, None
        with st.spinner("Generating heatmap plot..."):
            try:
                fig, ax = plt.subplots(figsize=current_settings['figsize'])

                # Determine normalization (Log or Linear)
                norm = None
                if current_settings.get('heatmap_log_scale'):
                     # LogNorm requires vmin > 0 if provided
                     cmin_log = current_settings.get('cmin')
                     norm = mcolors.LogNorm(vmin=cmin_log if cmin_log and cmin_log > 0 else None,
                                            vmax=current_settings.get('cmax'))
                else:
                     norm = mcolors.Normalize(vmin=current_settings.get('cmin'),
                                              vmax=current_settings.get('cmax'))

                # Determine histogram range [[xmin, xmax], [ymin, ymax]]
                hist_range = None
                if current_settings.get('xmin') is not None and current_settings.get('xmax') is not None and \
                   current_settings.get('ymin') is not None and current_settings.get('ymax') is not None:
                    hist_range = [
                        [current_settings['xmin'], current_settings['xmax']],
                        [current_settings['ymin'], current_settings['ymax']]
                    ]

                # hist2d requires cmin >= 1 for LogNorm, usually better to handle via norm.vmin
                hist_cmin = current_settings.get('cmin') if not current_settings.get('heatmap_log_scale') else None

                counts, xedges, yedges, img = ax.hist2d(
                    x_data, y_data,
                    bins=current_settings.get('heatmap_bins', 100),
                    cmap=current_settings.get('cmap', 'viridis'),
                    norm=norm,
                    range=hist_range,
                    cmin=hist_cmin # Only use cmin directly for linear scale
                )

                # --- Colorbar ---
                cbar = fig.colorbar(img, ax=ax)
                cbar.set_label(current_settings.get('clabel', 'Frequency'), fontsize=current_settings.get('clabel_fontsize', 10))
                cbar.ax.tick_params(labelsize=current_settings.get('tick_fontsize', 10)) # Match axis tick size

                # Apply common styles AFTER hist2d and colorbar
                _apply_common_mpl_styles(ax, fig, current_settings)

                st.pyplot(fig)
                download_plot_button(fig, f"heatmap_{x_param}_vs_{y_param}", plot_key_base)

            except Exception as e:
                st.error(f"Error generating heatmap plot: {e}")
                logging.error(f"Heatmap generation failed: {traceback.format_exc()}")
            finally:
                 if fig: plt.close(fig); gc.collect()
    elif st.session_state.get(f"generate_{plot_key_base}_clicked", False): # Check if button was clicked
        st.warning("Data processing failed or yielded no results. Click generate again if needed.")


def display_agent_state_comparison(available_ep_numbers: List[int], folder_path: str, config: Dict):
    """Displays Q-Table and Visit Count comparisons loaded from agent_state files."""
    st.write("### Agent State Comparison (Q-Tables & Visit Counts)")
    st.caption("Compares the learned tables between two saved agent states.")

    if not available_ep_numbers:
        st.info("No agent state files (`agent_state_episode_*.json`) found in this folder.")
        st.caption("Ensure `save_agent_state=True` during simulation and files are in the selected results folder.")
        return

    if len(available_ep_numbers) < 2:
        st.info("Requires at least two agent state files (e.g., start and end of training) to perform a comparison.")
        return

    # --- Episode Selection ---
    col1, col2 = st.columns(2)
    with col1:
        # Default to first available state
        ep1_num = st.selectbox("Select first episode state:", available_ep_numbers, index=0, key="q_ep1")
    with col2:
        # Default to last available state, ensure different from first
        valid_ep2_options = [ep for ep in available_ep_numbers if ep != ep1_num]
        default_ep2_index = len(valid_ep2_options) - 1 if valid_ep2_options else 0 # Last state index

        if not valid_ep2_options:
             st.warning("Only one agent state file available. Cannot compare.")
             return
        ep2_num = st.selectbox("Select second episode state:", valid_ep2_options, index=default_ep2_index, key="q_ep2")

    if ep1_num == ep2_num:
        st.warning("Please select two different episodes for comparison.")
        return

    # --- Load Agent State Data (Uses cached function) ---
    agent_state1 = None
    agent_state2 = None
    error_loading = False
    with st.spinner(f"Loading agent states for Ep {ep1_num} and Ep {ep2_num}..."):
        try:
            agent_state1 = load_agent_state_file(folder_path, ep1_num)
            if not agent_state1: st.error(f"Failed to load agent state for Episode {ep1_num}. File might be missing or corrupted."); error_loading = True
        except Exception as e1: st.error(f"Error loading state Ep {ep1_num}: {e1}"); error_loading = True

        try:
            agent_state2 = load_agent_state_file(folder_path, ep2_num)
            if not agent_state2: st.error(f"Failed to load agent state for Episode {ep2_num}. File might be missing or corrupted."); error_loading = True
        except Exception as e2: st.error(f"Error loading state Ep {ep2_num}: {e2}"); error_loading = True

    if error_loading or not agent_state1 or not agent_state2: return # Stop if loading failed

    # --- Extract Tables and Identify Common Gains ---
    q_tables1 = agent_state1.get('q_tables', {})
    q_tables2 = agent_state2.get('q_tables', {})
    visit_counts1 = agent_state1.get('visit_counts', {})
    visit_counts2 = agent_state2.get('visit_counts', {})

    # Find gain types present in both Q-tables OR both Visit Counts
    common_q_gains = set(q_tables1.keys()) & set(q_tables2.keys())
    common_v_gains = set(visit_counts1.keys()) & set(visit_counts2.keys())
    # Combine and sort the common gains available for selection
    available_gains = sorted(list(common_q_gains | common_v_gains))

    if not available_gains:
        st.warning("No common gain types (e.g., 'kp', 'ki', 'kd') found with data in both selected agent states.")
        return

    selected_gain = st.selectbox("Select gain type for comparison:", available_gains, key="q_gain_select")

    # --- Helper Function to Process and Display a Table Type (Q or Visits) ---
    def display_single_table_type(data1: Dict, data2: Dict, gain: str, ep1: int, ep2: int, table_type: str):
        st.subheader(f"{table_type} Comparison for Gain: `{gain.upper()}`")
        col11, col22 = st.columns(2) # Side-by-side display

        data1_gain = data1.get(gain)
        data2_gain = data2.get(gain)

        # Define standard action names (assuming 3 actions: Decrease, Keep, Increase)
        action_names_map = {0: 'Decrease', 1: 'Keep', 2: 'Increase'} # Map index to name

        for col, ep, data_gain, side in [(col11, ep1, data1_gain, "1"), (col22, ep2, data2_gain, "2")]:
            with col:
                st.write(f"**Episode {ep}:**")
                if data_gain is None:
                    st.warning(f"No {table_type} data for gain '{gain}' found.")
                    continue
                if not isinstance(data_gain, list) or not all(isinstance(item, dict) for item in data_gain):
                    st.error(f"Invalid {table_type} data format for gain '{gain}' (expected list of dicts).")
                    logging.error(f"Invalid agent state format for {table_type}/{gain}/Ep{ep}. Got: {type(data_gain)}")
                    continue
                if not data_gain:
                     st.info(f"{table_type} table for gain '{gain}' is empty.")
                     continue

                try:
                    df = pd.DataFrame(data_gain)
                    # Identify state columns (not action indices)
                    action_cols = [col for col in df.columns if str(col).isdigit()] # Actions are likely '0', '1', '2', etc.
                    state_cols = [col for col in df.columns if col not in action_cols]

                    if not state_cols:
                         st.warning("Could not identify state columns. Displaying raw table.")
                         index_set = False
                    else:
                         try:
                             # Attempt to set multi-index if multiple state cols
                             df = df.set_index(state_cols)
                             index_set = True
                         except KeyError:
                             st.warning(f"Duplicate states found for gain '{gain}'. Displaying without index.")
                             index_set = False


                    # Rename action columns based on index/integer value
                    col_rename = {col: action_names_map.get(int(col), f"Action_{col}") for col in action_cols if str(col).isdigit()}
                    # Keep non-numeric columns as is (shouldn't happen for actions)
                    col_rename.update({col: col for col in action_cols if not str(col).isdigit()})
                    df_renamed = df[action_cols].rename(columns=col_rename) # Select only action columns for display after indexing

                    # Apply styling
                    style = df_renamed.style
                    if table_type == "Q-Table":
                        style = style.apply(resaltar_maximo, axis=1).format(precision=4)
                    else: # Visit Counts
                        style = style.format(precision=0) # Integers

                    st.dataframe(style)

                except Exception as df_e:
                    st.error(f"Error processing/displaying {table_type} for gain '{gain}': {df_e}")
                    logging.error(f"DataFrame error {table_type}/Ep{ep}/Gain{gain}: {df_e}", exc_info=True)
                    # st.dataframe(data_gain) # Show raw data on error

        if table_type == "Q-Table":
            st.info("💡 Green cells highlight the action with the highest Q-value for that state (row).", icon="✅")
        st.divider()


    # --- Display Selected Gain Comparison ---
    display_q = selected_gain in common_q_gains
    display_v = selected_gain in common_v_gains

    if display_q:
        display_single_table_type(q_tables1, q_tables2, selected_gain, ep1_num, ep2_num, "Q-Table")
    else:
        st.info(f"Q-Table data for gain '{selected_gain}' not available in both selected episodes.")

    if display_v:
        display_single_table_type(visit_counts1, visit_counts2, selected_gain, ep1_num, ep2_num, "Visit Count")
    else:
        st.info(f"Visit Count data for gain '{selected_gain}' not available in both selected episodes.")


def plot_episode_details_graphs_mpl(episode_data: dict, episode_number: Union[int, str]):
    """Generates plots for selected variables from a single episode's data."""
    st.write(f"**Episode Trajectory Plots (Episode {episode_number})**")
    ep_num = episode_number
    plottable_vars = []
    if isinstance(episode_data, dict):
        for k, v in episode_data.items():
             # Check if it's a list/array and seems numeric (check first element)
             if isinstance(v, (list, np.ndarray)) and len(v) > 0 and k not in ['episode']:
                 try:
                     if isinstance(v[0], (int, float, np.number)): plottable_vars.append(k)
                 except (IndexError, TypeError): pass
        plottable_vars = sorted(plottable_vars)
    else: st.error("Invalid episode data provided."); return
    if not plottable_vars: st.warning("No plottable numeric trajectory data found in this episode."); return

    # --- State for selections (persists within the episode view) ---
    selection_state_key = f"plot_selections_ep_{ep_num}"
    if selection_state_key not in st.session_state:
        # Sensible defaults
        default_y = [p for p in ['pendulum_angle', 'cart_position', 'force', 'reward', 'cumulative_reward', 'error'] if p in plottable_vars][:4] # Show up to 4 common ones
        st.session_state[selection_state_key] = {
            'x_param': 'time' if 'time' in plottable_vars else plottable_vars[0],
            'y_params': default_y
        }
    plot_selections = st.session_state[selection_state_key]

    # --- Variable Selection Widgets ---
    col1, col2 = st.columns([1, 2])
    with col1:
        x_opts = plottable_vars
        x_idx = x_opts.index(plot_selections['x_param']) if plot_selections['x_param'] in x_opts else (x_opts.index('time') if 'time' in x_opts else 0)
        # Update selection in state if changed
        plot_selections['x_param'] = st.selectbox("X-axis:", x_opts, index=x_idx, key=f'detail_plot_x_{ep_num}')
    with col2:
        y_opts = [v for v in plottable_vars if v != plot_selections['x_param']]
        # Ensure default selection contains only valid options
        current_y_selection = [p for p in plot_selections.get('y_params', []) if p in y_opts]
        # Update selection in state if changed
        plot_selections['y_params'] = st.multiselect("Y-axis variables:", y_opts, default=current_y_selection, key=f'detail_plot_y_{ep_num}')

    x_param = plot_selections['x_param']
    y_params_to_plot = plot_selections['y_params']

    if not x_param or not y_params_to_plot:
        st.info("Select at least one Y-axis variable to plot.")
        return

    # --- Common Customization for all detail plots ---
    common_settings_key = f"{STATE_PREFIX_PLOT_SETTINGS}detail_common_{ep_num}"
    default_settings = {
        'title': '', 'xlabel': x_param.replace('_', ' ').title(), 'ylabel': '', # Y label set per plot
        'figsize_w': 10.0, 'figsize_h': 4.0, 'title_fontsize': 12, 'xlabel_fontsize': 11,
        'ylabel_fontsize': 11, 'tick_fontsize': 9, 'xtick_rotation': 0, 'line_color': "#1f77b4",
        'line_width': 1.5, 'line_style': '-', 'marker_color': "#ff7f0e", 'marker_size': 0,
        'marker_style': '', 'grid_on': True, 'num_xticks': 10, 'num_yticks': 5, 'show_legend': False
    }
    cached_settings = st.session_state.get(common_settings_key, default_settings)
    cached_settings['xlabel'] = x_param.replace('_', ' ').title() # Ensure X label matches selection

    customize = st.checkbox("Customize Detail Plots", key=f"customize_detail_plots_{ep_num}", value=False)
    widget_key_base = f"detail_common_widgets_{ep_num}" # Unique key for these widgets
    current_common_settings = render_mpl_plot_customization(widget_key_base, cached_settings, plot_type='line') if customize else cached_settings
    if customize and current_common_settings != cached_settings: st.session_state[common_settings_key] = current_common_settings

    # --- Extract X data once ---
    x_data_numeric = extract_trajectory_data(episode_data, x_param)
    if x_data_numeric is None:
        st.error(f"Could not extract valid numeric data for X-axis ('{x_param}'). Cannot generate plots."); return

    # --- Plot each selected Y variable ---
    st.write("---") # Separator
    for y_param in y_params_to_plot:
        plot_key_base = f"detail_{ep_num}_{x_param}_{y_param}"
        aligned_data_cache_key = f"{STATE_PREFIX_PLOT_DATA}aligned_{plot_key_base}"

        # --- Get Aligned Data (Cache or Calculate) ---
        aligned_data = None
        # Check cache first
        cached_aligned_data = st.session_state.get(aligned_data_cache_key)

        # Decide if recalculation is needed (e.g., if X or Y param changed implicitly)
        # For simplicity here, we recalculate if not in cache. More complex checks could be added.
        recalculate_alignment = aligned_data_cache_key not in st.session_state

        if recalculate_alignment:
            with st.spinner(f"Aligning {y_param} vs {x_param}..."):
                 y_data_numeric = extract_trajectory_data(episode_data, y_param)
                 aligned_pair = align_trajectory_data(x_data_numeric, y_data_numeric)
                 if aligned_pair:
                     st.session_state[aligned_data_cache_key] = {'x': aligned_pair[0], 'y': aligned_pair[1]}
                     aligned_data = st.session_state[aligned_data_cache_key]
                     logging.debug(f"Cached aligned data for {plot_key_base}")
                 else:
                     st.warning(f"No valid aligned data points found for '{y_param}' vs '{x_param}'. Skipping plot.")
                     st.session_state[aligned_data_cache_key] = None # Cache failure explicitly
                     continue # Skip to next y_param
        elif cached_aligned_data:
             aligned_data = cached_aligned_data
        else: # Cached but None (previous failure)
             st.warning(f"Skipping plot for '{y_param}' vs '{x_param}' due to previous alignment failure.")
             continue


        # --- Plotting (if aligned data exists) ---
        if aligned_data:
            x_final = aligned_data['x']
            y_final = aligned_data['y']
            fig, ax = None, None
            try:
                # Prepare settings specific to this plot
                plot_settings = current_common_settings.copy()
                plot_settings['ylabel'] = y_param.replace('_', ' ').title()
                plot_settings['title'] = f"Ep {ep_num}: {plot_settings['ylabel']} vs {plot_settings['xlabel']}"

                fig, ax = plt.subplots(figsize=plot_settings['figsize'])
                ax.plot(x_final, y_final,
                        color=plot_settings['line_color'],
                        linewidth=plot_settings['line_width'],
                        linestyle=plot_settings['line_style'],
                        marker=plot_settings['marker_style'],
                        markersize=plot_settings['marker_size'],
                        markerfacecolor=plot_settings['marker_color'],
                        markeredgecolor=plot_settings['marker_color'])

                # Apply common styles AFTER plotting
                _apply_common_mpl_styles(ax, fig, plot_settings)

                st.pyplot(fig)
                download_plot_button(fig, f"ep_{ep_num}_{y_param}_vs_{x_param}", plot_key_base)
                st.write("---") # Separator between plots

            except Exception as e:
                st.error(f"Failed to generate plot for '{y_param}': {e}")
                logging.error(f"Detail plot failed ({y_param} vs {x_param}): {e}", exc_info=True)
            finally:
                if fig: plt.close(fig); gc.collect()


def plot_episode_animation(episode_data: dict):
    """Handles the display and generation of the pendulum animation."""
    ep_id = episode_data.get('episode', 'N/A')
    st.write(f"**Animation (Episode {ep_id})**")

    # Check basic requirements
    req_keys = ['time', 'cart_position', 'pendulum_angle']
    has_required_data = all(isinstance(episode_data.get(k), (list, np.ndarray)) and len(episode_data[k]) > 0 for k in req_keys)
    if not has_required_data:
        st.warning("Animation cannot be generated: Missing essential trajectory data (time, cart_position, pendulum_angle).")
        return
    if not st.session_state.get('selected_folder_path'):
        st.warning("Cannot generate/find animation: Results folder path is not set.")
        return

    anim_key = f"anim_ep_{ep_id}"
    # Checkbox to enable animation generation/display
    show_anim = st.checkbox("Generate/Show Animation", value=False, key=f"{anim_key}_show")

    if not show_anim:
        st.caption("Enable the checkbox above to generate or display the animation.")
        return

    anim_fname = f"animation_episode_{ep_id}.gif"
    anim_path = os.path.join(st.session_state.selected_folder_path, anim_fname)
    regen_ui = False # Flag to show generation UI

    # --- Display Cached Animation or Trigger Generation ---
    if os.path.exists(anim_path):
        st.success(f"Cached animation found: `{os.path.basename(anim_fname)}`")
        try:
            st.image(anim_path, caption=f"Cached Animation - Episode {ep_id}")
            if st.button("🔄 Regenerate Animation", key=f"{anim_key}_regen"):
                try:
                    os.remove(anim_path)
                    st.info("Removed cached animation file.")
                    regen_ui = True
                except OSError as e:
                    st.error(f"Error removing cached animation file: {e}")
                    regen_ui = True # Still allow generation attempt
        except Exception as e:
            st.error(f"Error displaying cached animation: {e}")
            regen_ui = True # Allow regeneration if display fails
    else:
        st.info(f"Animation file not found. Generate it below.")
        regen_ui = True

    # --- Animation Generation UI ---
    if regen_ui:
        st.write("**Configure Animation Settings:**")
        cols = st.columns(3)
        with cols[0]:
            fps = st.slider("FPS (Frames Per Second)", 10, 60, 30, key=f"{anim_key}_fps")
            speed = st.slider("Playback Speed Multiplier", 0.1, 5.0, 1.0, 0.1, format="%.1fx", key=f"{anim_key}_speed")
            dpi = st.select_slider("Resolution (DPI)", [75, 100, 150, 200], 100, key=f"{anim_key}_dpi")
        with cols[1]:
            # Auto-detect X limits based on data
            cart_pos = extract_trajectory_data(episode_data, 'cart_position')
            max_abs_x = np.nanmax(np.abs(cart_pos)) if cart_pos is not None and len(cart_pos)>0 else 3.0
            max_abs_x = max(3.0, max_abs_x) # Ensure minimum width
            def_xlim = (-max_abs_x * 1.2, max_abs_x * 1.2)
            x_lim = st.slider("X-Axis Limits", -20.0, 20.0, tuple(map(float, def_xlim)), 0.5, key=f"{anim_key}_xlim")
        with cols[2]:
            # Auto-detect Y limits based on pendulum length from config (fallback)
            pend_len = st.session_state.get('config', {}).get('environment', {}).get('pendulum_length', 2.0)
            def_ylim = (-pend_len * 1.2, pend_len * 1.2)
            y_lim = st.slider("Y-Axis Limits", -5.0, 5.0, tuple(map(float, def_ylim)), 0.5, key=f"{anim_key}_ylim")

        # --- Generate Button ---
        if st.button("🎬 Generate Animation", key=f"{anim_key}_gen", type="primary"):
            # Prepare config dict for animator
            anim_config = {
                "fps": fps, "speed": speed, "x_lim": x_lim, "y_lim": y_lim, "dpi": dpi,
                "pendulum_length": pend_len # Pass pendulum length from config
            }
            progress_container = st.empty()
            progress_bar = progress_container.progress(0.0)
            status_text = st.empty()
            status_text.text("Initializing animation...")
            animator: Optional[PendulumAnimator] = None
            anim: Optional[animation.FuncAnimation] = None

            try:
                # 1. Create Animator instance (handles data validation internally)
                animator = PendulumAnimator(episode_data, anim_config)

                # 2. Create Animation object
                status_text.text("Creating animation frames...")
                anim = animator.create_animation()

                if anim:
                    # 3. Save Animation
                    status_text.text(f"Saving animation to {os.path.basename(anim_fname)}...")
                    total_frames = animator.num_frames # Get actual number of frames from animator
                    def progress_callback(current_frame, total_frames):
                        p = min(1.0, (current_frame + 1) / total_frames) if total_frames > 0 else 0
                        progress_bar.progress(p)

                    # Use pillow writer for GIF
                    anim.save(anim_path, writer='pillow', fps=anim_config['fps'], dpi=anim_config['dpi'],
                              progress_callback=progress_callback)

                    progress_bar.progress(1.0)
                    progress_container.empty()
                    status_text.empty()
                    st.success(f"Animation saved successfully!")
                    st.image(anim_path, caption=f"Generated Animation - Episode {ep_id}")
                else:
                    # Error message handled within create_animation
                    progress_container.empty()
                    status_text.empty()
                    st.error("Animation creation failed. Check logs.")

            except ValueError as ve: # Catch specific init errors from PendulumAnimator
                progress_container.empty(); status_text.empty()
                st.error(f"Animation Initialization Error: {ve}")
            except Exception as e:
                progress_container.empty(); status_text.empty()
                st.error(f"Failed during animation generation/saving: {e}")
                logging.error(f"Animation process failed: {e}\n{traceback.format_exc()}")
            finally:
                # IMPORTANT: Close the figure associated with the animator
                if animator:
                    animator.close_figure()
                gc.collect()


# =============================================================================
# == Tab Rendering Functions ==
# =============================================================================

def display_simulation_config():
    """Displays the loaded simulation configuration from metadata."""
    st.subheader("Simulation Configuration (from metadata.json)")
    config_data = st.session_state.get('config', {})
    if config_data:
        try:
            st.text_area("Config JSON", value=json.dumps(config_data, indent=2), height=300, key="config_display")
        except Exception as e:
            st.error(f"Error displaying config JSON: {e}")
    else:
        st.warning("Configuration data not available in `metadata.json` or metadata file not loaded.")

def introduction_page():
    """Renders the Introduction & Config tab content."""
    col1, col2 = st.columns([1, 2]) # Adjust column ratio if needed
    with col1:
        # Display logo if available
        logo_path = 'logo_final.png' # Assuming logo is in the same directory as app.py
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width='auto')
        else:
            logging.warning(f"Logo file not found at: {logo_path}")
            st.caption("Logo image not found.")

        st.markdown("### Simulation Results Dashboard")
        st.caption("Visualize and analyze dynamic system simulation outputs.")
        st.divider()
        st.subheader("Contact Information:")
        st.markdown("""
        **Ph.D (s) Davor Matías Samuel Ibarra Pérez**
        *UPV / USACH*
        📧 `dibasam@doctor.upv.es` / `davor.ibarra@usach.cl`
        """, unsafe_allow_html=True)
        st.markdown("""
        **Ph.D Javier Sanchis**
        *Universitat Politècnica de València (UPV)*
        📧 `jsanchis@isa.upv.es`
        """, unsafe_allow_html=True)
        st.divider()

    with col2:
        if st.session_state.get('selected_folder_path'):
            st.write(f"**Selected Folder:** `{st.session_state.get('selected_folder_name', 'N/A')}`")
            st.divider()
            # --- Display Config ---
            display_simulation_config()
            st.divider()
            # --- Display Summary Statistics ---
            if st.session_state.get('summary_df') is not None:
                st.subheader("Overall Summary Statistics")
                st.caption("Descriptive statistics from `summary.xlsx` (or `.csv`).")
                try:
                    st.dataframe(st.session_state.summary_df.describe().style.format(precision=4))
                except Exception as e:
                    st.error(f"Error displaying summary statistics: {e}")
                    logging.error(f"Summary describe() error: {e}", exc_info=True)
            else:
                st.warning("Summary data (`summary.xlsx` or `.csv`) not loaded or found in the selected folder.")
        else:
            st.info("Select a results folder from the sidebar to view details.")


def performance_overview():
    """Displays performance plots and agent state analysis."""
    st.subheader("Performance Overview & Agent State Analysis")

    # --- Termination Reason Plot ---
    summary_df = st.session_state.get('summary_df')
    if summary_df is not None:
        plot_termination_reason_distribution_mpl(summary_df)
    else:
        st.info("Summary data (`summary.xlsx` or `.csv`) not loaded. Termination plot unavailable.")
    st.divider()

    # --- Trajectory-Based Plots (Overview Metrics & Heatmap) ---
    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    sim_data = loaded_data.get('simulation_data', [])
    if sim_data:
        st.markdown("#### Trajectory-Based Analysis (Requires Loaded Episodes)")
        plot_overview_metrics_mpl(sim_data)
        plot_heatmap(sim_data)
        st.divider()
    elif not st.session_state.get('selected_folder_path'):
         st.info("Select a results folder from the sidebar.") # Should not happen if navigation works
    elif st.session_state.get('selected_episode_numbers_to_load') and not st.session_state.get('available_loaded_episode_numbers'):
         st.warning("Episodes are selected in the sidebar, but trajectory data is not loaded. Click 'Load Selected Trajectory Data' first.")
    else: # Folder selected, but no episodes selected/loaded
         st.info("Load trajectory data from the sidebar (using filters if needed) to see overview plots and heatmaps based on actual trajectories.")

    # --- Agent State Comparison Section ---
    st.markdown("#### Agent State Analysis (Requires Saved Agent States)")
    available_agent_state_eps = st.session_state.get('available_agent_state_episodes', [])
    folder_path = st.session_state.get('selected_folder_path')
    config = st.session_state.get('config', {})

    if folder_path and available_agent_state_eps:
        display_agent_state_comparison(available_agent_state_eps, folder_path, config)
    elif not folder_path:
         st.info("Select a results folder first to analyze agent states.")
    else: # Folder selected, but no agent state files found
         st.info("No agent state files (`agent_state_episode_*.json`) found in this folder.")
         st.caption("To enable agent state comparison, ensure `save_agent_state=True` was set during the simulation run.")


def episode_details():
    """Renders the Episode Details tab content."""
    st.subheader("Detailed Episode Analysis")

    loaded_data = st.session_state.get('loaded_episode_data', {'simulation_data': []})
    loaded_sim_data = loaded_data.get('simulation_data', [])
    available_loaded_eps = st.session_state.get('available_loaded_episode_numbers', [])

    if not loaded_sim_data or not available_loaded_eps:
        st.warning("No trajectory data loaded. Please select episodes and click 'Load Selected Trajectory Data' in the sidebar.")
        return

    # --- Sidebar Navigation for Episodes ---
    st.sidebar.divider()
    st.sidebar.subheader("Navigate Loaded Episodes")
    num_eps = len(available_loaded_eps)
    min_ep, max_ep = min(available_loaded_eps), max(available_loaded_eps)

    # Ensure current_episode_index is valid
    if 'current_episode_index' not in st.session_state or \
       st.session_state.current_episode_index < 0 or \
       st.session_state.current_episode_index >= num_eps:
        st.session_state.current_episode_index = 0 # Default to first loaded episode

    current_ep_num_in_state = available_loaded_eps[st.session_state.current_episode_index]

    # Input for jumping to an episode (ensure it updates state correctly)
    # Use a temporary key to detect changes made by the user vs code
    target_ep_num_input = st.sidebar.number_input(
        f"Select Episode ({min_ep}-{max_ep}):",
        min_value=min_ep, max_value=max_ep,
        value=current_ep_num_in_state,
        step=1, key="detail_episode_input_widget", format="%d"
    )

    # Check if user changed the number input value
    needs_rerun = False
    if target_ep_num_input != current_ep_num_in_state:
        try:
            # Find the index of the user-selected episode
            if target_ep_num_input in available_loaded_eps:
                new_idx = available_loaded_eps.index(target_ep_num_input)
                if new_idx != st.session_state.current_episode_index:
                    st.session_state.current_episode_index = new_idx
                    needs_rerun = True # Rerun to show the new episode
            else:
                # If entered number not loaded, find closest available and navigate
                closest_ep = min(available_loaded_eps, key=lambda x: abs(x - target_ep_num_input))
                new_idx = available_loaded_eps.index(closest_ep)
                if new_idx != st.session_state.current_episode_index:
                     st.session_state.current_episode_index = new_idx
                     st.sidebar.info(f"Episode {target_ep_num_input} not loaded. Showing closest: {closest_ep}")
                     # Force the number input to update on the *next* run
                     st.session_state.detail_episode_input_widget = closest_ep
                     needs_rerun = True
                else:
                     # User entered same closest episode again, no rerun needed unless value mismatch
                     if st.session_state.detail_episode_input_widget != closest_ep:
                          st.session_state.detail_episode_input_widget = closest_ep
                          needs_rerun = True

        except Exception as nav_e:
            st.sidebar.error(f"Navigation error: {nav_e}")

    # Previous/Next Buttons
    col_prev, col_next = st.sidebar.columns(2)
    prev_disabled = (st.session_state.current_episode_index == 0)
    next_disabled = (st.session_state.current_episode_index >= num_eps - 1)

    if col_prev.button("⬅️ Previous", key="prev_ep_btn", use_container_width=True, disabled=prev_disabled):
        st.session_state.current_episode_index -= 1
        # Update number input to match button click
        st.session_state.detail_episode_input_widget = available_loaded_eps[st.session_state.current_episode_index]
        needs_rerun = True

    if col_next.button("Next ➡️", key="next_ep_btn", use_container_width=True, disabled=next_disabled):
        st.session_state.current_episode_index += 1
        # Update number input to match button click
        st.session_state.detail_episode_input_widget = available_loaded_eps[st.session_state.current_episode_index]
        needs_rerun = True

    # --- Trigger Rerun if Navigation Occurred ---
    if needs_rerun:
        st.rerun()

    # --- Display Selected Episode Details ---
    try:
        # Get data for the currently selected index
        selected_ep_data = loaded_sim_data[st.session_state.current_episode_index]
        actual_ep_num = selected_ep_data.get('episode') # Should exist and be correct now

        # Display episode info header
        term_reason = selected_ep_data.get('termination_reason', 'N/A')
        time_list = extract_trajectory_data(selected_ep_data, 'time')
        reward_list = extract_trajectory_data(selected_ep_data, 'cumulative_reward')
        final_time = time_list[-1] if time_list is not None and len(time_list)>0 and np.isfinite(time_list[-1]) else np.nan
        final_reward = reward_list[-1] if reward_list is not None and len(reward_list)>0 and np.isfinite(reward_list[-1]) else np.nan
        st.write(f"#### Details for Episode: {actual_ep_num}")
        st.info(f"Termination: **{term_reason}** | Final Time: **{final_time:.3f}s** | Final Reward: **{final_reward:.3f}**")

        # --- Display Data Table and Stats ---
        col_table, col_stats = st.columns(2)
        df_episode = None
        with col_table:
            st.write("**Episode Data Table**")
            try:
                # Create DataFrame from episode data (excluding complex objects if any)
                display_data = {k: v for k, v in selected_ep_data.items() if isinstance(v, (list, np.ndarray, int, float, str, bool, np.number))}
                # Find max length of list-like data to pad scalars
                max_len = 0
                for v in display_data.values():
                    if isinstance(v, (list, np.ndarray)): max_len = max(max_len, len(v))
                if max_len == 0 and display_data: max_len = 1 # Handle case with only scalars

                if max_len > 0:
                    df_data_dict = {}
                    for k, v in display_data.items():
                        if isinstance(v, (list, np.ndarray)):
                            current_len = len(v)
                            padded_v = list(v) + [np.nan] * (max_len - current_len) if current_len < max_len else v[:max_len]
                            df_data_dict[k] = padded_v
                        else: # Scalar
                             df_data_dict[k] = [v] * max_len

                    if df_data_dict:
                        df_episode = pd.DataFrame(df_data_dict)
                        # Define preferred column order
                        preferred_cols = ['time', 'pendulum_angle', 'pendulum_velocity', 'cart_position', 'cart_velocity', 'force', 'reward', 'cumulative_reward', 'kp', 'ki', 'kd', 'error', 'epsilon', 'learning_rate', 'action_kp', 'action_ki', 'action_kd', 'termination_reason']
                        existing_preferred = [c for c in preferred_cols if c in df_episode.columns]
                        other_cols = sorted([c for c in df_episode.columns if c not in existing_preferred and c != 'episode'])
                        final_cols = ['episode'] + existing_preferred + other_cols
                        st.dataframe(df_episode[final_cols], height=300)
                    else: st.warning("No suitable data found for table.")
                else: st.warning("No list-like data found in this episode.")
            except Exception as e:
                st.error(f"Error preparing episode data table: {e}")
                logging.error(f"Episode table prep error: {e}", exc_info=True)

        with col_stats:
            st.write("**Summary Statistics**")
            if df_episode is not None and not df_episode.empty:
                try:
                    numeric_df = df_episode.select_dtypes(include=np.number)
                    if not numeric_df.empty:
                         st.dataframe(numeric_df.describe().T.style.format(precision=4))
                    else: st.warning("No numeric columns for statistics.")
                except Exception as e:
                    st.error(f"Error calculating statistics: {e}")
            else:
                st.warning("Data table unavailable for statistics.")

        st.divider()
        # --- Plotting Trajectories ---
        plot_episode_details_graphs_mpl(selected_ep_data, actual_ep_num)

        st.divider()
        # --- Animation ---
        plot_episode_animation(selected_ep_data)

    except IndexError:
        st.error(f"Error accessing episode index {st.session_state.current_episode_index}. Data might not be loaded correctly or index out of bounds.")
        logging.error(f"IndexError accessing loaded_sim_data at index {st.session_state.current_episode_index}")
        gc.collect()
    except Exception as detail_e:
        st.error(f"An unexpected error occurred displaying episode details: {detail_e}")
        logging.error(f"Episode detail display error: {traceback.format_exc()}")
        gc.collect()


def variables_discovery():
    """Placeholder tab for future analysis configuration."""
    st.subheader("Analysis Configurator (Future Feature)")
    st.info("This section is planned for configuring more advanced analysis tasks.")
    st.write("Potential features:")
    st.markdown("- Define variable sets for correlation analysis.")
    st.markdown("- Configure feature importance calculations (e.g., relative to reward).")
    st.markdown("- Set up clustering analyses based on trajectory features.")
    st.selectbox("Example Analysis Type:", ["Correlation Matrix", "Feature Importance (Random Forest)", "Trajectory Clustering (KMeans)"], disabled=True)
    st.multiselect("Example Variables:", ["pendulum_angle", "cart_velocity", "kp_mean", "total_reward", "episode_time"], default=["pendulum_angle", "total_reward"], disabled=True)
    st.button("Generate Analysis Config (Inactive)", disabled=True)

# =============================================================================
# == Main Application Logic ==
# =============================================================================
def main():
    """Sets up the Streamlit page and orchestrates the application flow."""
    st.set_page_config(page_title="RL Control Dashboard", page_icon="🤖", layout="wide")

    # Critical check for utilities import
    if not utils_imports_ok:
        st.error("Application cannot start due to missing utility functions. Please check the console logs and ensure `app_utils.py` is correct.")
        st.stop() # Halt execution if utils failed to import

    st.markdown("<h1 style='text-align: center; color: #023047;'><b>Dynamic System Simulation Dashboard</b></h1>", unsafe_allow_html=True)

    # Initialize session state if it's the first run
    initialize_session_state()

    # Render the sidebar (handles folder selection, filtering, load triggers)
    render_sidebar()

    # --- Main Content Area ---
    # Define tabs
    tabs = ["Introduction & Config", "Performance Overview", "Episode Details", "Analysis Configurator"]
    icons = ['house-door', 'bar-chart-line', 'search-heart', 'gear'] # Choose appropriate FontAwesome icons

    # Create horizontal navigation menu
    selected_page = option_menu(
        menu_title=None, # No main title for the menu
        options=tabs,
        icons=icons,
        menu_icon='cast', # Optional menu icon
        default_index=0,
        orientation='horizontal',
        styles={ # Custom styling (optional)
            "container": {"padding": "5px !important", "background-color": "#f0f2f6", "border-radius": "5px"},
            "icon": {"color": "#023047", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px", "text-align": "center", "margin":"0px 5px",
                "--hover-color": "#dfe6e9", "color": "#023047", "border-radius": "5px",
                "padding": "10px 15px"
            },
            "nav-link-selected": {"background-color": "#219ebc", "color": "white", "font-weight": "bold"},
        }
    )

    # --- Display Content Based on Selected Tab ---
    # Only show tab content if a folder has been successfully selected
    if st.session_state.get('selected_folder_path'):
        if selected_page == "Introduction & Config":
            introduction_page()
        elif selected_page == "Performance Overview":
            performance_overview()
        elif selected_page == "Episode Details":
            episode_details()
        elif selected_page == "Analysis Configurator":
            variables_discovery()
    else:
        # Prompt user to select a folder if none is selected yet
        st.info("👋 **Welcome!** Please select a simulation results folder from the sidebar to begin.")
        st.caption(f"Ensure your results folders are located within the `{RESULTS_FOLDER}` directory.")

if __name__ == "__main__":
    main()