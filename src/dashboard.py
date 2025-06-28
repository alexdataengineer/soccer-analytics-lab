"""
Streamlit Dashboard for Soccer Match Analysis
Interactive web interface for viewing match insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import FieldVisualizer, HeatmapVisualizer, PossessionVisualizer, MovementVisualizer


def load_analysis_data(data_dir):
    """Load analysis results from directory."""
    data_path = Path(data_dir)
    
    # Look for analysis files
    analysis_file = None
    for file in data_path.glob("*.json"):
        if "analysis" in file.name.lower():
            analysis_file = file
            break
    
    if not analysis_file:
        st.error(f"No analysis results found in {data_dir}")
        return None
    
    try:
        with open(analysis_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading analysis data: {e}")
        return None


def create_possession_chart(possession_stats):
    """Create possession pie chart using Plotly."""
    labels = ['Left Team', 'Right Team']
    values = [possession_stats['left_team_possession'], possession_stats['right_team_possession']]
    colors = ['#1f77b4', '#ff7f0e']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, marker_colors=colors)])
    fig.update_layout(
        title="Ball Possession Distribution",
        showlegend=True,
        height=400
    )
    
    return fig


def create_possession_timeline(possession_periods):
    """Create possession timeline using Plotly."""
    if not possession_periods:
        return None
    
    # Prepare data
    times = []
    teams = []
    colors = []
    
    for period in possession_periods:
        start_time = period['start_time']
        end_time = period.get('end_time', start_time + period['duration'] / 30.0)
        
        times.extend([start_time, end_time])
        teams.extend([period['team'], period['team']])
        colors.extend(['blue' if period['team'] == 'left' else 'red'] * 2)
    
    fig = go.Figure()
    
    # Add timeline segments
    for i in range(0, len(times), 2):
        if i + 1 < len(times):
            fig.add_trace(go.Scatter(
                x=[times[i], times[i+1]],
                y=[teams[i], teams[i]],
                mode='lines',
                line=dict(color=colors[i], width=10),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="Ball Possession Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Team",
        height=300,
        yaxis=dict(
            ticktext=['Left Team', 'Right Team'],
            tickvals=['left', 'right']
        )
    )
    
    return fig


def create_field_heatmap(team_positions, team_name, field_width=105, field_height=68):
    """Create field heatmap using Plotly."""
    if not team_positions:
        return None
    
    # Create field outline
    field_outline = [
        [0, 0], [field_width, 0], [field_width, field_height], [0, field_height], [0, 0]
    ]
    
    # Create heatmap data
    positions_array = np.array(team_positions)
    x = positions_array[:, 0]
    y = positions_array[:, 1]
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=20, 
                                            range=[[0, field_width], [0, field_height]])
    
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=heatmap.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='Viridis',
        showscale=True,
        name='Position Frequency'
    ))
    
    # Add field outline
    fig.add_trace(go.Scatter(
        x=[p[0] for p in field_outline],
        y=[p[1] for p in field_outline],
        mode='lines',
        line=dict(color='white', width=3),
        name='Field Outline',
        showlegend=False
    ))
    
    # Add center line
    fig.add_trace(go.Scatter(
        x=[field_width/2, field_width/2],
        y=[0, field_height],
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{team_name} Field Occupation Heatmap",
        xaxis_title="Field Width (m)",
        yaxis_title="Field Height (m)",
        height=500,
        xaxis=dict(range=[0, field_width]),
        yaxis=dict(range=[0, field_height]),
        plot_bgcolor='green',
        paper_bgcolor='green'
    )
    
    return fig


def create_movement_stats(movement_stats):
    """Create movement statistics charts."""
    if 'player_details' not in movement_stats:
        return None
    
    player_data = movement_stats['player_details']
    
    # Prepare data
    distances = [data['total_distance'] for data in player_data.values()]
    speeds = [data['avg_speed'] for data in player_data.values()]
    areas = [data['area_covered'] for data in player_data.values()]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distance Distribution', 'Speed Distribution', 
                       'Area Coverage', 'Track Length'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # Distance histogram
    fig.add_trace(
        go.Histogram(x=distances, name='Distance', nbinsx=20, marker_color='skyblue'),
        row=1, col=1
    )
    
    # Speed histogram
    fig.add_trace(
        go.Histogram(x=speeds, name='Speed', nbinsx=20, marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Area histogram
    fig.add_trace(
        go.Histogram(x=areas, name='Area', nbinsx=20, marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Track length histogram
    lengths = [data['track_length'] for data in player_data.values()]
    fig.add_trace(
        go.Histogram(x=lengths, name='Track Length', nbinsx=20, marker_color='gold'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Player Movement Statistics",
        height=600,
        showlegend=False
    )
    
    return fig


def create_player_trajectories(player_tracks, max_players=5, field_width=105, field_height=68):
    """Create player trajectory visualization."""
    if not player_tracks:
        return None
    
    fig = go.Figure()
    
    # Field outline
    field_outline = [
        [0, 0], [field_width, 0], [field_width, field_height], [0, field_height], [0, 0]
    ]
    
    fig.add_trace(go.Scatter(
        x=[p[0] for p in field_outline],
        y=[p[1] for p in field_outline],
        mode='lines',
        line=dict(color='white', width=3),
        name='Field Outline',
        showlegend=False
    ))
    
    # Add center line
    fig.add_trace(go.Scatter(
        x=[field_width/2, field_width/2],
        y=[0, field_height],
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))
    
    # Plot trajectories for selected players
    colors = px.colors.qualitative.Set1
    for i, (track_id, track) in enumerate(list(player_tracks.items())[:max_players]):
        if len(track) < 2:
            continue
        
        positions = []
        for detection in track:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            positions.append([center_x, center_y])
        
        positions_array = np.array(positions)
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=positions_array[:, 0],
            y=positions_array[:, 1],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=3),
            name=f'Player {track_id}',
            showlegend=True
        ))
        
        # Add start and end markers
        fig.add_trace(go.Scatter(
            x=[positions_array[0, 0]],
            y=[positions_array[0, 1]],
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=10, symbol='circle'),
            name=f'Player {track_id} Start',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[positions_array[-1, 0]],
            y=[positions_array[-1, 1]],
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=10, symbol='square'),
            name=f'Player {track_id} End',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Player Movement Trajectories",
        xaxis_title="Field Width (m)",
        yaxis_title="Field Height (m)",
        height=600,
        xaxis=dict(range=[0, field_width]),
        yaxis=dict(range=[0, field_height]),
        plot_bgcolor='green',
        paper_bgcolor='green'
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Soccer Match Insights",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚽ Soccer Match Insights Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data directory selection
    data_dir = st.sidebar.text_input(
        "Analysis Results Directory",
        value="outputs",
        help="Directory containing analysis results"
    )
    
    # Load data
    analysis_data = load_analysis_data(data_dir)
    
    if analysis_data is None:
        st.warning("Please provide a valid analysis results directory in the sidebar.")
        return
    
    # Main dashboard
    st.header("Match Analysis Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'possession' in analysis_data:
            left_possession = analysis_data['possession'].get('left_team_possession', 0)
            st.metric("Left Team Possession", f"{left_possession:.1%}")
    
    with col2:
        if 'possession' in analysis_data:
            right_possession = analysis_data['possession'].get('right_team_possession', 0)
            st.metric("Right Team Possession", f"{right_possession:.1%}")
    
    with col3:
        if 'movement' in analysis_data:
            total_players = analysis_data['movement'].get('total_players', 0)
            st.metric("Players Tracked", total_players)
    
    with col4:
        if 'movement' in analysis_data:
            avg_distance = analysis_data['movement'].get('avg_distance', 0)
            st.metric("Avg Distance", f"{avg_distance:.0f} px")
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Possession", "Field Occupation", "Movement", "Raw Data"])
    
    with tab1:
        st.header("Ball Possession Analysis")
        
        if 'possession' in analysis_data:
            possession_stats = analysis_data['possession']
            
            # Possession pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                possession_chart = create_possession_chart(possession_stats)
                st.plotly_chart(possession_chart, use_container_width=True)
            
            with col2:
                st.subheader("Possession Statistics")
                st.write(f"**Total Possession Time:** {possession_stats.get('total_possession_time', 0):.0f} frames")
                st.write(f"**Possession Periods:** {possession_stats.get('possession_periods', 0)}")
                st.write(f"**Average Duration:** {possession_stats.get('avg_possession_duration', 0):.1f} frames")
            
            # Possession timeline
            if 'possession_periods' in analysis_data:
                st.subheader("Possession Timeline")
                timeline_chart = create_possession_timeline(analysis_data['possession_periods'])
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
    
    with tab2:
        st.header("Field Occupation Analysis")
        
        if 'raw_data' in analysis_data and 'player_tracks' in analysis_data['raw_data']:
            player_tracks = analysis_data['raw_data']['player_tracks']
            
            # Extract team positions
            left_team_positions = []
            right_team_positions = []
            
            for track_id, track in player_tracks.items():
                for detection in track:
                    x1, y1, x2, y2 = detection['bbox']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    if center_x < 52.5:  # Assuming field width of 105m
                        left_team_positions.append([center_x, center_y])
                    else:
                        right_team_positions.append([center_x, center_y])
            
            # Create heatmaps
            col1, col2 = st.columns(2)
            
            with col1:
                left_heatmap = create_field_heatmap(left_team_positions, "Left Team")
                if left_heatmap:
                    st.plotly_chart(left_heatmap, use_container_width=True)
            
            with col2:
                right_heatmap = create_field_heatmap(right_team_positions, "Right Team")
                if right_heatmap:
                    st.plotly_chart(right_heatmap, use_container_width=True)
    
    with tab3:
        st.header("Player Movement Analysis")
        
        if 'movement' in analysis_data:
            movement_stats = analysis_data['movement']
            
            # Movement statistics
            st.subheader("Movement Statistics")
            movement_chart = create_movement_stats(movement_stats)
            if movement_chart:
                st.plotly_chart(movement_chart, use_container_width=True)
            
            # Player trajectories
            if 'raw_data' in analysis_data and 'player_tracks' in analysis_data['raw_data']:
                st.subheader("Player Trajectories")
                max_players = st.slider("Number of players to show", 1, 10, 5)
                trajectory_chart = create_player_trajectories(
                    analysis_data['raw_data']['player_tracks'], 
                    max_players
                )
                if trajectory_chart:
                    st.plotly_chart(trajectory_chart, use_container_width=True)
    
    with tab4:
        st.header("Raw Data")
        
        # Display raw data as JSON
        st.subheader("Analysis Results")
        st.json(analysis_data)
        
        # Download button
        st.download_button(
            label="Download Analysis Results (JSON)",
            data=json.dumps(analysis_data, indent=2),
            file_name="match_analysis_results.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main() 