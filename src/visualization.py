"""
Soccer Match Visualization Module
Creates visualizations for match analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2


class FieldVisualizer:
    def __init__(self, field_width=105, field_height=68):
        """
        Initialize field visualizer.
        
        Args:
            field_width (float): Field width in meters
            field_height (float): Field height in meters
        """
        self.field_width = field_width
        self.field_height = field_height
        
    def create_field_background(self, ax):
        """Create soccer field background."""
        # Field outline
        ax.add_patch(plt.Rectangle((0, 0), self.field_width, self.field_height, 
                                  fill=False, color='white', linewidth=2))
        
        # Center line
        ax.axvline(x=self.field_width/2, color='white', linewidth=2)
        
        # Center circle
        center_circle = plt.Circle((self.field_width/2, self.field_height/2), 9.15, 
                                  fill=False, color='white', linewidth=2)
        ax.add_patch(center_circle)
        
        # Penalty areas
        # Left penalty area
        ax.add_patch(plt.Rectangle((0, self.field_height/2 - 20.15), 16.5, 40.3, 
                                  fill=False, color='white', linewidth=2))
        # Right penalty area
        ax.add_patch(plt.Rectangle((self.field_width - 16.5, self.field_height/2 - 20.15), 16.5, 40.3, 
                                  fill=False, color='white', linewidth=2))
        
        # Goal areas
        # Left goal area
        ax.add_patch(plt.Rectangle((0, self.field_height/2 - 7.32/2 - 5.5), 5.5, 7.32 + 11, 
                                  fill=False, color='white', linewidth=2))
        # Right goal area
        ax.add_patch(plt.Rectangle((self.field_width - 5.5, self.field_height/2 - 7.32/2 - 5.5), 5.5, 7.32 + 11, 
                                  fill=False, color='white', linewidth=2))
        
        # Set field properties
        ax.set_xlim(0, self.field_width)
        ax.set_ylim(0, self.field_height)
        ax.set_aspect('equal')
        ax.set_facecolor('green')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


class HeatmapVisualizer:
    def __init__(self, field_width=105, field_height=68):
        """Initialize heatmap visualizer."""
        self.field_width = field_width
        self.field_height = field_height
        self.field_viz = FieldVisualizer(field_width, field_height)
        
    def create_team_heatmap(self, team_positions, team_name, output_path=None):
        """
        Create heatmap for a team's field occupation.
        
        Args:
            team_positions (list): List of (x, y) positions
            team_name (str): Name of the team
            output_path (str): Path to save the plot
        """
        if not team_positions:
            print(f"No positions data for {team_name}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create field background
        self.field_viz.create_field_background(ax)
        
        # Create heatmap data
        positions_array = np.array(team_positions)
        x = positions_array[:, 0]
        y = positions_array[:, 1]
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=20, 
                                                range=[[0, self.field_width], [0, self.field_height]])
        
        # Plot heatmap
        extent = [0, self.field_width, 0, self.field_height]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                      cmap='hot', alpha=0.7, aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Position Frequency', rotation=270, labelpad=20)
        
        # Add title
        ax.set_title(f'{team_name} Field Occupation Heatmap', fontsize=16, color='white')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"Heatmap saved to: {output_path}")
        
        plt.show()
        
    def create_comparison_heatmap(self, left_team_positions, right_team_positions, output_path=None):
        """
        Create comparison heatmap for both teams.
        
        Args:
            left_team_positions (list): Left team positions
            right_team_positions (list): Right team positions
            output_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left team heatmap
        self.field_viz.create_field_background(ax1)
        if left_team_positions:
            positions_array = np.array(left_team_positions)
            heatmap1, _, _ = np.histogram2d(positions_array[:, 0], positions_array[:, 1], 
                                          bins=20, range=[[0, self.field_width], [0, self.field_height]])
            im1 = ax1.imshow(heatmap1.T, extent=[0, self.field_width, 0, self.field_height], 
                           origin='lower', cmap='Blues', alpha=0.7, aspect='equal')
            plt.colorbar(im1, ax=ax1)
        ax1.set_title('Left Team Field Occupation', fontsize=14, color='white')
        
        # Right team heatmap
        self.field_viz.create_field_background(ax2)
        if right_team_positions:
            positions_array = np.array(right_team_positions)
            heatmap2, _, _ = np.histogram2d(positions_array[:, 0], positions_array[:, 1], 
                                          bins=20, range=[[0, self.field_width], [0, self.field_height]])
            im2 = ax2.imshow(heatmap2.T, extent=[0, self.field_width, 0, self.field_height], 
                           origin='lower', cmap='Reds', alpha=0.7, aspect='equal')
            plt.colorbar(im2, ax=ax2)
        ax2.set_title('Right Team Field Occupation', fontsize=14, color='white')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"Comparison heatmap saved to: {output_path}")
        
        plt.show()


class PossessionVisualizer:
    def __init__(self):
        """Initialize possession visualizer."""
        pass
        
    def create_possession_timeline(self, possession_periods, output_path=None):
        """
        Create timeline visualization of possession changes.
        
        Args:
            possession_periods (list): List of possession period data
            output_path (str): Path to save the plot
        """
        if not possession_periods:
            print("No possession data available")
            return
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Create timeline data
        times = []
        teams = []
        
        for period in possession_periods:
            start_time = period['start_time']
            end_time = period['end_time'] if 'end_time' in period else start_time + period['duration'] / 30.0
            
            times.extend([start_time, end_time])
            teams.extend([period['team'], period['team']])
        
        # Create color mapping
        colors = {'left': 'blue', 'right': 'red'}
        
        # Plot possession periods
        for i in range(0, len(times), 2):
            if i + 1 < len(times):
                team = teams[i]
                ax.axvspan(times[i], times[i+1], alpha=0.3, color=colors[team], label=team if i == 0 else "")
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Possession')
        ax.set_title('Ball Possession Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Possession timeline saved to: {output_path}")
        
        plt.show()
    
    def create_possession_pie_chart(self, possession_stats, output_path=None):
        """
        Create pie chart of possession percentages.
        
        Args:
            possession_stats (dict): Possession statistics
            output_path (str): Path to save the plot
        """
        sizes = [possession_stats['left_team_possession'], possession_stats['right_team_possession']]
        if sum(sizes) == 0:
            print("No possession data available to plot pie chart.")
            if output_path:
                with open(str(output_path) + '.txt', 'w') as f:
                    f.write('No possession data available to plot pie chart.')
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = ['Left Team', 'Right Team']
        colors = ['lightblue', 'lightcoral']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90)
        
        ax.set_title('Ball Possession Distribution', fontsize=16)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Possession pie chart saved to: {output_path}")
        
        plt.show()


class MovementVisualizer:
    def __init__(self, field_width=105, field_height=68):
        """Initialize movement visualizer."""
        self.field_width = field_width
        self.field_height = field_height
        self.field_viz = FieldVisualizer(field_width, field_height)
        
    def create_player_trajectories(self, player_tracks, max_players=5, output_path=None):
        """
        Create visualization of player movement trajectories.
        
        Args:
            player_tracks (dict): Dictionary of player tracks
            max_players (int): Maximum number of players to show
            output_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create field background
        self.field_viz.create_field_background(ax)
        
        # Plot trajectories for selected players
        colors = plt.cm.tab10(np.linspace(0, 1, min(max_players, len(player_tracks))))
        
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
            ax.plot(positions_array[:, 0], positions_array[:, 1], 
                   color=colors[i], linewidth=2, alpha=0.7, label=f'Player {track_id}')
            
            # Mark start and end points
            ax.scatter(positions_array[0, 0], positions_array[0, 1], 
                      color=colors[i], s=100, marker='o', edgecolors='white', linewidth=2)
            ax.scatter(positions_array[-1, 0], positions_array[-1, 1], 
                      color=colors[i], s=100, marker='s', edgecolors='white', linewidth=2)
        
        ax.set_title('Player Movement Trajectories', fontsize=16, color='white')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"Player trajectories saved to: {output_path}")
        
        plt.show()
    
    def create_movement_statistics(self, movement_stats, output_path=None):
        """
        Create bar charts of movement statistics.
        
        Args:
            movement_stats (dict): Movement statistics
            output_path (str): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distance distribution
        if 'player_details' in movement_stats:
            distances = [data['total_distance'] for data in movement_stats['player_details'].values()]
            ax1.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Total Distance (pixels)')
            ax1.set_ylabel('Number of Players')
            ax1.set_title('Distance Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Speed distribution
        if 'player_details' in movement_stats:
            speeds = [data['avg_speed'] for data in movement_stats['player_details'].values()]
            ax2.hist(speeds, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_xlabel('Average Speed (pixels/frame)')
            ax2.set_ylabel('Number of Players')
            ax2.set_title('Speed Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Area covered distribution
        if 'player_details' in movement_stats:
            areas = [data['area_covered'] for data in movement_stats['player_details'].values()]
            ax3.hist(areas, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('Area Covered (pixels²)')
            ax3.set_ylabel('Number of Players')
            ax3.set_title('Area Coverage Distribution')
            ax3.grid(True, alpha=0.3)
        
        # Track length distribution
        if 'player_details' in movement_stats:
            lengths = [data['track_length'] for data in movement_stats['player_details'].values()]
            ax4.hist(lengths, bins=20, alpha=0.7, color='gold', edgecolor='black')
            ax4.set_xlabel('Track Length (frames)')
            ax4.set_ylabel('Number of Players')
            ax4.set_title('Track Length Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Movement statistics saved to: {output_path}")
        
        plt.show()


class DashboardVisualizer:
    def __init__(self, field_width=105, field_height=68):
        """Initialize dashboard visualizer."""
        self.field_width = field_width
        self.field_height = field_height
        self.heatmap_viz = HeatmapVisualizer(field_width, field_height)
        self.possession_viz = PossessionVisualizer()
        self.movement_viz = MovementVisualizer(field_width, field_height)
        
    def create_comprehensive_dashboard(self, analysis_results, output_dir):
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            analysis_results (dict): Complete analysis results
            output_dir (str): Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Creating comprehensive dashboard...")
        
        # 1. Field occupation heatmaps
        if 'field_occupation' in analysis_results:
            field_data = analysis_results['field_occupation']
            left_positions = analysis_results['raw_data']['player_tracks']
            right_positions = analysis_results['raw_data']['player_tracks']
            
            # Extract positions from tracks
            left_team_positions = []
            right_team_positions = []
            
            for track_id, track in left_positions.items():
                for detection in track:
                    x1, y1, x2, y2 = detection['bbox']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    if center_x < self.field_width / 2:
                        left_team_positions.append([center_x, center_y])
                    else:
                        right_team_positions.append([center_x, center_y])
            
            self.heatmap_viz.create_comparison_heatmap(
                left_team_positions, right_team_positions,
                output_path / "field_occupation_comparison.png"
            )
        
        # 2. Possession visualizations
        if 'possession' in analysis_results:
            possession_stats = analysis_results['possession']
            self.possession_viz.create_possession_pie_chart(
                possession_stats,
                output_path / "possession_distribution.png"
            )
            
            if 'possession_periods' in analysis_results:
                self.possession_viz.create_possession_timeline(
                    analysis_results['possession_periods'],
                    output_path / "possession_timeline.png"
                )
        
        # 3. Movement visualizations
        if 'movement' in analysis_results and 'raw_data' in analysis_results:
            movement_stats = analysis_results['movement']
            player_tracks = analysis_results['raw_data']['player_tracks']
            
            self.movement_viz.create_player_trajectories(
                player_tracks,
                output_path=output_path / "player_trajectories.png"
            )
            
            self.movement_viz.create_movement_statistics(
                movement_stats,
                output_path=output_path / "movement_statistics.png"
            )
        
        print(f"Dashboard created successfully in: {output_path}")


def main():
    """Example usage of the visualization module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Soccer Match Visualization')
    parser.add_argument('--analysis-results', required=True, help='Path to analysis results JSON file')
    parser.add_argument('--output', default='outputs/visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load analysis results
    with open(args.analysis_results, 'r') as f:
        analysis_results = json.load(f)
    
    # Create dashboard
    dashboard = DashboardVisualizer()
    dashboard.create_comprehensive_dashboard(analysis_results, args.output)


if __name__ == "__main__":
    main() 