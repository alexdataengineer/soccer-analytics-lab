"""
Soccer Match Analysis Module
Calculates match statistics from tracking data.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns


class PossessionAnalyzer:
    def __init__(self, field_width=105, field_height=68):
        """
        Initialize possession analyzer.
        
        Args:
            field_width (float): Field width in meters
            field_height (float): Field height in meters
        """
        self.field_width = field_width
        self.field_height = field_height
        self.possession_data = []
        
    def calculate_ball_center(self, bbox):
        """Calculate ball center from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def find_closest_player(self, ball_pos, players):
        """Find the player closest to the ball."""
        if not players:
            return None, float('inf')
        
        min_distance = float('inf')
        closest_player = None
        
        for player in players:
            player_center = self.calculate_player_center(player['bbox'])
            distance = euclidean(ball_pos, player_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player
        
        return closest_player, min_distance
    
    def calculate_player_center(self, bbox):
        """Calculate player center from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def analyze_possession(self, ball_positions, player_tracks, possession_threshold=3.0):
        """
        Analyze ball possession based on player proximity to ball.
        
        Args:
            ball_positions (list): List of ball position data
            player_tracks (dict): Dictionary of player tracks
            possession_threshold (float): Distance threshold for possession (in pixels)
        """
        possession_periods = []
        current_possession = None
        
        for ball_pos in ball_positions:
            frame_id = ball_pos['frame_id']
            ball_center = self.calculate_ball_center(ball_pos['bbox'])
            
            # Find all players in this frame
            frame_players = []
            for track_id, track in player_tracks.items():
                for detection in track:
                    if detection['frame_id'] == frame_id:
                        frame_players.append(detection)
                        break
            
            # Find closest player to ball
            closest_player, distance = self.find_closest_player(ball_center, frame_players)
            
            if closest_player and distance < possession_threshold:
                # Determine team (simplified: left side vs right side)
                player_center = self.calculate_player_center(closest_player['bbox'])
                team = 'left' if player_center[0] < self.field_width / 2 else 'right'
                
                if current_possession is None or current_possession['team'] != team:
                    # End current possession period
                    if current_possession:
                        current_possession['end_frame'] = frame_id
                        current_possession['duration'] = frame_id - current_possession['start_frame']
                        possession_periods.append(current_possession)
                    
                    # Start new possession period
                    current_possession = {
                        'team': team,
                        'start_frame': frame_id,
                        'start_time': frame_id / 30.0,  # Assuming 30 FPS
                        'player_center': player_center,
                        'ball_center': ball_center
                    }
            else:
                # Ball not in possession
                if current_possession:
                    current_possession['end_frame'] = frame_id
                    current_possession['duration'] = frame_id - current_possession['start_frame']
                    possession_periods.append(current_possession)
                    current_possession = None
        
        # Handle final possession period
        if current_possession:
            current_possession['end_frame'] = ball_positions[-1]['frame_id']
            current_possession['duration'] = current_possession['end_frame'] - current_possession['start_frame']
            possession_periods.append(current_possession)
        
        self.possession_data = possession_periods
        return possession_periods
    
    def get_possession_statistics(self):
        """Calculate possession statistics."""
        if not self.possession_data:
            return {
                'left_team_possession': 0,
                'right_team_possession': 0,
                'total_possession_time': 0,
                'possession_periods': 0
            }
        
        left_team_time = sum(p['duration'] for p in self.possession_data if p['team'] == 'left')
        right_team_time = sum(p['duration'] for p in self.possession_data if p['team'] == 'right')
        total_time = left_team_time + right_team_time
        
        return {
            'left_team_possession': left_team_time / total_time if total_time > 0 else 0,
            'right_team_possession': right_team_time / total_time if total_time > 0 else 0,
            'total_possession_time': total_time,
            'possession_periods': len(self.possession_data),
            'avg_possession_duration': np.mean([p['duration'] for p in self.possession_data]) if self.possession_data else 0
        }


class FieldOccupationAnalyzer:
    def __init__(self, field_width=105, field_height=68, grid_size=10):
        """
        Initialize field occupation analyzer.
        
        Args:
            field_width (float): Field width in meters
            field_height (float): Field height in meters
            grid_size (int): Size of grid cells for heatmap
        """
        self.field_width = field_width
        self.field_height = field_height
        self.grid_size = grid_size
        self.occupation_data = {
            'left_team': [],
            'right_team': []
        }
        
    def calculate_player_center(self, bbox):
        """Calculate player center from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def analyze_field_occupation(self, player_tracks):
        """
        Analyze field occupation patterns for each team.
        
        Args:
            player_tracks (dict): Dictionary of player tracks
        """
        left_team_positions = []
        right_team_positions = []
        
        for track_id, track in player_tracks.items():
            for detection in track:
                player_center = self.calculate_player_center(detection['bbox'])
                
                # Determine team based on x-position
                if player_center[0] < self.field_width / 2:
                    left_team_positions.append(player_center)
                else:
                    right_team_positions.append(player_center)
        
        self.occupation_data['left_team'] = left_team_positions
        self.occupation_data['right_team'] = right_team_positions
        
        return self.occupation_data
    
    def create_heatmap_data(self, team_positions):
        """Create heatmap data from team positions."""
        if not team_positions:
            return np.zeros((self.grid_size, self.grid_size))
        
        # Create grid
        x_bins = np.linspace(0, self.field_width, self.grid_size + 1)
        y_bins = np.linspace(0, self.field_height, self.grid_size + 1)
        
        # Count positions in each grid cell
        heatmap = np.zeros((self.grid_size, self.grid_size))
        
        for pos in team_positions:
            x, y = pos
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1
            
            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                heatmap[y_idx, x_idx] += 1
        
        return heatmap
    
    def get_occupation_statistics(self):
        """Calculate field occupation statistics."""
        left_heatmap = self.create_heatmap_data(self.occupation_data['left_team'])
        right_heatmap = self.create_heatmap_data(self.occupation_data['right_team'])
        
        return {
            'left_team_positions': len(self.occupation_data['left_team']),
            'right_team_positions': len(self.occupation_data['right_team']),
            'left_team_heatmap': left_heatmap.tolist(),
            'right_team_heatmap': right_heatmap.tolist(),
            'left_team_density': np.sum(left_heatmap) / (self.grid_size * self.grid_size),
            'right_team_density': np.sum(right_heatmap) / (self.grid_size * self.grid_size)
        }


class MovementAnalyzer:
    def __init__(self):
        """Initialize movement analyzer."""
        self.movement_data = {}
        
    def analyze_player_movement(self, player_tracks):
        """
        Analyze player movement patterns.
        
        Args:
            player_tracks (dict): Dictionary of player tracks
        """
        for track_id, track in player_tracks.items():
            if len(track) < 2:
                continue
            
            # Calculate movement statistics
            total_distance = 0
            speeds = []
            positions = []
            
            for i in range(1, len(track)):
                prev_pos = self.calculate_player_center(track[i-1]['bbox'])
                curr_pos = self.calculate_player_center(track[i]['bbox'])
                
                distance = euclidean(prev_pos, curr_pos)
                total_distance += distance
                
                # Calculate speed (pixels per frame)
                frame_diff = track[i]['frame_id'] - track[i-1]['frame_id']
                if frame_diff > 0:
                    speed = distance / frame_diff
                    speeds.append(speed)
                
                positions.append(curr_pos)
            
            # Calculate additional metrics
            if positions:
                # Calculate area covered (convex hull approximation)
                positions_array = np.array(positions)
                x_range = np.max(positions_array[:, 0]) - np.min(positions_array[:, 0])
                y_range = np.max(positions_array[:, 1]) - np.min(positions_array[:, 1])
                area_covered = x_range * y_range
                
                # Calculate average speed
                avg_speed = np.mean(speeds) if speeds else 0
                max_speed = np.max(speeds) if speeds else 0
                
                self.movement_data[track_id] = {
                    'total_distance': total_distance,
                    'avg_speed': avg_speed,
                    'max_speed': max_speed,
                    'area_covered': area_covered,
                    'track_length': len(track),
                    'positions': positions
                }
        
        return self.movement_data
    
    def calculate_player_center(self, bbox):
        """Calculate player center from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_movement_statistics(self):
        """Calculate overall movement statistics."""
        if not self.movement_data:
            return {
                'total_players': 0,
                'avg_distance': 0,
                'avg_speed': 0,
                'total_distance': 0
            }
        
        distances = [data['total_distance'] for data in self.movement_data.values()]
        speeds = [data['avg_speed'] for data in self.movement_data.values()]
        
        return {
            'total_players': len(self.movement_data),
            'avg_distance': np.mean(distances),
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds) if speeds else 0,
            'total_distance': np.sum(distances),
            'player_details': self.movement_data
        }


class MatchAnalyzer:
    def __init__(self):
        """Initialize match analyzer with all analysis components."""
        self.possession_analyzer = PossessionAnalyzer()
        self.occupation_analyzer = FieldOccupationAnalyzer()
        self.movement_analyzer = MovementAnalyzer()
        self.analysis_results = {}
        
    def analyze_match(self, tracking_results_dir):
        """
        Perform complete match analysis.
        
        Args:
            tracking_results_dir (str): Directory containing tracking results
        """
        tracking_dir = Path(tracking_results_dir)
        
        # Load tracking data
        with open(tracking_dir / "player_tracks.json", 'r') as f:
            player_tracks = json.load(f)
        
        with open(tracking_dir / "ball_positions.json", 'r') as f:
            ball_positions = json.load(f)
        
        print(f"Analyzing match with {len(player_tracks)} player tracks and {len(ball_positions)} ball positions...")
        
        # Perform analyses
        possession_data = self.possession_analyzer.analyze_possession(ball_positions, player_tracks)
        occupation_data = self.occupation_analyzer.analyze_field_occupation(player_tracks)
        movement_data = self.movement_analyzer.analyze_player_movement(player_tracks)
        
        # Compile results
        self.analysis_results = {
            'possession': self.possession_analyzer.get_possession_statistics(),
            'field_occupation': self.occupation_analyzer.get_occupation_statistics(),
            'movement': self.movement_analyzer.get_movement_statistics(),
            'possession_periods': possession_data,
            'raw_data': {
                'player_tracks': player_tracks,
                'ball_positions': ball_positions
            }
        }
        
        return self.analysis_results
    
    def save_analysis_results(self, output_dir):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar types
                return obj.item()
            else:
                return obj
        
        # Save main results
        results_file = output_path / "match_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(convert_numpy_types(self.analysis_results), f, indent=2)
        
        # Save summary
        summary = {
            'possession': self.analysis_results['possession'],
            'field_occupation': {
                'left_team_positions': self.analysis_results['field_occupation']['left_team_positions'],
                'right_team_positions': self.analysis_results['field_occupation']['right_team_positions']
            },
            'movement': {
                'total_players': self.analysis_results['movement']['total_players'],
                'avg_distance': self.analysis_results['movement']['avg_distance'],
                'avg_speed': self.analysis_results['movement']['avg_speed']
            }
        }
        
        summary = convert_numpy_types(summary)
        
        summary_file = output_path / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis results saved to: {output_path}")
        print(f"Full results: {results_file}")
        print(f"Summary: {summary_file}")
        
        return summary


def main():
    """Example usage of the analysis module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Soccer Match Analysis')
    parser.add_argument('--tracking-results', required=True, help='Directory with tracking results')
    parser.add_argument('--output', default='outputs', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MatchAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze_match(args.tracking_results)
    
    # Save results
    summary = analyzer.save_analysis_results(args.output)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Left team possession: {summary['possession']['left_team_possession']:.2%}")
    print(f"Right team possession: {summary['possession']['right_team_possession']:.2%}")
    print(f"Total players analyzed: {summary['movement']['total_players']}")
    print(f"Average player distance: {summary['movement']['avg_distance']:.2f} pixels")


if __name__ == "__main__":
    main() 