"""
Soccer Match Tracking Module
Tracks players and ball across frames using detection results.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from collections import defaultdict
import json
from pathlib import Path
from typing import List, Dict, Tuple


class PlayerTracker:
    def __init__(self, max_distance=100, min_track_length=10):
        """
        Initialize player tracker.
        
        Args:
            max_distance (float): Maximum distance to associate detections between frames
            min_track_length (int): Minimum number of frames for a valid track
        """
        self.max_distance = max_distance
        self.min_track_length = min_track_length
        self.tracks = {}  # track_id -> list of detections
        self.next_track_id = 0
        
    def get_bbox_center(self, bbox):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes."""
        center1 = self.get_bbox_center(bbox1)
        center2 = self.get_bbox_center(bbox2)
        return euclidean(center1, center2)
    
    def update_tracks(self, frame_detections):
        """
        Update tracks with new frame detections.
        
        Args:
            frame_detections (dict): Detection results for current frame
        """
        current_players = frame_detections['players']
        frame_id = frame_detections['frame_id']
        
        # If no tracks exist, create new ones
        if not self.tracks:
            for player in current_players:
                self.tracks[self.next_track_id] = [{
                    'frame_id': frame_id,
                    'bbox': player['bbox'],
                    'confidence': player['confidence']
                }]
                self.next_track_id += 1
            return
        
        # Match current detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # Find best matches
        for track_id, track in self.tracks.items():
            if not track:  # Skip empty tracks
                continue
                
            last_detection = track[-1]
            best_distance = float('inf')
            best_detection_idx = -1
            
            for i, player in enumerate(current_players):
                if i in matched_detections:
                    continue
                    
                distance = self.calculate_distance(
                    last_detection['bbox'], 
                    player['bbox']
                )
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            # Update track if good match found
            if best_detection_idx >= 0:
                matched_detections.add(best_detection_idx)
                matched_tracks.add(track_id)
                
                self.tracks[track_id].append({
                    'frame_id': frame_id,
                    'bbox': current_players[best_detection_idx]['bbox'],
                    'confidence': current_players[best_detection_idx]['confidence']
                })
        
        # Create new tracks for unmatched detections
        for i, player in enumerate(current_players):
            if i not in matched_detections:
                self.tracks[self.next_track_id] = [{
                    'frame_id': frame_id,
                    'bbox': player['bbox'],
                    'confidence': player['confidence']
                }]
                self.next_track_id += 1
    
    def get_valid_tracks(self):
        """Get tracks that meet minimum length requirement."""
        valid_tracks = {}
        for track_id, track in self.tracks.items():
            if len(track) >= self.min_track_length:
                valid_tracks[track_id] = track
        return valid_tracks
    
    def get_track_statistics(self):
        """Calculate statistics for all tracks."""
        valid_tracks = self.get_valid_tracks()
        
        stats = {
            'total_tracks': len(valid_tracks),
            'avg_track_length': 0,
            'max_track_length': 0,
            'min_track_length': float('inf'),
            'track_details': {}
        }
        
        if valid_tracks:
            track_lengths = [len(track) for track in valid_tracks.values()]
            stats['avg_track_length'] = np.mean(track_lengths)
            stats['max_track_length'] = np.max(track_lengths)
            stats['min_track_length'] = np.min(track_lengths)
            
            for track_id, track in valid_tracks.items():
                stats['track_details'][track_id] = {
                    'length': len(track),
                    'start_frame': track[0]['frame_id'],
                    'end_frame': track[-1]['frame_id'],
                    'avg_confidence': np.mean([d['confidence'] for d in track])
                }
        
        return stats


class BallTracker:
    def __init__(self, max_distance=50):
        """
        Initialize ball tracker.
        
        Args:
            max_distance (float): Maximum distance to associate ball detections
        """
        self.max_distance = max_distance
        self.ball_positions = []
        
    def get_bbox_center(self, bbox):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate distance between two bounding boxes."""
        center1 = self.get_bbox_center(bbox1)
        center2 = self.get_bbox_center(bbox2)
        return euclidean(center1, center2)
    
    def update_track(self, frame_detections):
        """
        Update ball track with new frame detection.
        
        Args:
            frame_detections (dict): Detection results for current frame
        """
        frame_id = frame_detections['frame_id']
        ball_detection = frame_detections.get('ball')
        
        if ball_detection:
            # Check if this detection is close to previous position
            if self.ball_positions:
                last_position = self.ball_positions[-1]
                distance = self.calculate_distance(
                    last_position['bbox'], 
                    ball_detection['bbox']
                )
                
                # Only add if it's a reasonable distance (not a false detection)
                if distance < self.max_distance:
                    self.ball_positions.append({
                        'frame_id': frame_id,
                        'bbox': ball_detection['bbox'],
                        'confidence': ball_detection['confidence']
                    })
            else:
                # First ball detection
                self.ball_positions.append({
                    'frame_id': frame_id,
                    'bbox': ball_detection['bbox'],
                    'confidence': ball_detection['confidence']
                })
    
    def get_ball_statistics(self):
        """Calculate ball tracking statistics."""
        if not self.ball_positions:
            return {
                'total_detections': 0,
                'detection_rate': 0,
                'avg_confidence': 0
            }
        
        total_frames = max(pos['frame_id'] for pos in self.ball_positions) + 1
        
        return {
            'total_detections': len(self.ball_positions),
            'detection_rate': len(self.ball_positions) / total_frames,
            'avg_confidence': np.mean([pos['confidence'] for pos in self.ball_positions]),
            'positions': self.ball_positions
        }


class MatchTracker:
    def __init__(self):
        """Initialize match tracker with player and ball trackers."""
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.detections = []
        
    def process_detections(self, detections_file):
        """
        Process detection results and create tracks.
        
        Args:
            detections_file (str): Path to JSON file with detection results
        """
        with open(detections_file, 'r') as f:
            self.detections = json.load(f)
        
        print(f"Processing {len(self.detections)} frames...")
        
        for frame_detections in self.detections:
            self.player_tracker.update_tracks(frame_detections)
            self.ball_tracker.update_track(frame_detections)
        
        print("Tracking completed!")
    
    def save_tracking_results(self, output_dir):
        """Save tracking results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save player tracks
        valid_tracks = self.player_tracker.get_valid_tracks()
        tracks_file = output_path / "player_tracks.json"
        with open(tracks_file, 'w') as f:
            json.dump(valid_tracks, f, indent=2)
        
        # Save ball positions
        ball_file = output_path / "ball_positions.json"
        with open(ball_file, 'w') as f:
            json.dump(self.ball_tracker.ball_positions, f, indent=2)
        
        # Save statistics
        stats = {
            'player_tracks': self.player_tracker.get_track_statistics(),
            'ball_tracking': self.ball_tracker.get_ball_statistics()
        }
        
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
        
        stats = convert_numpy_types(stats)
        
        stats_file = output_path / "tracking_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Tracking results saved to: {output_path}")
        print(f"Player tracks: {tracks_file}")
        print(f"Ball positions: {ball_file}")
        print(f"Statistics: {stats_file}")
        
        return stats


def main():
    """Example usage of the tracking module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Soccer Match Tracking')
    parser.add_argument('--detections', required=True, help='Path to detection results JSON file')
    parser.add_argument('--output', default='outputs', help='Output directory for tracking results')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = MatchTracker()
    
    # Process detections
    tracker.process_detections(args.detections)
    
    # Save results
    stats = tracker.save_tracking_results(args.output)
    
    # Print summary
    print("\nTracking Summary:")
    print(f"Player tracks: {stats['player_tracks']['total_tracks']}")
    print(f"Ball detections: {stats['ball_tracking']['total_detections']}")


if __name__ == "__main__":
    main() 