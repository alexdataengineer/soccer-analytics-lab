"""
Soccer Match Detection Module
Uses YOLOv8 to detect players and the ball in soccer match videos.
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime


class SoccerDetector:
    def __init__(self, model_path=None):
        """
        Initialize the soccer detector with YOLOv8 model.
        
        Args:
            model_path (str): Path to custom YOLOv8 model. If None, uses default COCO model.
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use default YOLOv8 model (will detect person class)
            self.model = YOLO('yolov8n.pt')
        
        # Soccer-specific classes we're interested in
        self.person_class = 0  # COCO person class
        self.sports_ball_class = 32  # COCO sports ball class
        
        # Detection results storage
        self.detections = []
        self.frame_count = 0
        
    def detect_frame(self, frame):
        """
        Detect players and ball in a single frame.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            dict: Detection results for the frame
        """
        results = self.model(frame, verbose=False)
        
        frame_detections = {
            'frame_id': self.frame_count,
            'timestamp': self.frame_count / 30.0,  # Assuming 30 FPS
            'players': [],
            'ball': None
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id
                    }
                    
                    # Categorize detections
                    if class_id == self.person_class and confidence > 0.5:
                        frame_detections['players'].append(detection)
                    elif class_id == self.sports_ball_class and confidence > 0.3:
                        frame_detections['ball'] = detection
        
        self.detections.append(frame_detections)
        self.frame_count += 1
        
        return frame_detections
    
    def process_video(self, video_path, output_dir=None, max_frames=None):
        """
        Process entire video and extract detections.
        
        Args:
            video_path (str): Path to input video file
            output_dir (str): Directory to save results
            max_frames (int): Maximum number of frames to process (for testing)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        # Process frames
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            self.detect_frame(frame)
        
        cap.release()
        
        # Save results
        if output_dir:
            self.save_results(output_dir)
        
        return self.detections
    
    def save_results(self, output_dir):
        """Save detection results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detections
        detections_file = output_path / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(detections_file, 'w') as f:
            json.dump(self.detections, f, indent=2)
        
        # Save summary statistics
        summary = self.get_summary_stats()
        summary_file = output_path / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        print(f"Detections: {detections_file}")
        print(f"Summary: {summary_file}")
    
    def get_summary_stats(self):
        """Get summary statistics from detections."""
        total_players = sum(len(frame['players']) for frame in self.detections)
        total_ball_detections = sum(1 for frame in self.detections if frame['ball'] is not None)
        
        return {
            'total_frames': len(self.detections),
            'total_player_detections': total_players,
            'total_ball_detections': total_ball_detections,
            'avg_players_per_frame': total_players / len(self.detections) if self.detections else 0,
            'ball_detection_rate': total_ball_detections / len(self.detections) if self.detections else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Soccer Match Detection using YOLOv8')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default='outputs', help='Output directory for results')
    parser.add_argument('--model', help='Path to custom YOLOv8 model')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SoccerDetector(model_path=args.model)
    
    # Process video
    detections = detector.process_video(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames
    )
    
    # Print summary
    summary = detector.get_summary_stats()
    print("\nDetection Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 