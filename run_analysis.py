#!/usr/bin/env python3
"""
Main script to run the complete soccer match analysis pipeline.
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detection import SoccerDetector
from tracking import MatchTracker
from analysis import MatchAnalyzer
from visualization import DashboardVisualizer


def run_pipeline(video_path, output_dir="outputs", max_frames=None, skip_detection=False):
    """
    Run the complete analysis pipeline.
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Output directory for results
        max_frames (int): Maximum frames to process (for testing)
        skip_detection (bool): Skip detection if results already exist
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("⚽ SOCCER MATCH ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Detection
    if not skip_detection:
        print("\n1. RUNNING DETECTION...")
        print("-" * 30)
        
        detector = SoccerDetector()
        detections = detector.process_video(
            video_path=video_path,
            output_dir=output_dir,
            max_frames=max_frames
        )
        
        print(f"✓ Detection completed: {len(detections)} frames processed")
    else:
        print("\n1. SKIPPING DETECTION (using existing results)")
    
    # Step 2: Tracking
    print("\n2. RUNNING TRACKING...")
    print("-" * 30)
    
    # Find latest detection file
    detection_files = list(output_path.glob("detections_*.json"))
    if not detection_files:
        print("❌ No detection files found. Run detection first.")
        return
    
    latest_detection_file = max(detection_files, key=lambda x: x.stat().st_mtime)
    print(f"Using detection file: {latest_detection_file.name}")
    
    tracker = MatchTracker()
    tracker.process_detections(str(latest_detection_file))
    tracking_stats = tracker.save_tracking_results(output_dir)
    
    print(f"✓ Tracking completed: {tracking_stats['player_tracks']['total_tracks']} player tracks")
    
    # Step 3: Analysis
    print("\n3. RUNNING ANALYSIS...")
    print("-" * 30)
    
    analyzer = MatchAnalyzer()
    analysis_results = analyzer.analyze_match(output_dir)
    summary = analyzer.save_analysis_results(output_dir)
    
    print("✓ Analysis completed")
    print(f"  - Left team possession: {summary['possession']['left_team_possession']:.1%}")
    print(f"  - Right team possession: {summary['possession']['right_team_possession']:.1%}")
    print(f"  - Players analyzed: {summary['movement']['total_players']}")
    
    # Step 4: Visualization
    print("\n4. CREATING VISUALIZATIONS...")
    print("-" * 30)
    
    dashboard = DashboardVisualizer()
    dashboard.create_comprehensive_dashboard(analysis_results, output_path / "visualizations")
    
    print("✓ Visualizations completed")
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Visualizations: {output_path / 'visualizations'}")
    print(f"Analysis summary: {output_path / 'analysis_summary.json'}")
    
    print("\nTo view the interactive dashboard, run:")
    print(f"streamlit run src/dashboard.py")
    
    return analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Match Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_analysis.py --video data/match.mp4
  
  # Test with limited frames
  python run_analysis.py --video data/match.mp4 --max-frames 100
  
  # Skip detection (use existing results)
  python run_analysis.py --video data/match.mp4 --skip-detection
        """
    )
    
    parser.add_argument(
        "--video", 
        required=True, 
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output", 
        default="outputs", 
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--max-frames", 
        type=int, 
        help="Maximum frames to process (for testing)"
    )
    
    parser.add_argument(
        "--skip-detection", 
        action="store_true", 
        help="Skip detection step (use existing results)"
    )
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        print("Please provide a valid video file path.")
        return
    
    try:
        results = run_pipeline(
            video_path=args.video,
            output_dir=args.output,
            max_frames=args.max_frames,
            skip_detection=args.skip_detection
        )
        
        if results:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")
            
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 