#!/usr/bin/env python3
"""
Test script to verify project setup and dependencies.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'ultralytics',
        'cv2',
        'pandas',
        'numpy',
        'sklearn',
        'streamlit',
        'matplotlib',
        'seaborn',
        'plotly',
        'tqdm',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_files = [
        'src/detection.py',
        'src/tracking.py',
        'src/analysis.py',
        'src/visualization.py',
        'src/dashboard.py',
        'requirements.txt',
        'run_analysis.py',
        'README.md'
    ]
    
    required_dirs = [
        'src',
        'data',
        'outputs',
        'notebooks'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}/")
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("\n✅ Project structure is correct!")
        return True

def test_video_file():
    """Test if video file exists."""
    print("\nTesting video file...")
    
    video_path = Path("data/FULL MATCH _ Liverpool 3-1 Manchester City _ FA Community Shield 2022-23.mp4")
    
    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"✓ Video file found: {video_path.name}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
    else:
        print("❌ Video file not found!")
        print("Please ensure the Liverpool vs Manchester City video is in the data/ directory")
        return False

def test_module_imports():
    """Test if our custom modules can be imported."""
    print("\nTesting custom module imports...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    try:
        from detection import SoccerDetector
        print("✓ SoccerDetector imported")
    except ImportError as e:
        print(f"✗ SoccerDetector: {e}")
        return False
    
    try:
        from tracking import MatchTracker
        print("✓ MatchTracker imported")
    except ImportError as e:
        print(f"✗ MatchTracker: {e}")
        return False
    
    try:
        from analysis import MatchAnalyzer
        print("✓ MatchAnalyzer imported")
    except ImportError as e:
        print(f"✗ MatchAnalyzer: {e}")
        return False
    
    try:
        from visualization import DashboardVisualizer
        print("✓ DashboardVisualizer imported")
    except ImportError as e:
        print(f"✗ DashboardVisualizer: {e}")
        return False
    
    print("\n✅ All custom modules imported successfully!")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("SOCCER MATCH ANALYSIS - SETUP TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_project_structure,
        test_video_file,
        test_module_imports
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Create virtual environment: python -m venv venv")
        print("2. Activate it: source venv/bin/activate")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Run analysis: python run_analysis.py --video \"data/FULL MATCH _ Liverpool 3-1 Manchester City _ FA Community Shield 2022-23.mp4\" --max-frames 100")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check if all files are in the correct locations")
        print("- Ensure the video file is in the data/ directory")

if __name__ == "__main__":
    main() 