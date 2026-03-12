#!/usr/bin/env python3
"""
Quick test script - Test imports và basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🧪 Quick Test - JDE Components")
    print("-" * 30)
    
    # Test 1: Basic imports
    print("1. Testing imports...")
    try:
        from yolo_jde import JDETracker, YOLOJDE
        print("   ✅ Main imports OK")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: JDETracker init
    print("2. Testing JDETracker...")
    try:
        tracker = JDETracker(model_path=None, mode="detect")
        print("   ✅ JDETracker init OK")
    except Exception as e:
        print(f"   ❌ JDETracker failed: {e}")
        return False
    
    # Test 3: YOLO-JDE class
    print("3. Testing YOLO-JDE class...")
    try:
        # Just test class creation without model loading
        yolo_class = YOLOJDE
        print("   ✅ YOLO-JDE class OK")
    except Exception as e:
        print(f"   ❌ YOLO-JDE failed: {e}")
        return False
    
    print("-" * 30)
    print("🎉 All quick tests passed!")
    print("💡 Ready to run full test: python test_detection_coco.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
