#!/usr/bin/env python3
"""
Master Script: Generate All Case Studies for Professor
====================================================

This script generates all the case study materials requested by the professor:
1. Sample-level analysis with neighborhood structure
2. Examples where strong baselines fail but GemGNN succeeds
3. Multi-view analysis demonstrating method advantages
4. Professional visualizations for thesis defense

Usage: python generate_all_case_studies.py

Output: Complete case study package in outputs/ folder
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name: str) -> bool:
    """Run a case study script and handle errors."""
    try:
        print(f"\n🔄 Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    """Generate all case study materials."""
    print("=" * 80)
    print("🎓 GENERATING COMPLETE CASE STUDY PACKAGE FOR PROFESSOR")
    print("=" * 80)
    print("📋 Requirements:")
    print("   ✓ Sample-level analysis with neighborhood structure")
    print("   ✓ Examples where baselines fail but GemGNN succeeds")
    print("   ✓ Multi-view semantic analysis") 
    print("   ✓ Professional visualizations")
    print("=" * 80)
    
    # Change to case study directory
    case_study_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        import os
        os.chdir(case_study_dir)
        
        scripts = [
            "final_case_study.py",      # Overall performance analysis
            "sample_level_analysis.py",  # Sample-level neighborhood analysis
            "professor_case_study.py"    # Professor's specific requirements
        ]
        
        success_count = 0
        for script in scripts:
            if run_script(script):
                success_count += 1
        
        print("\n" + "=" * 80)
        print("📊 CASE STUDY GENERATION COMPLETE")
        print("=" * 80)
        print(f"✅ Successfully generated: {success_count}/{len(scripts)} components")
        
        # List all generated files
        outputs_dir = case_study_dir / "outputs"
        viz_dir = case_study_dir / "visualizations"
        
        print("\n📁 Generated Files:")
        
        if outputs_dir.exists():
            print("\n📄 Reports & Data (outputs/):")
            for file in sorted(outputs_dir.iterdir()):
                if file.is_file():
                    print(f"   📝 {file.name}")
        
        if viz_dir.exists():
            print("\n📊 Visualizations (visualizations/):")
            for file in sorted(viz_dir.iterdir()):
                if file.is_file():
                    print(f"   📈 {file.name}")
        
        print("\n🎯 KEY OUTPUTS FOR PROFESSOR:")
        print("   📄 professor_case_study.md      - Main defense report")
        print("   📊 professor_case_study.png     - Professional visualization")
        print("   📝 sample_level_case_study.md   - Detailed sample analysis")
        print("   📈 sample_level_analysis.png    - Sample-level visualization")
        print("   📋 detailed_case_study.md       - Comprehensive comparison")
        
        print(f"\n📂 All files saved to: {outputs_dir}")
        print("\n🎓 Ready for thesis defense!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False
    finally:
        os.chdir(original_dir)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)