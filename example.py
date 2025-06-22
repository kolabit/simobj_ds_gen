#!/usr/bin/env python3
"""
Example script demonstrating SAM2 YOLO annotation generation.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main as run_main
from utils import setup_logging, check_dependencies

def create_sample_structure():
    """Create sample directory structure for demonstration."""
    base_dir = Path("sample_data")
    
    # Create directories
    input_dir = base_dir / "images"
    output_dir = base_dir / "annotations"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created sample directory structure:")
    print(f"  Input images: {input_dir}")
    print(f"  Output annotations: {output_dir}")
    
    return str(input_dir), str(output_dir)

def run_example():
    """Run the example with sample data."""
    print("SAM2 YOLO Annotation Generator - Example")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("❌ Some dependencies are missing. Please install them first.")
        return False
    
    print("✅ All dependencies are available")
    
    # Create sample structure
    input_dir, output_dir = create_sample_structure()
    
    # Check if there are images in the input directory
    image_files = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
    
    if not image_files:
        print(f"\n⚠️  No images found in {input_dir}")
        print("Please add some images to the input directory and run again.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return False
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Setup logging
    setup_logging("INFO")
    
    # Run the main processing
    print("\nStarting annotation generation...")
    print("This may take a while depending on the number and size of images.")
    
    try:
        # Simulate command line arguments
        sys.argv = [
            "main.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--similarity_threshold", "0.7",
            "--min_area", "100",
            "--max_area", "100000",
            "--device", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
            "--log_level", "INFO"
        ]
        
        run_main()
        
        print("\n✅ Processing completed successfully!")
        print(f"Check the output directory: {output_dir}")
        
        # Show results
        annotation_files = list(Path(output_dir).glob("*.txt"))
        if annotation_files:
            print(f"\nGenerated {len(annotation_files)} annotation files:")
            for ann_file in annotation_files:
                print(f"  - {ann_file.name}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return False

def show_usage_examples():
    """Show various usage examples."""
    print("\nUsage Examples:")
    print("=" * 30)
    
    examples = [
        {
            "description": "Basic usage with default settings",
            "command": "python main.py --input_dir images --output_dir annotations"
        },
        {
            "description": "High similarity threshold (fewer classes)",
            "command": "python main.py --input_dir images --output_dir annotations --similarity_threshold 0.9"
        },
        {
            "description": "Low similarity threshold (more classes)",
            "command": "python main.py --input_dir images --output_dir annotations --similarity_threshold 0.5"
        },
        {
            "description": "Filter small objects",
            "command": "python main.py --input_dir images --output_dir annotations --min_area 500"
        },
        {
            "description": "Use CPU instead of GPU",
            "command": "python main.py --input_dir images --output_dir annotations --device cpu"
        },
        {
            "description": "Debug mode with detailed logging",
            "command": "python main.py --input_dir images --output_dir annotations --log_level DEBUG"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

def main():
    """Main function for the example script."""
    print("SAM2 YOLO Annotation Generator")
    print("Example and Usage Guide")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage_examples()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_usage_examples()
        return
    
    # Run the example
    success = run_example()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated annotation files")
        print("2. Adjust parameters based on your needs")
        print("3. Process your own images")
        print("\nFor more examples, run: python example.py --examples")
    else:
        print("\n💡 Tips:")
        print("1. Make sure you have images in the input directory")
        print("2. Check that all dependencies are installed")
        print("3. Ensure you have sufficient disk space")
        print("4. For GPU usage, make sure CUDA is available")

if __name__ == "__main__":
    main() 