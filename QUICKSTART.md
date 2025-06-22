# Quick Start Guide

Get up and running with SAM2 YOLO Annotation Generator in minutes!

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended, but CPU works too)
- At least 8GB RAM

## Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd simobj_ds_gen
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation**:
   ```bash
   python test_installation.py
   ```

## Quick Usage

1. **Prepare your images**:
   - Put your images in a folder (e.g., `my_images/`)
   - Supported formats: JPG, PNG, BMP, TIFF

2. **Run the annotation generator**:
   ```bash
   python main.py --input_dir my_images --output_dir my_annotations
   ```

3. **Check the results**:
   - Look in `my_annotations/` for the generated `.txt` files
   - Each image will have a corresponding annotation file

## Example with Sample Data

1. **Create sample directories**:
   ```bash
   mkdir -p sample_data/images
   mkdir -p sample_data/annotations
   ```

2. **Add some images** to `sample_data/images/`

3. **Run the example**:
   ```bash
   python example.py
   ```

## Common Use Cases

### High-Quality Annotations (Fewer Classes)
```bash
python main.py \
    --input_dir images \
    --output_dir annotations \
    --similarity_threshold 0.9 \
    --min_area 200
```

### Detailed Annotations (More Classes)
```bash
python main.py \
    --input_dir images \
    --output_dir annotations \
    --similarity_threshold 0.5 \
    --min_area 50
```

### CPU Processing (No GPU)
```bash
python main.py \
    --input_dir images \
    --output_dir annotations \
    --device cpu
```

### Debug Mode
```bash
python main.py \
    --input_dir images \
    --output_dir annotations \
    --log_level DEBUG
```

## Output Format

Each annotation file contains YOLO format lines:
```
class_id center_x center_y width height
```

Example:
```
0 0.250000 0.300000 0.100000 0.150000
0 0.450000 0.350000 0.120000 0.140000
1 0.700000 0.600000 0.080000 0.100000
```

## Troubleshooting

### "CUDA out of memory"
- Use CPU: `--device cpu`
- Reduce image resolution
- Close other GPU applications

### "No objects detected"
- Lower `--min_area` (e.g., 50)
- Check image quality
- Verify image format

### "Too many/few classes"
- Adjust `--similarity_threshold`
- Higher = fewer classes
- Lower = more classes

## Next Steps

1. **Review generated annotations** in your output directory
2. **Adjust parameters** based on your specific needs
3. **Process your full dataset**
4. **Use annotations** with YOLO training frameworks

## Getting Help

- Run `python example.py --help` for usage examples
- Check the full README.md for detailed documentation
- Use `--log_level DEBUG` for detailed error information

## Performance Tips

- **GPU**: Use `--device cuda` for 5-10x faster processing
- **Batch size**: Process multiple images in parallel
- **Memory**: Close other applications to free up GPU memory
- **Storage**: Ensure sufficient disk space for annotations

Happy annotating! 🎉 