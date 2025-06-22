# SAM2 YOLO Annotation Generator

A Python project that automatically generates YOLO format annotations using Segment Anything 2 (SAM2) for object detection and segmentation. The system detects objects with similar volume, shape, and color characteristics, then groups them into classes and generates corresponding YOLO annotation files.

## Features

- **Automatic Segmentation**: Uses SAM2 model for automatic object segmentation
- **Similar Object Detection**: Groups objects based on volume, shape, and color similarity
- **YOLO Format Output**: Generates standard YOLO format annotation files
- **Batch Processing**: Processes entire directories of images
- **Configurable Parameters**: Adjustable similarity thresholds and area filters
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB RAM (16GB+ recommended for large images)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd simobj_ds_gen
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SAM2 model** (optional):
   The system will automatically download the SAM2 model on first run, or you can manually download it:
   ```bash
   # Download SAM2 model (will be done automatically)
   # Model will be saved to the current directory
   ```

## Usage

### Basic Usage

```bash
python main.py --input_dir /path/to/images --output_dir /path/to/annotations
```

### Advanced Usage

```bash
python main.py \
    --input_dir /path/to/images \
    --output_dir /path/to/annotations \
    --similarity_threshold 0.8 \
    --min_area 200 \
    --max_area 50000 \
    --device cuda \
    --log_level DEBUG
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dir` | str | Required | Directory containing input images |
| `--output_dir` | str | Required | Directory to save YOLO annotation files |
| `--model_path` | str | None | Path to SAM2 model checkpoint (optional) |
| `--device` | str | "cuda" | Device to run inference on ("cuda" or "cpu") |
| `--similarity_threshold` | float | 0.7 | Threshold for object similarity (0.0-1.0) |
| `--min_area` | int | 100 | Minimum object area in pixels |
| `--max_area` | int | 100000 | Maximum object area in pixels |
| `--log_level` | str | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Output Format

The system generates YOLO format annotation files (.txt) with the following structure:

```
class_id center_x center_y width height
```

Where:
- `class_id`: Integer ID for the object class (0-based)
- `center_x, center_y`: Normalized center coordinates (0.0-1.0)
- `width, height`: Normalized bounding box dimensions (0.0-1.0)

### Example Output

```
0 0.250000 0.300000 0.100000 0.150000
0 0.450000 0.350000 0.120000 0.140000
1 0.700000 0.600000 0.080000 0.100000
```

## Object Detection Algorithm

The system uses a multi-stage approach:

1. **SAM2 Segmentation**: Automatically segments all objects in the image
2. **Feature Extraction**: Extracts features for each object:
   - Area (normalized by image size)
   - Aspect ratio
   - Circularity (shape measure)
   - Mean RGB color values
3. **Clustering**: Uses DBSCAN clustering to group similar objects
4. **Class Assignment**: Assigns unique class IDs to each group

## Configuration

### Similarity Threshold

The `similarity_threshold` parameter controls how similar objects need to be to be grouped together:
- Higher values (0.8-0.9): More strict grouping, fewer classes
- Lower values (0.5-0.7): More lenient grouping, more classes

### Area Filtering

Use `min_area` and `max_area` to filter objects by size:
- `min_area`: Excludes very small objects (noise)
- `max_area`: Excludes very large objects (background)

## Performance Tips

1. **Use GPU**: Set `--device cuda` for faster processing
2. **Adjust batch size**: For large images, consider processing in smaller batches
3. **Memory management**: Close other applications to free up GPU memory
4. **Similarity threshold**: Start with 0.7 and adjust based on results

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce image resolution
   - Close other GPU applications
   - Use CPU mode: `--device cpu`

2. **No objects detected**:
   - Lower the `min_area` threshold
   - Check image quality and contrast
   - Verify image format is supported

3. **Too many/few object classes**:
   - Adjust `similarity_threshold`
   - Modify area filtering parameters

### Logging

Use `--log_level DEBUG` for detailed debugging information:

```bash
python main.py --input_dir images --output_dir annotations --log_level DEBUG
```

## Project Structure

```
simobj_ds_gen/
├── main.py              # Main script
├── sam2_processor.py    # SAM2 model handling
├── object_detector.py   # Object similarity detection
├── yolo_annotator.py    # YOLO annotation generation
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Dependencies

- `torch`: PyTorch for deep learning
- `opencv-python`: Image processing
- `numpy`: Numerical computations
- `scikit-learn`: Clustering algorithms
- `segment-anything-2`: SAM2 model
- `tqdm`: Progress bars
- `Pillow`: Image handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Meta AI for the Segment Anything 2 model
- The YOLO community for the annotation format
- Open source contributors for the supporting libraries
