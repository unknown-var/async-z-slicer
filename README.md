# 3d printing slicer prototype with Asynchronous Z-Stepping

This is a basic 3d printing slice meant to test out a unique slicing method which aims to avoid stair-stepping and increase overhang performance.
The slicer converts STL files to GCode using **trimesh** for 3D model handling and **shapely** for 2D geometric operations.

## Explanation of algorithm
I will write this when i have time. 

## Features

- **STL Loading**: Uses trimesh to load and validate 3D models
- **Layer Slicing**: Slices models into horizontal layers using plane intersections
- **Shapely Integration**: Converts cross-sections to shapely Polygons for advanced geometry operations
- **Printer Profiles**: JSON-based printer configuration system
- **Slicer Profiles**: JSON-based profile overrides for slicing parameters
- **GCode Generation**: Generates printer-specific GCode with custom start/end sequences
- **Easy Extensibility**: Clean class-based architecture for customization

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- **trimesh**: 3D model loading and mesh operations
- **shapely**: 2D geometry operations and polygon handling
- **numpy**: Numerical operations

## Project Structure

```
python-slicer/
├── slicer.py              # Main slicer implementation
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── printer_settings/      # Printer configuration files
    └── flashforge_finder.json
└── slicer_profiles/       # Slicer profile overrides
    └── fast.json
```

## Usage

### Basic Usage

```bash
python slicer.py model.stl
```

This will:
1. Load the STL file
2. Use the default Flashforge Finder printer settings
3. Slice the model into lines
4. Generate GCode
5. Save to `model.gcode`

### With slicer profiles

You can provide a slicer profile by name, with or without `.json`:

```bash
python slicer.py model.stl fast
python slicer.py model.stl fast.json
python slicer.py model.stl flashforge_finder fast output.gcode
```

## Printer Settings Format

Printer settings are JSON files with the following structure:

```json
{
  "printer_name": "Printer Model Name",
  "bed_width": 140,
  "bed_height": 140,
  "bed_depth": 140,
  "nozzle_diameter": 0.4,
  "default_layer_height": 0.28,
  "default_print_speed": 2400,
  "default_nozzle_temp": 210,
  "default_bed_temp": 0,
  "has_heated_bed": false,
  "has_cooling_fan": true,
  "prime_line_x_start": -60,
  "prime_line_y_start": -55,
  "prime_line_x_end": 60,
  "prime_line_y_end": -55,
  "prime_line_extrusion": 15,
  "prime_line_speed": 1200,
  "extrusion_factor": 0.04,
  "start_gcode": [
    "G90 ; Absolute positioning",
    "M82 ; Absolute extrusion mode",
    "G28 ; Home axes",
    "M104 S{nozzle_temp} ; Set nozzle temp",
    "M140 S{bed_temp} ; Set bed temp",
    "M106 S255 ; Turn on fan"
  ],
  "end_gcode": [
    "M104 S0 ; Turn off heater",
    "M140 S0 ; Turn off bed",
    "G28 X Y ; Home X and Y",
    "M18 ; Disable stepper motors"
  ]
}
```

### Required Settings

- `bed_width`: Bed width in mm
- `bed_height`: Bed height in mm
- `bed_depth`: Bed depth in mm
- `nozzle_diameter`: Nozzle diameter in mm
- `default_layer_height`: Default layer height in mm

## Slicer Profile Format

Slicer profiles are JSON files located in the `slicer_profiles/` directory. They can override printer settings and configure specific slicing options.

Example profile:

```json
{
  "layer_height": 0.3,
  "number_of_walls": 2,
  "solid_bottom_layer": true,
  "infill": true,
  "infill_spacing": 2.5
}
```

### Configurable Slicing Settings

You can override printer properties (e.g. `layer_height`, `number_of_walls`) as well as the following slicing parameters in your profile:

- `solid_bottom_layer` (bool): If true, slices the first actual layer of every wall layer as a solid filled layer. Default is `true`.
- `infill` (bool): If true, generates linear infill inside the innermost wall boundary. Default is `true`.
- `infill_spacing` (float): Spacing between parallel infill lines in mm. Default is `2.5` mm.
- `point_spacing` (float): Distance between generated points along a path in mm. Default is `0.1` mm.
- `z_samples_per_layer` (int): Number of Z-level samples per layer height (asynchronous wall resolution). Default is `10`.
- `horizontal_detection_distance_multiple` (float): Influences how tightly path lines are packed in leaning geometry. Default is `0.7`.
- `vertical_offset_multiple` (float): Slicing Z-offset scaling multiplier for inner walls. Default is `1.15`.
- `min_line_segments` (int): Minimum number of segments in a path before it gets discarded to prevent blobs. Default is `20`.
- `long_line_sample_bias` (float): Distance sample bias for straight lines. Default is `0.9`.

### GCode Template Variables

In `start_gcode` and `end_gcode`, use these variables:

- `{nozzle_temp}`: Nozzle temperature (from `default_nozzle_temp`)
- `{bed_temp}`: Bed temperature (from `default_bed_temp`)
- `{prime_x_start}`: Prime line start X (from `prime_line_x_start`)
- `{prime_y_start}`: Prime line start Y (from `prime_line_y_start`)
- `{prime_x_end}`: Prime line end X (from `prime_line_x_end`)
- `{prime_y_end}`: Prime line end Y (from `prime_line_y_end`)
- `{prime_extrusion}`: Prime line extrusion (from `prime_line_extrusion`)
- `{prime_speed}`: Prime line speed (from `prime_line_speed`)

## License

MIT

## Contributing
Disclaimer this is not meant to be any kind of general slicer it is specifically meant to test and develop this new kind of slicing!

Right now the most important thing is getting one outer wall to print well and consistently so if you want to contribute that would be a good place to start. Otherwise feel free to extend this slicer with additional features like infill patterns or multiple wall layers. 
