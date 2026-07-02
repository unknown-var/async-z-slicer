"""
Basic 3D Printing Slicer (Main Entry Point)
Converts STL files to GCode slices using trimesh and shapely
"""
from pathlib import Path
import sys
import json

# Import our split classes
from mesh_slicer import PrinterSettings, Slicer
from gcode_generator import GCodeGenerator
from path_optimizer import PathOptimizer
from profiler import SlicerProfiler


def _resolve_json_file(directory: Path, name: str) -> Path | None:
    """Resolve a JSON file by name, accepting optional .json extension."""
    candidate = Path(name)
    candidates = [candidate.name]
    if candidate.suffix.lower() != '.json':
        candidates.append(f"{candidate.name}.json")

    for file_name in candidates:
        resolved = directory / file_name
        if resolved.exists():
            return resolved

    return None


def _normalize_profile_settings(profile_settings: dict) -> dict:
    """Map slicer profile keys to the printer settings keys used in the app."""
    key_mapping = {
        'layer_height': 'default_layer_height',
        'print_speed': 'default_print_speed',
        'nozzle_temp': 'default_nozzle_temp',
        'bed_temp': 'default_bed_temp',
    }

    normalized = {}
    for key, value in profile_settings.items():
        normalized[key_mapping.get(key, key)] = value
    return normalized


def main():
    """Command-line interface for the slicer"""
    perf_flag = False
    if '--perf' in sys.argv:
        perf_flag = True
        sys.argv.remove('--perf')
        SlicerProfiler.get_instance().enable()

    profiler = SlicerProfiler.get_instance()
    profiler.start("Total execution time")

    if len(sys.argv) < 2:
        print("Usage: python slicer.py <stl_file> [printer_name|profile_name] [profile_name|output_gcode] [output_gcode] [--perf]")
        print()
        print("Example:")
        print("  python slicer.py model.stl --perf")
        print("  python slicer.py model.stl flashforge_finder fast output.gcode --perf")
        print("  python slicer.py model.stl fast")
        print("  python slicer.py model.stl fast.json")
        sys.exit(1)

    stl_file = sys.argv[1]
    optional_args = sys.argv[2:]
    if len(optional_args) > 3:
        print("❌ Too many arguments.")
        print("Usage: python slicer.py <stl_file> [printer_name|profile_name] [profile_name|output_gcode] [output_gcode]")
        sys.exit(1)

    settings_dir = Path(__file__).parent / 'printer_settings'
    profile_dir = Path(__file__).parent / 'slicer_profiles'
    printer_name = 'flashforge_finder.json'
    profile_name = None
    output_gcode = None

    if len(optional_args) >= 1:
        first_arg = optional_args[0]
        if _resolve_json_file(settings_dir, first_arg):
            printer_name = first_arg
        elif _resolve_json_file(profile_dir, first_arg):
            profile_name = first_arg
        else:
            # Keep legacy behavior: first optional arg is treated as printer.
            printer_name = first_arg

    if len(optional_args) >= 2:
        second_arg = optional_args[1]
        if second_arg.lower().endswith('.gcode'):
            output_gcode = second_arg
        else:
            profile_name = second_arg

    if len(optional_args) == 3:
        output_gcode = optional_args[2]

    stl_path = Path(stl_file)
    if not stl_path.exists():
        print(f"❌ STL file not found: {stl_file}")
        sys.exit(1)

    # Load printer settings
    settings_file = _resolve_json_file(settings_dir, printer_name)

    if settings_file is None:
        print(f"❌ Printer settings not found: {printer_name}")
        print(f"\nAvailable printers in {settings_dir}:")
        for f in sorted(settings_dir.glob('*.json')):
            print(f"  - {f.name}")
        sys.exit(1)

    printer_settings = PrinterSettings(str(settings_file))

    if not printer_settings.validate():
        print("❌ Invalid printer settings!")
        sys.exit(1)

    if profile_name is not None:
        profile_file = _resolve_json_file(profile_dir, profile_name)
        if profile_file is None:
            print(f"❌ Slicer profile not found: {profile_name}")
            print(f"\nAvailable slicer profiles in {profile_dir}:")
            for f in sorted(profile_dir.glob('*.json')):
                print(f"  - {f.name}")
            sys.exit(1)

        with open(profile_file, 'r') as f:
            profile_settings = json.load(f)

        printer_settings.config.update(_normalize_profile_settings(profile_settings))
        print(f"✓ Loaded slicer profile: {profile_file.name}")

    print(f"✓ Loaded printer: {printer_settings.get('printer_name')}")
    print(f"✓ Bed size: {printer_settings.get('bed_width')} x {printer_settings.get('bed_height')} mm")
    print(f"✓ Layer height: {printer_settings.get('default_layer_height')} mm")
    print()

    # Slice the model
    print("Slicing model...")
    slicer = Slicer(stl_file, printer_settings)
    layers = slicer.slice()

    print(f"✓ Model sliced into {slicer.get_layer_count()} layers")
    print()

    # Optimizing path
    print("Optimizing path...")
    optimizer = PathOptimizer(layers)
    paths = optimizer.optimize()

    # Generate GCode
    print("Generating GCode...")
    generator = GCodeGenerator(paths, printer_settings)
    gcode = generator.generate()

    print(f"✓ Generated {len(gcode)} lines of GCode")
    print()

    # Save output
    if output_gcode is None:
        output_gcode = stl_path.stem + '.gcode'

    generator.save_gcode(output_gcode)
    print(f"✓ Saved to {output_gcode}")

    profiler.stop("Total execution time")
    if perf_flag:
        SlicerProfiler.get_instance().save_report("perf_diagnostics")


if __name__ == '__main__':
    main()
