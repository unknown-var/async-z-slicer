"""
Basic 3D Printing Slicer (Main Entry Point)
Converts STL files to GCode slices using trimesh and shapely
"""
from pathlib import Path
import sys

# Import our split classes
from mesh_slicer import PrinterSettings, Slicer
from gcode_generator import GCodeGenerator
from path_optimizer import PathOptimizer


def main():
    """Command-line interface for the slicer"""
    if len(sys.argv) < 2:
        print("Usage: python slicer.py <stl_file> [printer_name] [output_gcode]")
        print()
        print("Example:")
        print("  python slicer.py model.stl")
        print("  python slicer.py model.stl flashforge_finder.json output.gcode")
        sys.exit(1)

    stl_file = sys.argv[1]
    printer_name = sys.argv[2] if len(sys.argv) > 2 else 'flashforge_finder.json'
    output_gcode = sys.argv[3] if len(sys.argv) > 3 else None

    stl_path = Path(stl_file)
    if not stl_path.exists():
        print(f"❌ STL file not found: {stl_file}")
        sys.exit(1)

    # Load printer settings
    settings_dir = Path(__file__).parent / 'printer_settings'
    settings_file = settings_dir / printer_name

    if not settings_file.exists():
        print(f"❌ Printer settings not found: {settings_file}")
        print(f"\nAvailable printers in {settings_dir}:")
        for f in sorted(settings_dir.glob('*.json')):
            print(f"  - {f.name}")
        sys.exit(1)

    printer_settings = PrinterSettings(str(settings_file))

    if not printer_settings.validate():
        print("❌ Invalid printer settings!")
        sys.exit(1)

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


if __name__ == '__main__':
    main()
