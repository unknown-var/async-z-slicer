from __future__ import annotations
from typing import List
import math

# Import components from our newly split slicer logic
from mesh_slicer import Point, PrinterSettings
from profiler import SlicerProfiler
class GCodeGenerator:
    """Generates GCode from sliced layers"""

    def __init__(self, layers: List[List[List[tuple[float, float, float]]]], printer_settings: PrinterSettings):
        """
        Initialize GCode generator

        Args:
            layers: List of sliced layers with Polygons
            printer_settings: PrinterSettings object
        """
        self.layers: list[list[list[tuple[float, float, float]]]] = layers
        self.printer = printer_settings
        self.gcode = []
        self.current_z = 0.0
        self.current_extrusion = 0.0
        self.nozzle_diameter = printer_settings.get('nozzle_diameter', 0.4)
        self.layer_height = printer_settings.get('default_layer_height', 0.2)
        self.extrusion_factor = printer_settings.get('extrusion_factor', 0.04)
        self.print_speed = printer_settings.get('default_print_speed', 2400)

    def generate(self) -> List[str]:
        """
        Generate full GCode from layers

        Returns:
            List of GCode lines
        """
        profiler = SlicerProfiler.get_instance()
        profiler.start("4. GCode generation (Total)")
        self._add_start_gcode()
        self._generate_layer_gcode()
        self._add_end_gcode()
        profiler.stop("4. GCode generation (Total)")
        return self.gcode

    def _add_start_gcode(self):
        """Add printer-specific start GCode"""
        start_gcode = self.printer.get('start_gcode', [])

        # Format template values
        format_dict = {
            'nozzle_temp': self.printer.get('default_nozzle_temp', 210),
            'bed_temp': self.printer.get('default_bed_temp', 0),
            'prime_x_start': self.printer.get('prime_line_x_start', -10),
            'prime_y_start': self.printer.get('prime_line_y_start', -10),
            'prime_x_end': self.printer.get('prime_line_x_end', 10),
            'prime_y_end': self.printer.get('prime_line_y_end', -10),
            'prime_extrusion': self.printer.get('prime_line_extrusion', 10),
            'prime_speed': self.printer.get('prime_line_speed', 1200),
        }

        for line in start_gcode:
            try:
                formatted_line = line.format(**format_dict)
                self.gcode.append(formatted_line)
            except KeyError:
                self.gcode.append(line)

    def _generate_layer_gcode(self):
        """Generate GCode for all layers"""
        for layer_idx, layer_traces in enumerate(self.layers):
            self.current_z = layer_traces[0][0][2]

            indicator_template = self.printer.get('layer_height_indicator', ';layer:{layer_height:.2f}')
            if isinstance(indicator_template, str):
                templates = [indicator_template]
            else:
                templates = indicator_template

            format_dict = {
                'layer_height': self.layer_height,
                'layer_z': self.current_z,
                'layer_num': layer_idx
            }

            for template in templates:
                try:
                    formatted_indicator = template.format(**format_dict)
                    self.gcode.append(formatted_indicator)
                except (KeyError, ValueError):
                    self.gcode.append(template)

            # Move to layer height
            self.gcode.append(f"; Layer {layer_idx}")
            self.gcode.append(f"G0 Z{self.current_z:.3f} F420")

            for poly_idx, trace in enumerate(layer_traces):
                self._generate_trace_gcode(trace, layer_idx, poly_idx)

    def _generate_trace_gcode(self, trace: list[tuple[float, float, float]], layer_idx: int, line_idx: int):
            """Generate GCode to trace a line using simplified coordinate tuples"""

            # If we don't have at least a start and an end, we can't draw a line
            if len(trace) < 2:
                return 

            first_x, first_y, _ = trace[0]

            # Move to start position
            self.gcode.append(f"G0 X{first_x:.3f} Y{first_y:.3f} F4800 ; Move to line start")

            # Force specific width and height for slicer previewers (e.g., OrcaSlicer)
            self.gcode.append(f";WIDTH:{self.nozzle_diameter * 1.1:.3f}")
            self.gcode.append(f";HEIGHT:{self.layer_height:.3f}")

            # Draw the simplified lines
            for i in range(len(trace) - 1):
                p1 = trace[i]
                p2 = trace[i + 1]

                extrusion = self._calculate_extrusion(p1, p2)
                self.current_extrusion += extrusion
                
                self.gcode.append(
                    f"G1 X{p2[0]:.3f} Y{p2[1]:.3f} E{self.current_extrusion:.3f} F{self.print_speed}"
                )

    def _calculate_extrusion(self, p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
            """Calculate extrusion amount for a line segment defined by (x, y) tuples"""
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance = math.hypot(dx, dy)
            return distance * self.extrusion_factor * self.layer_height

    def _add_end_gcode(self):
        """Add printer-specific end GCode"""
        end_gcode = self.printer.get('end_gcode', [])
        self.gcode.append("; End GCode")
        self.gcode.extend(end_gcode)

    def save_gcode(self, output_file: str):
        """Save GCode to file"""
        with open(output_file, 'w') as f:
            f.write('\n'.join(self.gcode))
        print(f"GCode saved to {output_file}")
