from __future__ import annotations  

import json
from typing import List, Dict, Any, Optional
import trimesh
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import numpy as np
import math
import itertools
from collections import defaultdict

# Import the SDF generator
from sdf_generator import generate_inset_mesh

# This is the bias which decides how much shorter the distance to a point has to be if the last point was in a line
LONG_LINE_SAMPLE_BIAS = 0.9

# This multiple influences how tight lines are packed in leaning geomitry. Lower is more lines
HORIZONTAL_DETECTION_DISTANCE_MULTIPLE = 0.7

# This decides the vertical distance between to wall layers
VERTICAL_OFFSET_MULTIPLE = 1.15

MIN_LINE_SEGMENTS = 20

Z_SAMPLES_PER_LAYER = 10

POINT_SPACING = 0.1

class PrinterSettings:
    """Loads and manages printer configuration from JSON"""
    def __init__(self, settings_file: str):
        with open(settings_file, 'r') as f:
            self.config = json.load(f)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return self.config

    def validate(self) -> bool:
        required = [
            'bed_width', 'bed_height', 'bed_depth',
            'nozzle_diameter', 'default_layer_height'
        ]
        return all(key in self.config for key in required)


class Point:
    """Represents a point along a filament line to be printed"""
    def __init__(self, x: float, y: float, z: float, prev: Point | None = None, next: Point | None = None):
        self.x = x
        self.y = y
        self.z = z
        self.prev = prev
        self.next = next

    def to_pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Slicer:
    """Main slicer class for converting STL to layers"""

    def __init__(self, stl_file: str, printer_settings: PrinterSettings):
        loaded = trimesh.load_mesh(stl_file)
        if isinstance(loaded, trimesh.Scene):
            if hasattr(loaded, "to_mesh"):
                self.main_mesh = loaded.to_mesh()
            else:
                self.main_mesh = loaded.dump(concatenate=True)
        else:
            self.main_mesh = loaded
            
        self.printer = printer_settings
        self.layer_height = printer_settings.get('default_layer_height', 0.2)
        self.bed_width = printer_settings.get('bed_width')
        self.bed_height = printer_settings.get('bed_height')
        self.nozzle_diameter = printer_settings.get('nozzle_diameter', 0.4)
        
        self.box_height = self.layer_height
        self.box_width = self.nozzle_diameter
        self.point_spacing = printer_settings.get('point_spacing', printer_settings.get('POINT_SPACING', POINT_SPACING))
        self.z_samples_per_layer = printer_settings.get('z_samples_per_layer', printer_settings.get('Z_SAMPLES_PER_LAYER', Z_SAMPLES_PER_LAYER))
        self.z_sample_height = self.layer_height / self.z_samples_per_layer
        self.z_detection_distance = self.layer_height
        
        self.horizontal_detection_distance_multiple = printer_settings.get(
            'horizontal_detection_distance_multiple',
            printer_settings.get('HORIZONTAL_DETECTION_DISTANCE_MULTIPLE', HORIZONTAL_DETECTION_DISTANCE_MULTIPLE)
        )
        d = math.sqrt(4 * self.nozzle_diameter * self.layer_height / math.pi)
        self.horizontal_detection_distance = self.horizontal_detection_distance_multiple * d
        self.min_line_segments = printer_settings.get('min_line_segments', printer_settings.get('MIN_LINE_SEGMENTS', MIN_LINE_SEGMENTS))
        self.long_line_sample_bias = printer_settings.get('long_line_sample_bias', printer_settings.get('LONG_LINE_SAMPLE_BIAS', LONG_LINE_SAMPLE_BIAS))
        self.vertical_offset_multiple = printer_settings.get('vertical_offset_multiple', printer_settings.get('VERTICAL_OFFSET_MULTIPLE', VERTICAL_OFFSET_MULTIPLE))
        self.num_walls = printer_settings.get('number_of_walls', 1)
        self.debug = True # Set to True to export inset meshes as STL
        
        print(f"✓ Slicing settings: point_spacing={self.point_spacing}, z_samples_per_layer={self.z_samples_per_layer}, horizontal_detection_distance_multiple={self.horizontal_detection_distance_multiple}, min_line_segments={self.min_line_segments}, long_line_sample_bias={self.long_line_sample_bias}, vertical_offset_multiple={self.vertical_offset_multiple}")

        self.layers: List[List[Point]] = []

    def slice(self) -> List[List[Point]]:
        """
        Slice the mesh into independent wall layers and merge by global Z height.
        Returns a list of layers, where each layer contains a list of start Points.
        """
        z_min = float(self.main_mesh.bounds[0][2])
        z_max = float(self.main_mesh.bounds[1][2])

        print(f"Model Z range: {z_min:.2f} to {z_max:.2f} mm")
        
        epsilon = 1e-6
        z_start = z_min + epsilon
        z_stop = z_max - epsilon
        if z_stop <= z_start:
            return []

        # Generate global Z values
        global_z_values = np.arange(z_start, z_stop + self.layer_height, self.z_sample_height)
        
        # Group starting Point objects by Z height: dict[Z_Height, list[Point]]
        combined_layers: dict[float, list[Point]] = defaultdict(list)

        for wall_idx in range(self.num_walls):
            print(f"--- Wall {wall_idx + 1}/{self.num_walls} ---")
            
            # Use 0.5 * nozzle_diameter to place the centerline correctly
            h_offset = (wall_idx + 0.5) * self.nozzle_diameter
            
            # For standard walls, we don't want a vertical offset here as it causes Z-phase issues
            v_offset = (wall_idx + 0.5) * self.horizontal_detection_distance * self.vertical_offset_multiple
            
            print(f"Generating inset mesh (offset={h_offset:.2f}mm)...")
            current_mesh = generate_inset_mesh(
                self.main_mesh, 
                horizontal_offset=h_offset, 
                vertical_offset=v_offset,
                pitch=0.1
            )
            
            if current_mesh is None:
                print(f"Geometry collapsed at wall {wall_idx + 1}. Stopping inner walls.")
                break

            if self.debug:
                debug_name = f"debug_wall_{wall_idx+1}.stl"
                current_mesh.export(debug_name)
                print(f"Debug mesh saved to {debug_name}")

            # Process mesh independently to allow for future parallelization
            print(f"Slicing paths from mesh...")
            wall_starts = self._process_single_mesh(current_mesh, global_z_values)
            
            for z, start_points in wall_starts.items():
                combined_layers[z].extend(start_points)

        print("Finalizing layers...")

        # Order by Z-height and save to self.layers
        sorted_z_keys = sorted(combined_layers.keys())
        self.layers = []
        
        for z in sorted_z_keys:
            if combined_layers[z]:
                self.layers.append(combined_layers[z])

        return self.layers

    def _process_single_mesh(
        self, 
        mesh: trimesh.Trimesh, 
        z_values: np.ndarray
    ) -> dict[float, list[Point]]:
        """
        Slices a single mesh and maintains isolated state.
        Returns a dictionary grouping line start Points by Z height.
        """
        local_boxes: dict[tuple[int, int, int], list[Point]] = {}
        layer_dict: dict[float, list[Point]] = defaultdict(list)

        for z in z_values:
            polygons = self._polygons_at_z(mesh, z)
            layer_line_starts: list[Point] = []
            
            for polygon in polygons:
                layer_line_starts.extend(self._slice_polygon(z, polygon, local_boxes))
            
            if layer_line_starts:
                layer_dict[z].extend(layer_line_starts)

        return layer_dict

    def _slice_polygon(self, z: float, polygon: Polygon, local_boxes: dict) -> List[Point]:
        if isinstance(polygon, MultiPolygon):
            starts = []
            for sub_poly in polygon.geoms:
                starts.extend(self._slice_polygon(z, sub_poly, local_boxes))
            return starts
        if isinstance(polygon, GeometryCollection):
            starts = []
            for geom in polygon.geoms:
                starts.extend(self._slice_polygon(z, geom, local_boxes))
            return starts
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return []

        coords = list(polygon.exterior.coords)
        if len(coords) < 2:
            return []

        polygon_start_points = []
        prev_coord = (coords[0][0], coords[0][1], z)
        prev_point: Point | None = None

        prev_point = self._check_pos(prev_coord, prev_point, local_boxes, 1)
        if prev_point is not None:
            polygon_start_points.append(prev_point)
        past_coords = []
        for coord2d in coords:
            coord = (coord2d[0], coord2d[1], z)
            for pos in self._points_to_point(prev_coord, coord):
                past_coords.append(pos)
                multiple = 1
                if prev_point != None:
                    multiple = self.long_line_sample_bias

                point = self._check_pos(pos, prev_point, local_boxes, multiple=multiple)
                if point is not None:
                    if prev_point is None:
                        start_point = self._grow_line_backwards(past_coords, point, local_boxes=local_boxes)
                        if start_point != None:
                            polygon_start_points.append(start_point)
                prev_point = point
            prev_coord = coord
            
        for line_start in polygon_start_points[:]:
            if not self._check_min_line_length(line_start):
                polygon_start_points.remove(line_start)
                self._remove_line(line_start, local_boxes)
                
        return polygon_start_points 

    def _grow_line_backwards(self, positions: list[tuple[float, float, float]], end_point: Point, local_boxes: dict) -> Point | None:
        #skiping last element because it is the current element from which to grow backwards
        for pos in reversed(positions[:-1]):
            # This scenario seems impossible the more I think about it, but it can't hurt, right?
            point = self._check_if_pos_is_point(pos, local_boxes=local_boxes)
            if point != None:
                print("the impossible is possible why did it happen??")
                point.next = end_point
                return None

            point = self._check_pos(pos, None, local_boxes, multiple=self.long_line_sample_bias)
            if point == None:
                return end_point
            end_point.prev = point
            point.next = end_point
            end_point = point
        return end_point



    def _check_if_pos_is_point(self, pos: tuple[float, float, float], local_boxes: dict) -> Point | None:
        box_coord = self._to_box_coord(pos)
        if box_coord in local_boxes:
            for point in local_boxes[box_coord]:
                if point.x == pos[0] and point.y == pos[1] and point.z == pos[2]:
                    return point
        return None




    def _remove_line(self, line_start: Point | None, local_boxes: dict):
        while line_start is not None:
            box_coord = self._to_box_coord(line_start.to_pos())
            if box_coord in local_boxes and line_start in local_boxes[box_coord]:
                local_boxes[box_coord].remove(line_start)
            line_start = line_start.next
    
    def _check_min_line_length(self, line_start: Point | None) -> bool:
        for _ in range(self.min_line_segments):
            if line_start is None:
                return False
            line_start = line_start.next
        return True

    def _check_pos(self, pos: tuple[float, float, float], prev: Point | None, local_boxes: dict, multiple: float) -> Point | None:
        box_coord = self._to_box_coord(pos)
        x_range = range(-1, 2)
        y_range = range(-1, 2)
        z_range = range(-1, 1)
        for x, y, z in itertools.product(x_range, y_range, z_range):
            coord = (box_coord[0]+x, box_coord[1]+y, box_coord[2]+z)
            if self._check_box_against_pos(coord, pos, local_boxes, mutliple=multiple):
                return None
                
        point = Point(pos[0], pos[1], pos[2], prev)
        if prev is not None:
            prev.next = point

        if box_coord not in local_boxes:
            local_boxes[box_coord] = []
        local_boxes[box_coord].append(point)
        return point

    def _check_box_against_pos(self, box_coord: tuple[int, int, int], pos: tuple[float, float, float], local_boxes: dict, mutliple: float) -> bool:
        if box_coord not in local_boxes:
            return False
            
        box = local_boxes[box_coord]
        denom_width = self.horizontal_detection_distance ** 2
        denom_height = self.z_detection_distance ** 2
        
        for point in box:
            if point.z == pos[2]:
                continue
            x_component = ((point.x - pos[0]) ** 2) / denom_width
            y_component = ((point.y - pos[1]) ** 2) / denom_width
            z_component = ((point.z - pos[2]) ** 2) / denom_height

            result = x_component + y_component + z_component
            if result < 0.99 * mutliple:
                return True
        return False

    def _to_box_coord(self, pos: tuple[float, float, float]) -> tuple[int, int, int]:
        box_x = int(pos[0] // self.box_width)
        box_y = int(pos[1] // self.box_width)
        box_z = int(pos[2] // self.box_height)
        return (box_x, box_y, box_z)

    def _points_to_point(self, start_point: tuple[float, float, float], stop_point: tuple[float, float, float]) -> List[tuple[float, float, float]]:
        dx = stop_point[0] - start_point[0]
        dy = stop_point[1] - start_point[1]
        dz = stop_point[2] - start_point[2]
        
        total_dist = math.hypot(dx, dy, dz)
        if total_dist <= self.point_spacing / 2 or total_dist == 0:
            return [stop_point]
        
        num_segments = max(1, round(total_dist / self.point_spacing))
        points = []
        for i in range(1, num_segments + 1):
            t = i / num_segments
            x = start_point[0] + t * dx
            y = start_point[1] + t * dy
            z = start_point[2] + t * dz
            points.append((x, y, z))
            
        return points

    def _polygons_at_z(self, mesh: trimesh.Trimesh, z: float) -> List[Polygon]:
        segments_3d = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=[0.0, 0.0, 1.0],
            plane_origin=[0.0, 0.0, float(z)],
        )

        if segments_3d is None or len(segments_3d) == 0:
            return []

        scale = 1_000_000.0
        edge_set = set()
        neighbors = {}

        def key_from_point(p):
            return (int(round(float(p[0]) * scale)), int(round(float(p[1]) * scale)))

        def edge_key(a_key, b_key):
            return (a_key, b_key) if a_key <= b_key else (b_key, a_key)

        for seg in segments_3d:
            a_key = key_from_point(seg[0])
            b_key = key_from_point(seg[1])
            if a_key == b_key:
                continue
            e_key = edge_key(a_key, b_key)
            if e_key in edge_set:
                continue
            edge_set.add(e_key)
            neighbors.setdefault(a_key, []).append(b_key)
            neighbors.setdefault(b_key, []).append(a_key)

        if not edge_set:
            return []

        visited_edges = set()
        polygons: List[Polygon] = []

        for a_start, b_start in list(edge_set):
            start_edge = edge_key(a_start, b_start)
            if start_edge in visited_edges:
                continue

            loop = [a_start, b_start]
            visited_edges.add(start_edge)
            prev = a_start
            curr = b_start

            while True:
                candidates = []
                for nxt in neighbors.get(curr, []):
                    e = edge_key(curr, nxt)
                    if nxt != prev and e not in visited_edges:
                        candidates.append(nxt)

                if not candidates:
                    break

                nxt = candidates[0]
                visited_edges.add(edge_key(curr, nxt))
                loop.append(nxt)
                prev, curr = curr, nxt

                if curr == loop[0]:
                    coords = [(k[0] / scale, k[1] / scale) for k in loop[:-1]]
                    if len(coords) >= 3:
                        poly = Polygon(coords)
                        # Filter out tiny artifacts smaller than ~0.04mm^2 (e.g., 0.4mm nozzle * 0.1mm segment)
                        min_area = self.nozzle_diameter * 0.1
                        if poly.is_valid and poly.area > min_area:
                            polygons.append(poly)
                    break

        buffered_polygons = []
        half_nozzle = self.nozzle_diameter / 2.0

        for poly in polygons:
            shrunk_poly = poly.buffer(-half_nozzle, join_style=2)
            buffered_polygons.extend(self._flatten_geometries(shrunk_poly))

        return buffered_polygons

    def _flatten_geometries(self, geometry):
        if geometry.is_empty:
            return []
        if isinstance(geometry, Polygon):
            return [geometry]
        if isinstance(geometry, MultiPolygon):
            return [poly for poly in geometry.geoms if poly.is_valid and poly.area > 0]
        if isinstance(geometry, GeometryCollection):
            polys = []
            for geom in geometry.geoms:
                polys.extend(self._flatten_geometries(geom))
            return polys
        return []

    def get_layer_count(self) -> int:
        """Get number of layers"""
        return len(self.layers)
