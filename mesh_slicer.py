from __future__ import annotations  

import json
from typing import List, Dict, Any, Optional
import trimesh
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import numpy as np
import math
import itertools


class PrinterSettings:
    """Loads and manages printer configuration from JSON"""

    def __init__(self, settings_file: str):
        """Load printer settings from JSON file"""
        with open(settings_file, 'r') as f:
            self.config = json.load(f)

    def get(self, key: str, default=None):
        """Get a setting value"""
        return self.config.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get all settings"""
        return self.config

    def validate(self) -> bool:
        """Validate required settings are present"""
        required = [
            'bed_width', 'bed_height', 'bed_depth',
            'nozzle_diameter', 'default_layer_height'
        ]
        return all(key in self.config for key in required)


class Point:
    """Represents a point along a fillament line to be printed"""

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
        """
        Initialize slicer with STL file and printer settings

        Args:
            stl_file: Path to STL file
            printer_settings: PrinterSettings object with printer configuration
        """
        loaded = trimesh.load_mesh(stl_file)
        if isinstance(loaded, trimesh.Scene):
            if hasattr(loaded, "to_mesh"):
                self.mesh = loaded.to_mesh()
            else:
                self.mesh = loaded.dump(concatenate=True)
        else:
            self.mesh = loaded
        self.printer = printer_settings
        self.layer_height = printer_settings.get('default_layer_height', 0.2)
        self.bed_width = printer_settings.get('bed_width')
        self.bed_height = printer_settings.get('bed_height')
        self.nozzle_diameter = printer_settings.get('nozzle_diameter', 0.4)
        self.box_height = self.layer_height
        self.box_width = self.nozzle_diameter
        self.boxes:dict[tuple[int, int, int], list[Point]] = {}
        self.layers: list[list[Point]] = []
        self.point_spacing = 0.1
        self.z_sample_height = 0.03
        self.z_detection_distance = self.layer_height
        self.horizontal_detection_distance = 0.5 * self.nozzle_diameter
        self.min_line_segments = 5

    def slice(self) -> List[List[Point]]:
        """
        Slice the mesh into layers

        Returns:
            List of layers, each layer contains a list of Shapely Polygons
        """
        z_min = float(self.mesh.bounds[0][2])
        z_max = float(self.mesh.bounds[1][2])

        print(f"Model Z range: {z_min:.2f} to {z_max:.2f} mm")
        print(f"Layer height: {self.layer_height} mm")
        self.layers = []

        # Avoid slicing exactly on triangle vertices/edges.
        epsilon = 1e-6
        z_start = z_min + epsilon
        z_stop = z_max - epsilon
        if z_stop <= z_start:
            return []

        # Generate layer z-coordinates
        z_values = np.arange(z_start, z_stop + self.layer_height, self.z_sample_height)
        print(f"Total layers: {len(z_values)}")

        for i, z in enumerate(z_values):
            layer_line_starts = self._slice_at_z(z)
            if layer_line_starts != []:
                self.layers.append(layer_line_starts)
            if (i + 1) % 10 == 0:
                print(f"Processed layer {i + 1}/{len(z_values)}")

        return self.layers

    def _slice_at_z(self, z: float) -> List[Point]:
        polygons = self._polygons_at_z(z)
        layer_line_starts = []
        for polygon in polygons:
            layer_line_starts.extend(self._slice_polygon(z, polygon))
        return layer_line_starts

    def _slice_polygon(self, z: float, polygon: Polygon) -> List[Point]:
        """Generate Points to trace a polygon"""
        if isinstance(polygon, MultiPolygon):
            for sub_poly_idx, sub_poly in enumerate(polygon.geoms):
                self._slice_polygon(z, sub_poly)
            return []
        if isinstance(polygon, GeometryCollection):
            for sub_poly_idx, geom in enumerate(polygon.geoms):
                self._slice_polygon(z, geom)
            return []
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return []

        coords = list(polygon.exterior.coords)

        if len(coords) < 2:
            return []

        # Create start point list
        polygon_start_points = []

        # Move to start position
        prev_coord = (coords[0][0], coords[0][1], z)
        prev_point: Point | None = None

        # check initial pos
        prev_point = self._check_pos(prev_coord, prev_point)
        if prev_point != None:
            polygon_start_points.append(prev_point)

        for coord2d in coords:
            coord = (coord2d[0], coord2d[1], z)
            for pos in self._points_to_point(prev_coord, coord):
                point = self._check_pos(pos, prev_point)
                if point != None:
                    if prev_point == None:
                        polygon_start_points.append(point)

                prev_point = point
            prev_coord = coord
        for line_start in polygon_start_points[:]:
            if not self._check_min_line_length(line_start):
                polygon_start_points.remove(line_start)
                self._remove_line(line_start)
        return polygon_start_points 

    def _remove_line(self, line_start: Point | None):
        while line_start != None:
            box_coord = self._to_box_coord(line_start.to_pos())
            self.boxes[box_coord].remove(line_start)
            line_start = line_start.next
    
    def _check_min_line_length(self, line_start: Point | None) -> bool:
        """checks whether the line has more segments than minimum line segments. 
        Returns True if it has enogh."""
        for i in range(self.min_line_segments):
            if (line_start == None):
                return False
            line_start = line_start.next
        return True

    def _check_pos(self, pos: tuple[float, float, float], prev: Point | None) -> Point | None:
        box_coord = self._to_box_coord(pos)
        x_range = range(-1, 2)  # -1, 0, 1
        y_range = range(-1, 2)  # -1, 0, 1
        z_range = range(-1, 1)  # -1, 0

        # itertools.product creates the Cartesian product (all combinations)
        for x, y, z in itertools.product(x_range, y_range, z_range):
            coord = (box_coord[0]+x, box_coord[1]+y, box_coord[2]+z)
            if(self._check_box_against_pos(coord, pos)):
                return None
        point = Point(pos[0], pos[1], pos[2], prev)
        if prev != None:
            prev.next = point

        if not box_coord in self.boxes:
            self.boxes[box_coord] = []
        box = self.boxes[box_coord]
        box.append(point)
        return point

    def _check_box_against_pos(self, box_coord: tuple[int, int, int], pos: tuple[float, float, float]) -> bool:
        """checks if a box contains a Point which overlaps with the pos position
           returns false if not and true if it overlaps
        """
        if not box_coord in self.boxes:
            return False
        box = self.boxes[box_coord]
        denom_width = self.horizontal_detection_distance ** 2
        denom_height = self.z_detection_distance ** 2
        for point in box:
            if (point.z == pos[2]):
                continue
            # Calculate each component of the ellipsoid distance formula
            x_component = ((point.x - pos[0]) ** 2) / denom_width
            y_component = ((point.y - pos[1]) ** 2) / denom_width
            z_component = ((point.z - pos[2]) ** 2) / denom_height

            # Total sum
            result = x_component + y_component + z_component

            if(result < 0.99):
                return True
        return False

    def _to_box_coord(self, pos: tuple[float, float, float]) -> tuple[int, int, int]:
        # Use floor division to find which integer grid cell the float falls into
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

    def _polygons_at_z(self, z: float) -> List[Polygon]:
        """Get cross-section polygons at a specific Z height"""
        segments_3d = trimesh.intersections.mesh_plane(
            mesh=self.mesh,
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
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)
                    break

        buffered_polygons = []
        half_nozzle = self.nozzle_diameter / 2.0

        for poly in polygons:
            shrunk_poly = poly.buffer(-half_nozzle, join_style=2)
            buffered_polygons.extend(self._flatten_geometries(shrunk_poly))

        return buffered_polygons

    def _flatten_geometries(self, geometry):
        """Return only Polygon parts from a Shapely geometry."""
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
