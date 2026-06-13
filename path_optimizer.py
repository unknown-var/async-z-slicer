from __future__ import annotations
from typing import List
import math

# Import components from our newly split slicer logic
from mesh_slicer import Point, PrinterSettings

class Line:
    def __init__(self, points: list[tuple[float, float, float]]):
        self.points = points
    def get_first (self):
        return self.points[0]
    def get_last (self):
        return self.points[len(self.points) - 1]

class Loop:
    def __init__(self, points: list[tuple[float, float, float]]):
        self.points = points

class PathOptimizer:
    """Optimizes in which order lines are printed"""

    def __init__(self, layers: List[List[Point]]):
        self.layers = layers
        self._optimized_layers: list[list[list[tuple[float, float, float]]]] = []

    def optimize (self) -> list[list[list[tuple[float, float, float]]]]:
        start_point = None
        for layer in self.layers:
            optimized_layer, start_point = self._optimize_layer(layer, start_point)
            self._optimized_layers.append(optimized_layer)
        return self._optimized_layers

    def _optimize_layer(self, input_layer: list[Point], start_point: tuple[float, float, float] | None) -> tuple[list[list[tuple[float, float, float]]], tuple[float, float, float]]:
            """
            Takes the start point from the previous layer and efficiently orders paths.
            Returns the optimized list of paths and the final position of the tool head.
            """
            lines, loops = self._convert_layer(input_layer)
            
            optimized_paths: list[list[tuple[float, float, float]]] = []
            
            # If no start point is provided (e.g., very first layer), default to the 
            # first available coordinate so the algorithm has a place to begin.
            current_pos = start_point
            if current_pos is None:
                if lines:
                    current_pos = lines[0].get_first()
                elif loops:
                    current_pos = loops[0].points[0]
                else:
                    return [], (0.0, 0.0, 0.0) # Fallback for an empty layer

            # Helper function for 3D distance
            def get_dist(p1, p2):
                return math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

            unprinted_lines = list(lines)
            unprinted_loops = list(loops)

            while unprinted_lines or unprinted_loops:
                best_dist = float('inf')
                best_is_line = True
                best_line_reverse = False
                best_loop_start_idx = 0
                best_entity_idx = -1

                # 1. Check all unprinted lines (both start and end points)
                for i, line in enumerate(unprinted_lines):
                    d_first = get_dist(current_pos, line.get_first())
                    d_last = get_dist(current_pos, line.get_last())
                    
                    if d_first < best_dist:
                        best_dist = d_first
                        best_is_line = True
                        best_line_reverse = False
                        best_entity_idx = i
                        
                    if d_last < best_dist:
                        best_dist = d_last
                        best_is_line = True
                        best_line_reverse = True
                        best_entity_idx = i

                # 2. Check all unprinted loops (every single point)
                for i, loop in enumerate(unprinted_loops):
                    for j, point in enumerate(loop.points):
                        d = get_dist(current_pos, point)
                        if d < best_dist:
                            best_dist = d
                            best_is_line = False
                            best_loop_start_idx = j
                            best_entity_idx = i

                # 3. Assemble the best path found
                if best_is_line:
                    # Reverse the line if the end was closer
                    entity = unprinted_lines.pop(best_entity_idx)
                    path = entity.points[::-1] if best_line_reverse else entity.points[:]
                    optimized_paths.append(path)
                    current_pos = path[-1]
                else:
                    # Rotate the loop so it starts at the closest point
                    entity = unprinted_loops.pop(best_entity_idx)
                    pts = entity.points
                    idx = best_loop_start_idx
                    
                    # Shift array: [idx to end] + [start to idx]
                    path = pts[idx:] + pts[:idx]
                    
                    # Re-append the new start point to physically close the gap
                    path.append(path[0]) 
                    
                    optimized_paths.append(path)
                    current_pos = path[-1]

            return optimized_paths, current_pos

    def _convert_layer (self, input_layer: List[Point]) -> tuple[list[Line], list[Loop]]:
        lines: list[Line] = []
        loops: list[Loop] = []
        seam_tolerance = 1e-6
        for start_point in input_layer:
            points = self._simplify_line(start_point)
            if len(points) > 1 and math.isclose(points[0][0], points[-1][0], abs_tol=seam_tolerance) and math.isclose(points[0][1], points[-1][1], abs_tol=seam_tolerance):
                if math.isclose(points[0][2], points[-1][2], abs_tol=seam_tolerance):
                    points.pop()
                loop = Loop(points)
                loops.append(loop)
            else:
                line = Line(points)
                lines.append(line)

        return (lines, loops)
        

        
    def _simplify_line(self, start_point: Point, tolerance: float = 0.001) -> list[tuple[float, float, float]]:
        """Simplifies a linked list of points into a list of (x, y) tuples."""
        simplified_points = [(start_point.x, start_point.y, start_point.z)]
        
        if start_point.next is None:
            return simplified_points

        current = start_point.next

        while current is not None and current.next is not None:
            p1_x, p1_y, _ = simplified_points[-1] 
            p2 = current                       
            p3 = current.next                  

            dx1 = p2.x - p1_x
            dy1 = p2.y - p1_y
            dx2 = p3.x - p1_x
            dy2 = p3.y - p1_y

            cross_product = abs((dx1 * dy2) - (dy1 * dx2))

            if cross_product > tolerance:
                simplified_points.append((p2.x, p2.y, p2.z))

            current = current.next

        if current is not None:
            last_point = (current.x, current.y, current.z)
            if simplified_points[-1] != last_point:
                simplified_points.append(last_point)

        return simplified_points
