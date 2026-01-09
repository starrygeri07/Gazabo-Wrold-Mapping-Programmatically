#!/usr/bin/env python3
"""
Automatically generate ground truth occupancy grid maps from Gazebo world files.

This tool parses a Gazebo .world file (SDF format), extracts all STATIC geometry, and generates a pixel-perfect occupancy grid map for ROS2 navigation.

Features:
    - Automatically detects and EXCLUDES dynamic obstacles (models with plugins)
    - Handles nested coordinate transforms (model -> link -> collision)
    - Auto-calculates optimal map bounds from world geometry
    - Configurable resolution and margin

Usage:
    python3 generate_map_from_gazebo_world.py /path/to/world.world

Output:
    - <world_name>_groundtruth.pgm  (occupancy grid image)
    - <world_name>_groundtruth.yaml (ROS2 map metadata)
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Pose:
    """6-DOF pose (x, y, z, roll, pitch, yaw)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    @classmethod
    def from_string(cls, pose_str: str) -> 'Pose':
        """Parse pose from SDF string format: 'x y z roll pitch yaw'."""
        if not pose_str or not pose_str.strip():
            return cls()
        parts = pose_str.strip().split()
        values = [float(p) for p in parts]
        # Pad with zeros if incomplete
        while len(values) < 6:
            values.append(0.0)
        return cls(*values[:6])
    
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Apply 2D transformation (translation + yaw rotation) to a point."""
        # Rotation matrix for yaw (rotation around Z-axis)
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        
        # Rotate then translate
        x_new = x * cos_yaw - y * sin_yaw + self.x
        y_new = x * sin_yaw + y * cos_yaw + self.y
        
        return x_new, y_new
    
    def combine(self, child: 'Pose') -> 'Pose':
        """Combine this pose with a child pose."""
        # Transform child position by parent
        new_x, new_y = self.transform_point(child.x, child.y)
        
        return Pose(
            x=new_x,
            y=new_y,
            z=self.z + child.z,
            roll=self.roll + child.roll,
            pitch=self.pitch + child.pitch,
            yaw=self.yaw + child.yaw
        )


@dataclass
class BoxGeometry:
    """Axis-aligned box obstacle."""
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    yaw: float = 0.0
    name: str = ""
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """Get the 4 corners of the (possibly rotated) box in world coordinates."""
        half_x = self.size_x / 2.0
        half_y = self.size_y / 2.0
        
        local_corners = [
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y)
        ]
        
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        
        world_corners = []
        for lx, ly in local_corners:
            wx = lx * cos_yaw - ly * sin_yaw + self.center_x
            wy = lx * sin_yaw + ly * cos_yaw + self.center_y
            world_corners.append((wx, wy))
        
        return world_corners
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box: (x_min, x_max, y_min, y_max)."""
        corners = self.get_corners()
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), max(xs), min(ys), max(ys)


@dataclass
class CylinderGeometry:
    """Cylindrical obstacle."""
    center_x: float
    center_y: float
    radius: float
    name: str = ""
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box: (x_min, x_max, y_min, y_max)."""
        return (
            self.center_x - self.radius,
            self.center_x + self.radius,
            self.center_y - self.radius,
            self.center_y + self.radius
        )


@dataclass
class WorldGeometry:
    """Container for all extracted world geometry."""
    boxes: List[BoxGeometry] = field(default_factory=list)
    cylinders: List[CylinderGeometry] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)
    world_name: str = "unknown"


# =============================================================================
# SDF PARSER
# =============================================================================

class GazeboWorldParser:
    """
    Parse Gazebo .world files (SDF format) and extract static geometry.
    
    Automatically excludes:
    - Models with <plugin> tags (dynamic/controlled objects)
    - Models without <static>1</static> tag
    - Non-physical elements (sun, ground_plane, ground, cameras)
    """
    
    # Models to always ignore (Gazebo built-ins)
    IGNORED_MODELS = {
        'sun', 'ground_plane', 'ground', 'camera', 'user_camera'
    }
    
    # Plugin indicators that suggest dynamic behavior
    DYNAMIC_PLUGIN_PATTERNS = [
        'planar_move',
        'diff_drive', 
        'skid_steer',
        'joint_state',
        'force',
        'velocity',
        'position',
        'actor'
    ]
    
    def __init__(self, world_file: str, verbose: bool = True):
        self.world_file = world_file
        self.verbose = verbose
        self.geometry = WorldGeometry()
        
    def parse(self) -> WorldGeometry:
        """Parse the world file and extract all static geometry."""
        if self.verbose:
            print(f"Parsing: {self.world_file}")
        
        try:
            tree = ET.parse(self.world_file)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML: {e}")
        
        # Find the world element
        world_elem = root.find('.//world')
        if world_elem is None:
            if root.tag == 'world':
                world_elem = root
            else:
                raise ValueError("No <world> element found in SDF file")
        
        # Extract world name
        self.geometry.world_name = world_elem.get('name', 'world')
        
        # Process all models
        for model in world_elem.findall('.//model'):
            self._process_model(model)
        
        if self.verbose:
            print(f"\nExtracted geometry:")
            print(f"  Static boxes: {len(self.geometry.boxes)}")
            print(f"  Static cylinders: {len(self.geometry.cylinders)}")
            print(f"  Excluded models: {len(self.geometry.excluded_models)}")
            if self.geometry.excluded_models:
                for name in self.geometry.excluded_models:
                    print(f"    - {name}")
        
        return self.geometry
    
    def _process_model(self, model_elem: ET.Element, parent_pose: Optional[Pose] = None):
        """Process a single model element."""
        model_name = model_elem.get('name', 'unnamed')
        
        # Check if this is a built-in model to ignore
        if model_name.lower() in self.IGNORED_MODELS:
            return
        
        # Check if model is included from URI (like sun, ground_plane)
        include_elem = model_elem.find('include')
        if include_elem is not None:
            uri_elem = include_elem.find('uri')
            if uri_elem is not None and uri_elem.text:
                uri_lower = uri_elem.text.lower()
                if any(ignored in uri_lower for ignored in self.IGNORED_MODELS):
                    return
        
        # Check if model has dynamic plugins
        if self._has_dynamic_plugin(model_elem):
            self.geometry.excluded_models.append(f"{model_name} (has motion plugin)")
            return
        
        # Check if model is static
        static_elem = model_elem.find('static')
        is_static = static_elem is not None and static_elem.text is not None and static_elem.text.strip() == '1'
        
        if not is_static:
            # Check if it has any plugin at all (might be controllable)
            if model_elem.find('.//plugin') is not None:
                self.geometry.excluded_models.append(f"{model_name} (non-static with plugin)")
                return
        
        # Get model pose
        model_pose = self._get_pose(model_elem)
        if parent_pose:
            model_pose = parent_pose.combine(model_pose)
        
        # Process all links in the model
        links = model_elem.findall('link')
        if not links:
            self._extract_geometry_from_element(model_elem, model_pose, model_name)
        else:
            for link in links:
                self._process_link(link, model_pose, model_name)
        
        # Process nested models
        for nested_model in model_elem.findall('model'):
            self._process_model(nested_model, model_pose)
    
    def _process_link(self, link_elem: ET.Element, model_pose: Pose, model_name: str):
        """Process a link element within a model."""
        link_name = link_elem.get('name', 'unnamed_link')
        link_pose = self._get_pose(link_elem)
        combined_pose = model_pose.combine(link_pose)
        
        # Extract geometry from collisions (preferred) or visuals
        collisions = link_elem.findall('collision')
        if collisions:
            for collision in collisions:
                self._extract_geometry_from_element(
                    collision, combined_pose, f"{model_name}/{link_name}"
                )
        else:
            # Fall back to visual geometry
            visuals = link_elem.findall('visual')
            for visual in visuals:
                self._extract_geometry_from_element(
                    visual, combined_pose, f"{model_name}/{link_name}"
                )
    
    def _extract_geometry_from_element(self, elem: ET.Element, parent_pose: Pose, name: str):
        """Extract geometry from a collision or visual element."""
        # Get element-level pose
        elem_pose = self._get_pose(elem)
        final_pose = parent_pose.combine(elem_pose)
        
        # Find geometry
        geometry = elem.find('geometry')
        if geometry is None:
            geometry = elem
        
        # Try to extract box
        box = geometry.find('box')
        if box is not None:
            size_elem = box.find('size')
            if size_elem is not None and size_elem.text:
                sizes = [float(s) for s in size_elem.text.strip().split()]
                if len(sizes) >= 2:
                    self.geometry.boxes.append(BoxGeometry(
                        center_x=final_pose.x,
                        center_y=final_pose.y,
                        size_x=sizes[0],
                        size_y=sizes[1],
                        yaw=final_pose.yaw,
                        name=name
                    ))
            return
        
        # Try to extract cylinder
        cylinder = geometry.find('cylinder')
        if cylinder is not None:
            radius_elem = cylinder.find('radius')
            if radius_elem is not None and radius_elem.text:
                radius = float(radius_elem.text.strip())
                self.geometry.cylinders.append(CylinderGeometry(
                    center_x=final_pose.x,
                    center_y=final_pose.y,
                    radius=radius,
                    name=name
                ))
            return
        
        # Try to extract sphere (treat as cylinder with radius)
        sphere = geometry.find('sphere')
        if sphere is not None:
            radius_elem = sphere.find('radius')
            if radius_elem is not None and radius_elem.text:
                radius = float(radius_elem.text.strip())
                self.geometry.cylinders.append(CylinderGeometry(
                    center_x=final_pose.x,
                    center_y=final_pose.y,
                    radius=radius,
                    name=f"{name}_sphere"
                ))
            return
    
    def _get_pose(self, elem: ET.Element) -> Pose:
        """Extract pose from an element."""
        pose_elem = elem.find('pose')
        if pose_elem is not None and pose_elem.text:
            return Pose.from_string(pose_elem.text)
        return Pose()
    
    def _has_dynamic_plugin(self, model_elem: ET.Element) -> bool:
        """Check if model has plugins indicating dynamic behavior."""
        for plugin in model_elem.findall('.//plugin'):
            filename = plugin.get('filename', '').lower()
            for pattern in self.DYNAMIC_PLUGIN_PATTERNS:
                if pattern in filename:
                    return True
            
            # Check plugin name
            plugin_name = plugin.get('name', '').lower()
            for pattern in self.DYNAMIC_PLUGIN_PATTERNS:
                if pattern in plugin_name:
                    return True
        
        return False


# =============================================================================
# MAP GENERATOR
# =============================================================================

class OccupancyGridGenerator:
    """Generate occupancy grid from extracted world geometry."""
    
    def __init__(self, 
                 geometry: WorldGeometry,
                 resolution: float = 0.05,
                 margin: float = 0.3,
                 verbose: bool = True):
        """
        Args:
            geometry: Extracted world geometry
            resolution: Map resolution in meters/pixel (default: 0.05 = 5cm)
            margin: Extra margin around world bounds in meters (default: 0.3m)
            verbose: Print progress information
        """
        self.geometry = geometry
        self.resolution = resolution
        self.margin = margin
        self.verbose = verbose
        
        # Computed values
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0
        self.grid = None
    
    def generate(self) -> np.ndarray:
        """
        Generate the occupancy grid.
        
        The map is generated such that:
        1. It is symmetric around the world origin (0,0)
        2. World origin (0,0) is at the CENTER of a pixel (critical for path planning)
        3. The map covers all geometry plus the specified margin
        """
        # Calculate world bounds from geometry
        bounds = self._calculate_bounds()
        if bounds is None:
            raise ValueError("No geometry found - cannot generate map")
        
        x_min, x_max, y_min, y_max = bounds
        
        # Calculate the maximum extent from origin in each direction
        # This ensures the map is symmetric around (0,0)
        geom_extent_x = max(abs(x_min), abs(x_max))
        geom_extent_y = max(abs(y_min), abs(y_max))
        
        # Add margin to the extent
        total_extent_x = geom_extent_x + self.margin
        total_extent_y = geom_extent_y + self.margin
        
        # Calculate the number of pixels needed for ONE HALF of the map
        half_pixels_x = int(np.ceil(total_extent_x / self.resolution))
        half_pixels_y = int(np.ceil(total_extent_y / self.resolution))
        
        # Total map dimensions: 2 * half + 1
        self.width = 2 * half_pixels_x + 1
        self.height = 2 * half_pixels_y + 1

        # Calculate the origin so that world (0,0) is at the CENTER of the center pixel
        self.origin_x = -self.width * self.resolution / 2.0
        self.origin_y = -self.height * self.resolution / 2.0
        
        # Calculate actual coverage for display
        actual_extent_x = half_pixels_x * self.resolution
        actual_extent_y = half_pixels_y * self.resolution
        
        if self.verbose:
            print(f"\nMap configuration:")
            print(f"  Geometry bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
            print(f"  Geometry extent from origin: ±{geom_extent_x:.3f}m (X), ±{geom_extent_y:.3f}m (Y)")
            print(f"  Margin: {self.margin}m")
            print(f"  Actual map extent: ±{actual_extent_x:.3f}m (X), ±{actual_extent_y:.3f}m (Y)")
            print(f"  Resolution: {self.resolution}m/pixel")
            print(f"  Map size: {self.width} x {self.height} pixels")
            print(f"  Origin: ({self.origin_x:.3f}, {self.origin_y:.3f})")
            print(f"  World (0,0) at pixel: ({self.width//2}, {self.height//2})")
        
        # Create empty grid (0 = free space)
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Draw all geometry
        if self.verbose:
            print(f"\nDrawing geometry...")
        
        for box in self.geometry.boxes:
            self._draw_box(box)
        
        for cyl in self.geometry.cylinders:
            self._draw_cylinder(cyl)
        
        # Statistics
        occupied = np.sum(self.grid == 100)
        total = self.width * self.height
        
        if self.verbose:
            print(f"\nMap statistics:")
            print(f"  Total cells: {total:,}")
            print(f"  Occupied: {occupied:,} ({100*occupied/total:.1f}%)")
            print(f"  Free: {total-occupied:,} ({100*(total-occupied)/total:.1f}%)")
        
        return self.grid
    
    def _calculate_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Calculate the bounding box of all geometry."""
        all_bounds = []
        
        for box in self.geometry.boxes:
            all_bounds.append(box.get_bounds())
        
        for cyl in self.geometry.cylinders:
            all_bounds.append(cyl.get_bounds())
        
        if not all_bounds:
            return None
        
        x_min = min(b[0] for b in all_bounds)
        x_max = max(b[1] for b in all_bounds)
        y_min = min(b[2] for b in all_bounds)
        y_max = max(b[3] for b in all_bounds)
        
        return x_min, x_max, y_min, y_max
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy
    
    def _draw_box(self, box: BoxGeometry):
        """Draw a box onto the grid."""
        if self.grid is None:
            raise ValueError("Grid not initialized - call generate() first")
        
        # Handle rotated boxes by checking each cell against the rotated rectangle
        x_min, x_max, y_min, y_max = box.get_bounds()
        
        gx_min, gy_min = self._world_to_grid(x_min, y_min)
        gx_max, gy_max = self._world_to_grid(x_max, y_max)
        
        # Clamp to grid bounds
        gx_min = max(0, gx_min)
        gx_max = min(self.width - 1, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(self.height - 1, gy_max)
        
        # For rotated boxes, check each cell
        if abs(box.yaw) > 0.01:  # Has significant rotation
            half_x = box.size_x / 2.0
            half_y = box.size_y / 2.0
            cos_yaw = np.cos(-box.yaw)  # Inverse rotation
            sin_yaw = np.sin(-box.yaw)
            
            for gy in range(gy_min, gy_max + 1):
                for gx in range(gx_min, gx_max + 1):
                    wx = self.origin_x + (gx + 0.5) * self.resolution
                    wy = self.origin_y + (gy + 0.5) * self.resolution
                    
                    # Transform to box-local coordinates
                    dx = wx - box.center_x
                    dy = wy - box.center_y
                    local_x = dx * cos_yaw - dy * sin_yaw
                    local_y = dx * sin_yaw + dy * cos_yaw
                    
                    # Check if inside box
                    if abs(local_x) <= half_x and abs(local_y) <= half_y:
                        self.grid[gy, gx] = 100
        else:
            # Axis-aligned box - simple fill
            self.grid[gy_min:gy_max+1, gx_min:gx_max+1] = 100
    
    def _draw_cylinder(self, cyl: CylinderGeometry):
        """Draw a cylinder onto the grid."""
        if self.grid is None:
            raise ValueError("Grid not initialized - call generate() first")
        
        x_min, x_max, y_min, y_max = cyl.get_bounds()
        
        gx_min, gy_min = self._world_to_grid(x_min, y_min)
        gx_max, gy_max = self._world_to_grid(x_max, y_max)
        
        gx_min = max(0, gx_min)
        gx_max = min(self.width - 1, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(self.height - 1, gy_max)
        
        # Check each cell in bounding box
        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                wx = self.origin_x + (gx + 0.5) * self.resolution
                wy = self.origin_y + (gy + 0.5) * self.resolution
                
                dist = np.sqrt((wx - cyl.center_x)**2 + (wy - cyl.center_y)**2)
                if dist <= cyl.radius:
                    self.grid[gy, gx] = 100
    
    def save_pgm(self, filename: str):
        """Save the grid as a PGM file (ROS map format)."""
        if self.grid is None:
            raise ValueError("Grid not generated - call generate() first")
        
        # Convert: occupied (100) -> black (0), free (0) -> white (254)
        pgm_data = np.where(self.grid >= 50, 0, 254).astype(np.uint8)
        
        # Flip vertically (PGM row 0 is top, ROS map row 0 is bottom)
        pgm_data = np.flipud(pgm_data)
        
        with open(filename, 'wb') as f:
            f.write(f"P5\n{self.width} {self.height}\n255\n".encode())
            f.write(pgm_data.tobytes())
        
        if self.verbose:
            print(f"Saved: {filename}")
    
    def save_yaml(self, yaml_filename: str, pgm_filename: str):
        """Save the map metadata as a YAML file."""
        pgm_basename = os.path.basename(pgm_filename)
        
        # We round to 6 decimal places for clean output
        origin_x_clean = round(self.origin_x, 6)
        origin_y_clean = round(self.origin_y, 6)
        
        yaml_content = f"""image: {pgm_basename}
mode: trinary
resolution: {self.resolution}
origin: [{origin_x_clean}, {origin_y_clean}, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
        
        with open(yaml_filename, 'w') as f:
            f.write(yaml_content)
        
        if self.verbose:
            print(f"Saved: {yaml_filename}")


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate ROS2 occupancy grid maps from Gazebo world files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my_world.world
  %(prog)s /path/to/world.world --resolution 0.05
  %(prog)s world.world --output-dir ./maps/ --margin 0.5

Output:
  Creates <world_name>_groundtruth.pgm and <world_name>_groundtruth.yaml

Notes:
  - Dynamic obstacles (models with motion plugins) are automatically excluded
  - Only static geometry is included in the map
  - Supports box, cylinder, and sphere geometries
  - World origin (0,0) is guaranteed to be at the center of a pixel
        """
    )
    
    parser.add_argument(
        'world_file',
        help='Path to the Gazebo .world file'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=float,
        default=0.05,
        help='Map resolution in meters/pixel (default: 0.05)'
    )
    
    parser.add_argument(
        '--margin', '-m',
        type=float,
        default=0.5,
        help='Extra margin around world bounds in meters (default: 0.5)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for map files (default: current directory)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.world_file):
        print(f"Error: File not found: {args.world_file}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("GAZEBO WORLD TO OCCUPANCY GRID MAP GENERATOR")
        print("=" * 60)
        print()
    
    try:
        # Parse world file
        parser_obj = GazeboWorldParser(args.world_file, verbose=verbose)
        geometry = parser_obj.parse()
        
        # Check if we found any geometry
        if not geometry.boxes and not geometry.cylinders:
            print("\nWarning: No static geometry found in world file!")
            print("The map will be empty. Check that your world has static obstacles.")
        
        # Generate map
        generator = OccupancyGridGenerator(
            geometry,
            resolution=args.resolution,
            margin=args.margin,
            verbose=verbose
        )
        generator.generate()
        
        # Determine output filenames
        world_name = Path(args.world_file).stem
        
        pgm_filename = os.path.join(args.output_dir, f"{world_name}_groundtruth.pgm")
        yaml_filename = os.path.join(args.output_dir, f"{world_name}_groundtruth.yaml")
        
        # Save files
        if verbose:
            print()
        generator.save_pgm(pgm_filename)
        generator.save_yaml(yaml_filename, pgm_filename)
        
        if verbose:
            print()
            print("=" * 60)
            print("MAP GENERATION COMPLETE!")
            print("=" * 60)
            print()
            print("Next steps:")
            print(f"  1. Copy files to your ROS2 maps directory")
            print(f"  2. Update your launch file to use: {os.path.basename(yaml_filename)}")
            print(f"  3. Rebuild your package if needed (Reccomended)")
            print()
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
