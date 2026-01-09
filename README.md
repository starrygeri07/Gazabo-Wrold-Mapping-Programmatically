# 🗺️ Gazebo World to Map Programmatically
Automatically generate ground truth occupancy grid maps from Gazebo world files.

**Best thing**: No manual mapping through your favorite SLAM technique required!

### 🎯 So, what is this?
When working with robot simulation in Gazebo, you typically need an occupancy grid map either for localization or navigation.

The traditional approach includes:
1. Launch Gazebo with your world
2. Launch SLAM (Cartographer, gmapping, etc.)
3. Manually drive the robot around the entire Gazebo environment or let it wander on its own for a few minutes through any Autonomous Collision Avoidance algorithm.
4. Save the map
5. Worst thing: repeat every time you modify the world 😫

### ✨ Features of this tool
| Feature                                     | Description                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| 🤖 **Automatic Dynamic Obstacle Exclusion** | Detects and excludes models with motion plugins (planar_move, diff_drive, etc.) |
| 📐 **Handles Complex Geometry**             | Supports boxes, cylinders, spheres, etc                                         |
| 📏 **Auto-calculated Bounds**               | Automatically determines optimal map size from world geometry                   |
| ⚙️ **Configurable**                         | Adjustable resolution, margin, and output directory                             |
| 📦 **Almost Zero Dependencies**             | Only requires NumPy (already included with ROS2)                                |
### How to use:
- Save the python file (`generate_map_from_gazebo_world.py`) in any directory/folder you like.
- Open the terminal from that directory.
- Run the below command:
```
python3 generate_map_from_gazebo_world.py \
	/path/to/world.world
```

Example usage if your `example.world` file is in the directory "`Downloads`":
```
python3 generate_map_from_gazebo_world.py \
	~/Downloads/example.world
```

This creates:
- `example_groundtruth.pgm` — The occupancy grid image
- `example_groundtruth.yaml` — ROS2 map metadata

In the "Example" folder, I have also uploaded `baseline.world` with which you can try for yourself, our you can just check the map files to verify its outputs.

### ⚠️ Limitations
- **Mesh geometries**: Currently not supported (only box, cylinder, sphere for now)
- **Multi-floor worlds**: A map is only a generated 2D projection. I have not tested how the code deals with multiple floor, so if you find it interesting to test, lemme know of the results you get, hehe.
- **No GUI interface**. Works only through terminal. However, very simple command to use.

Feel free to use for any research of project you like.
This code tool was developed during my own research for autonomous robots.

Shared with ❤️ for the robotics simulation community. Hope you'll find it helpful.

### Stop doing SLAM manually for static simulation worlds. Life's too short 🚀
