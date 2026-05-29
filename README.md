# Autonomous Maze Solving Robot Agent

Computer vision and robotics pipeline that captures a physical maze, extracts its geometry, plans a collision-free path, and converts the path into robot motion commands.

## Overview

This project connects perception, planning, and physical robot execution:

1. Capture a maze image from a camera.
2. Detect maze corners and apply a homography.
3. Convert the warped image into a grid/wall representation.
4. Detect start and goal markers.
5. Solve the maze using graph search.
6. Overlay the planned path for debugging.
7. Send motion waypoints to a Dobot robot arm.

## Key Features

- OpenCV-based camera capture and perspective correction.
- Grid extraction from real-world maze images.
- BFS path planning over the detected maze topology.
- Visual debugging overlays for corners, walls, cells, and solution path.
- Robot motion scripts for executing the generated path.

## Tech Stack

Python, OpenCV, NumPy, Pillow, pydobot, graph search, homography-based vision.

## Main Files

- `01_capture_image.py`: capture maze image.
- `02_maze_warp_from_json.py`: perspective warp from selected corners.
- `03_maze_circles_and_grid.py`: grid and wall extraction.
- `04_solve_maze.py`: path planning and solution generation.
- `05_unwrap_and_overlay_path.py`: map solution path back onto the original image.
- `Maze_Motion.py` / `robot_motion.py`: robot execution helpers.

## Representative Outputs

The repository includes generated overlays such as:

- `maze_corners_overlay.png`
- `grid_overlay.png`
- `walls_mask.png`
- `solution_overlay.png`
- `original_with_path.png`

These make the perception and planning pipeline reviewable without robot hardware.

## Safety

Robot execution should be tested first in a low-speed mode with an emergency stop or power cutoff available.

