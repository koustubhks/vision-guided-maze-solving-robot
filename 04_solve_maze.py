#!/usr/bin/env python3
"""
solve_maze.py

Reads a maze JSON (like the one you posted), allows choosing start/end,
solves the maze under the rule:
  - Traversable cells have value 1 only.
  - Start and End cells may be 0.
  - Movement is 4-connected (up/down/left/right).

Outputs:
  - A PNG with the path drawn on the original image.
  - A JSON with the path pixel points (including start/end pixel coordinates).

Usage examples:
  python solve_maze.py maze.json
  python solve_maze.py maze.json --start green --end red
  python solve_maze.py maze.json --start 9,1 --end 1,9
  python solve_maze.py maze.json --out-image path_overlay.png --out-json path_points.json
"""

import json
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    value: int
    center_px: Tuple[int, int]


def load_maze(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def cells_to_grid(cells_json: List[Dict[str, Any]], rows: int, cols: int) -> List[List[Cell]]:
    grid: List[List[Cell]] = [[None for _ in range(cols)] for _ in range(rows)]
    for c in cells_json:
        cell = Cell(
            row=c["row"],
            col=c["col"],
            value=int(c["value"]),
            center_px=(int(c["center_px"][0]), int(c["center_px"][1])),
        )
        grid[cell.row][cell.col] = cell

    # quick sanity
    for r in range(rows):
        for q in range(cols):
            if grid[r][q] is None:
                raise ValueError(f"Missing cell at ({r},{q}) in JSON.")
    return grid


def nearest_cell_by_pixel(grid: List[List[Cell]], x: float, y: float) -> Tuple[int, int]:
    """Return (row, col) of the cell whose center is nearest to (x,y)."""
    best = None
    best_d2 = 1e18
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cx, cy = grid[r][c].center_px
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (r, c)
    return best


# def parse_start_end(
#     grid: List[List[Cell]],
#     data: Dict[str, Any],
#     start_arg: Optional[str],
#     end_arg: Optional[str],
# ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
#     """
#     Returns:
#       start_rc, end_rc: (row,col) of chosen start/end cells
#       start_circle_px, end_circle_px: (x,y) pixel centers of the green/red dots
#     """
#     # Default: use the provided circles (green=start, red=end)
#     circles = data.get("circles", [])
#     green = next((c for c in circles if c.get("color") == "green"), None)
#     red = next((c for c in circles if c.get("color") == "red"), None)

#     if green is None or red is None:
#         raise ValueError("JSON must include green and red circles.")

#     green_px = (float(green["center"][0]), float(green["center"][1]))
#     red_px = (float(red["center"][0]), float(red["center"][1]))

#     default_start = nearest_cell_by_pixel(grid, *green_px)
#     default_end = nearest_cell_by_pixel(grid, *red_px)

#     start_rc = default_start
#     end_rc = default_end

#     # Allow overrides:
#     # --start green|red|r,c   (same for --end)
#     def parse_point(arg: Optional[str], default_rc: Tuple[int, int]) -> Tuple[int, int]:
#         if not arg:
#             return default_rc
#         a = arg.strip().lower()
#         if a == "green":
#             return default_start
#         if a == "red":
#             return default_end
#         if "," in a:
#             r_str, c_str = a.split(",", 1)
#             return (int(r_str), int(c_str))
#         raise ValueError(f"Invalid start/end value: {arg}. Use 'green', 'red', or 'r,c'.")

#     start_rc = parse_point(start_arg, default_start)
#     end_rc = parse_point(end_arg, default_end)

#     return start_rc, end_rc, (int(green_px[0]), int(green_px[1])), (int(red_px[0]), int(red_px[1]))


def parse_start_end(
    grid, data, start_arg, end_arg
):
    circles = data.get("circles", [])
    green = next((c for c in circles if c.get("color") == "green"), None)
    red   = next((c for c in circles if c.get("color") == "red"),   None)
    if green is None or red is None:
        raise ValueError("JSON must include green and red circles.")

    green_px = (float(green["center"][0]), float(green["center"][1]))
    red_px   = (float(red["center"][0]),   float(red["center"][1]))

    # Defaults: cells nearest to the green/red dots
    default_start = nearest_cell_by_pixel(grid, *green_px)
    default_end   = nearest_cell_by_pixel(grid, *red_px)

    def parse_point(arg, default_rc):
        if not arg:
            return default_rc
        a = arg.strip().lower()
        if a == "green": return default_start
        if a == "red":   return default_end
        if "," in a:
            r_str, c_str = a.split(",", 1)
            return (int(r_str), int(c_str))
        raise ValueError(f"Invalid start/end value: {arg}. Use 'green', 'red', or 'r,c'.")

    start_rc = parse_point(start_arg, default_start)
    end_rc   = parse_point(end_arg,   default_end)

    # Choose which circle pixel to use as the start/end anchor:
    # 1) If user explicitly said 'green' or 'red', honor that.
    # 2) If user gave coordinates, anchor to whichever circle is closer to that cell center.
    def choose_anchor(which, rc):
        if which and which.strip().lower() in ("green", "red"):
            return green_px if which.strip().lower() == "green" else red_px
        cx, cy = grid[rc[0]][rc[1]].center_px
        dg = (cx - green_px[0])**2 + (cy - green_px[1])**2
        dr = (cx - red_px[0])**2   + (cy - red_px[1])**2
        return green_px if dg <= dr else red_px

    start_dot = choose_anchor(start_arg, start_rc)
    end_dot   = choose_anchor(end_arg,   end_rc)

    # Return anchors in the same order as chosen start/end
    return start_rc, end_rc, (int(start_dot[0]), int(start_dot[1])), (int(end_dot[0]), int(end_dot[1]))

def bfs_path(
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    BFS shortest path, 4-connected.
    Rule:
      - You can stand on start even if value=0.
      - You can stand on end even if value=0.
      - All intermediate steps must be on cells with value=1.
    """
    rows, cols = len(grid), len(grid[0])
    sr, sc = start_rc
    tr, tc = end_rc

    def neighbors(r: int, c: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                yield rr, cc

    # We will allow entering end no matter its value,
    # and we allow starting from start no matter its value.
    q = deque()
    q.append((sr, sc))
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen = set([(sr, sc)])

    while q:
        r, c = q.popleft()
        if (r, c) == (tr, tc):
            # reconstruct
            path = [(r, c)]
            while (r, c) != (sr, sc):
                r, c = prev[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        for rr, cc in neighbors(r, c):
            if (rr, cc) in seen:
                continue

            # allow stepping onto end regardless of value
            if (rr, cc) == (tr, tc):
                prev[(rr, cc)] = (r, c)
                seen.add((rr, cc))
                q.append((rr, cc))
                continue

            # otherwise, must be a 1-cell to traverse
            if grid[rr][cc].value != 1:
                continue

            prev[(rr, cc)] = (r, c)
            seen.add((rr, cc))
            q.append((rr, cc))

    return None


def draw_path_on_image(
    image_path: str,
    out_image_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
    line_width: int = 5,
) -> None:
    """
    Draws the polyline from the green dot to the first cell center,
    through the path cells, and finally to the red dot.
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Build pixel polyline
    poly: List[Tuple[int, int]] = []

    # Start at the green circle center
    poly.append(start_circle_px)

    # Then the centers of each cell on the path (in order)
    for (r, c) in cell_path:
        poly.append(grid[r][c].center_px)

    # Finish at the red circle center
    poly.append(end_circle_px)

    # Draw the path
    draw.line(poly, width=line_width, fill=(255, 0, 0, 255))  # solid red path
    # Mark endpoints for clarity
    r_rad = max(6, line_width * 2)
    g_rad = max(6, line_width * 2)
    gx, gy = start_circle_px
    rx, ry = end_circle_px
    draw.ellipse((gx - g_rad, gy - g_rad, gx + g_rad, gy + g_rad), outline=(0, 255, 0, 255), width=3)
    draw.ellipse((rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad), outline=(255, 0, 0, 255), width=3)

    img.save(out_image_path)


def write_path_json(
    out_json_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
) -> None:
    """
    Writes a JSON containing:
      - start/end pixel coordinates (green/red dot centers)
      - start/end cells
      - path as cells (row,col,value,center_px)
      - path as pixels (list of [x,y] including the dots at beginning/end)
      - length (number of moves)
    """
    # path cells expanded
    path_cells_expanded = [
        {
            "row": r,
            "col": c,
            "value": grid[r][c].value,
            "center_px": [grid[r][c].center_px[0], grid[r][c].center_px[1]],
        }
        for (r, c) in cell_path
    ]

    # pixel polyline that matches drawn line (start dot -> all cell centers -> end dot)
    pixel_polyline: List[List[int]] = []
    pixel_polyline.append([start_circle_px[0], start_circle_px[1]])
    for (r, c) in cell_path:
        x, y = grid[r][c].center_px
        pixel_polyline.append([x, y])
    pixel_polyline.append([end_circle_px[0], end_circle_px[1]])

    out = {
        "start_cell": {"row": start_rc[0], "col": start_rc[1]},
        "end_cell": {"row": end_rc[0], "col": end_rc[1]},
        "start_circle_px": [start_circle_px[0], start_circle_px[1]],
        "end_circle_px": [end_circle_px[0], end_circle_px[1]],
        "path_cells": path_cells_expanded,
        "path_pixels": pixel_polyline,
        "moves": max(0, len(cell_path) - 1),  # moves between cells (excludes dot-to-cell hops)
        "notes": "Path respects rule: 1-only traversal; start/end cells may be 0.",
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Solve a grid maze from JSON and draw the path.")
    ap.add_argument("--json_path", default= "part_2_maze_solution/result.json" , help="Input maze JSON path.")
    ap.add_argument("--start", help="Start: 'green' (default), 'red', or 'row,col'", default='red')
    ap.add_argument("--end", help="End: 'red' (default), 'green', or 'row,col'", default='green')
    ap.add_argument("--out-image", help="Output image with path (PNG).",
                    default="part_2_maze_solution/solution_overlay.png")
    ap.add_argument("--out-json", help="Output JSON with path pixel points.",
                    default="part_2_maze_solution/solution_path_points.json")
    ap.add_argument("--line-width", type=int, default=5, help="Path line width on image.")
    args = ap.parse_args()

    data = load_maze(args.json_path)
    rows = int(data["grid_rows"])
    cols = int(data["grid_cols"])
    grid = cells_to_grid(data["cells"], rows, cols)

    start_rc, end_rc, start_circle_px, end_circle_px = parse_start_end(
        grid, data, args.start, args.end
    )

    path = bfs_path(grid, start_rc, end_rc)
    if path is None:
        raise SystemExit("No feasible path found under the given rules.")

    # Draw overlay on the image
    image_path = data["input"]
    if not Path(image_path).exists():
        raise SystemExit(f"Input image not found at: {image_path}")

    draw_path_on_image(
        image_path=image_path,
        out_image_path=args.out_image,
        cell_path=path,
        grid=grid,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
        line_width=args.line_width,
    )

    # Write path JSON
    write_path_json(
        out_json_path=args.out_json,
        cell_path=path,
        grid=grid,
        start_rc=start_rc,
        end_rc=end_rc,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
    )

    print(f"Done.\n - Path image: {args.out_image}\n - Path JSON: {args.out_json}")


if __name__ == "__main__":
    main()

# ______________________________


# #!/usr/bin/env python3
# """
# solve_maze.py (deterministic A* version)

# - Traversable cells have value==1.
# - Start/End may be on value==0 if --snap-to-free 0 (default), else they will be snapped
#   to the nearest reachable value==1 cell using Manhattan distance.
# - Movement is 4-connected.
# - Uses A* with deterministic neighbor ordering biased toward the goal,
#   plus a small straightness tie-breaker to avoid zig-zag.

# Usage:
#   python solve_maze.py maze.json                         # default: start=green, end=red
#   python solve_maze.py maze.json --start red --end green
#   python solve_maze.py maze.json --start 9,1 --end 1,9
#   python solve_maze.py maze.json --snap-to-free 1 --smooth 1
#   python solve_maze.py maze.json --out-image path.png --out-json path_points.json
# """

# import json
# import argparse
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Tuple, Dict, Optional, Any
# from heapq import heappush, heappop

# from PIL import Image, ImageDraw


# @dataclass(frozen=True)
# class Cell:
#     row: int
#     col: int
#     value: int
#     center_px: Tuple[int, int]


# def load_maze(json_path: str) -> Dict[str, Any]:
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def cells_to_grid(cells_json: List[Dict[str, Any]], rows: int, cols: int) -> List[List[Cell]]:
#     grid: List[List[Cell]] = [[None for _ in range(cols)] for _ in range(rows)]
#     for c in cells_json:
#         cell = Cell(
#             row=int(c["row"]),
#             col=int(c["col"]),
#             value=int(c["value"]),
#             center_px=(int(c["center_px"][0]), int(c["center_px"][1])),
#         )
#         grid[cell.row][cell.col] = cell

#     for r in range(rows):
#         for q in range(cols):
#             if grid[r][q] is None:
#                 raise ValueError(f"Missing cell at ({r},{q}) in JSON.")
#     return grid


# def nearest_cell_by_pixel(grid: List[List[Cell]], x: float, y: float) -> Tuple[int, int]:
#     best = None
#     best_d2 = 1e18
#     for r in range(len(grid)):
#         for c in range(len(grid[0])):
#             cx, cy = grid[r][c].center_px
#             d2 = (cx - x) ** 2 + (cy - y) ** 2
#             if d2 < best_d2:
#                 best_d2 = d2
#                 best = (r, c)
#     return best


# def parse_start_end(
#     grid: List[List[Cell]],
#     data: Dict[str, Any],
#     start_arg: Optional[str],
#     end_arg: Optional[str],
# ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
#     circles = data.get("circles", [])
#     green = next((c for c in circles if c.get("color") == "green"), None)
#     red   = next((c for c in circles if c.get("color") == "red"),   None)
#     if green is None or red is None:
#         raise ValueError("JSON must include green and red circles.")

#     gp = (float(green["center"][0]), float(green["center"][1]))
#     rp = (float(red["center"][0]),   float(red["center"][1]))

#     default_start = nearest_cell_by_pixel(grid, *gp)
#     default_end   = nearest_cell_by_pixel(grid, *rp)

#     def parse_point(arg: Optional[str], default_rc: Tuple[int, int]) -> Tuple[int, int]:
#         if not arg:
#             return default_rc
#         a = arg.strip().lower()
#         if a == "green":
#             return default_start
#         if a == "red":
#             return default_end
#         if "," in a:
#             r_str, c_str = a.split(",", 1)
#             return (int(r_str), int(c_str))
#         raise ValueError(f"Invalid start/end value: {arg}. Use 'green', 'red', or 'r,c'.")

#     start_rc = parse_point(start_arg, default_start)
#     end_rc   = parse_point(end_arg,   default_end)

#     return start_rc, end_rc, (int(gp[0]), int(gp[1])), (int(rp[0]), int(rp[1]))


# # ---------- helpers for A* ----------

# def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
#     return abs(a[0]-b[0]) + abs(a[1]-b[1])


# def in_bounds(grid, r, c) -> bool:
#     return 0 <= r < len(grid) and 0 <= c < len(grid[0])


# def snap_to_nearest_free(grid: List[List[Cell]], rc: Tuple[int,int]) -> Tuple[int,int]:
#     """If rc is on value==1, return it. Otherwise search outward in Manhattan shells
#        for the nearest value==1 cell. Deterministic scan order."""
#     if grid[rc[0]][rc[1]].value == 1:
#         return rc
#     rows, cols = len(grid), len(grid[0])
#     r0, c0 = rc
#     best = None
#     best_d = None
#     max_radius = rows + cols
#     for d in range(1, max_radius+1):
#         # scan a diamond ring at distance d in deterministic order
#         for dr in range(-d, d+1):
#             dc = d - abs(dr)
#             for sc in (-1, 1) if dc != 0 else (1,):
#                 rr = r0 + dr
#                 cc = c0 + sc*dc
#                 if in_bounds(grid, rr, cc) and grid[rr][cc].value == 1:
#                     if best is None or d < best_d or (d == best_d and (rr,cc) < best):
#                         best = (rr, cc)
#                         best_d = d
#         if best is not None:
#             return best
#     # fallback to original if no free found
#     return rc


# def astar_single_path(
#     grid: List[List[Cell]],
#     start_rc: Tuple[int, int],
#     end_rc: Tuple[int, int],
#     allow_start_on_zero: bool = True,
#     allow_end_on_zero: bool = True,
# ) -> Optional[List[Tuple[int,int]]]:
#     """
#     A* on 4-neighborhood with deterministic neighbor ordering biased toward end_rc.
#     - Traversal allowed only on value==1, except:
#       - start allowed if allow_start_on_zero
#       - end   allowed if allow_end_on_zero
#     - Tie-breaking favors continuing the same direction to reduce zigzag.
#     """
#     rows, cols = len(grid), len(grid[0])
#     sr, sc = start_rc
#     tr, tc = end_rc

#     # neighbor order biased toward the goal (deterministic):
#     def ordered_neighbors(r: int, c: int) -> List[Tuple[int,int]]:
#         # Four directions with vectors and an order score toward target
#         dirs = [(-1,0), (1,0), (0,-1), (0,1)]  # N,S,W,E
#         scored = []
#         for dr, dc in dirs:
#             rr, cc = r+dr, c+dc
#             if not in_bounds(grid, rr, cc):
#                 continue
#             # lower is better; bias toward Manhattan closeness to target
#             bias = manhattan((rr,cc), (tr,tc))
#             scored.append((bias, dr, dc, rr, cc))
#         # sort by bias, then by fixed direction order to be deterministic
#         scored.sort(key=lambda t: (t[0], t[1], t[2]))
#         return [(rr,cc,dr,dc) for _, dr, dc, rr, cc in scored]

#     # A* structures
#     g = { (sr,sc): 0.0 }
#     parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
#     # store the direction we came from for straightness bonus
#     came_dir: Dict[Tuple[int,int], Tuple[int,int]] = { (sr,sc): (0,0) }

#     heap = []
#     # small tiebreak epsilon that prefers nodes closer to goal, and continuing direction
#     def f_cost(rc: Tuple[int,int], dir_from: Tuple[int,int]) -> float:
#         return g[rc] + manhattan(rc, end_rc) * 1.0 + (0.001 if dir_from==(0,0) else 0.0)

#     heappush(heap, (f_cost((sr,sc),(0,0)), 0.0, (sr,sc)))

#     seen = set()

#     # helper to test if we can stand on rc
#     def passable(rc: Tuple[int,int]) -> bool:
#         r,c = rc
#         if rc == (sr,sc) and allow_start_on_zero:
#             return True
#         if rc == (tr,tc) and allow_end_on_zero:
#             return True
#         return grid[r][c].value == 1

#     if not passable((sr,sc)) and not allow_start_on_zero:
#         return None

#     while heap:
#         _, gcur, cur = heappop(heap)
#         if cur in seen:
#             continue
#         seen.add(cur)

#         if cur == (tr,tc):
#             # reconstruct path
#             path = [cur]
#             while cur in parent:
#                 cur = parent[cur]
#                 path.append(cur)
#             path.reverse()
#             return path

#         r,c = cur
#         prev_dir = came_dir.get(cur, (0,0))

#         for rr,cc,dr,dc in ordered_neighbors(r,c):
#             nxt = (rr,cc)
#             if not passable(nxt):
#                 continue
#             ng = gcur + 1.0

#             # tiny straightness bonus: continuing same direction slightly cheaper
#             if (dr,dc) == prev_dir:
#                 ng -= 1e-6

#             if ng < g.get(nxt, 1e18):
#                 g[nxt] = ng
#                 parent[nxt] = cur
#                 came_dir[nxt] = (dr,dc)
#                 heappush(heap, (ng + manhattan(nxt, end_rc), ng, nxt))

#     return None


# def smooth_cells_polyline(cells: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
#     """Remove collinear middle points from a grid polyline for cosmetic straight segments."""
#     if len(cells) <= 2:
#         return cells
#     out = [cells[0]]
#     for i in range(1, len(cells)-1):
#         r0,c0 = out[-1]
#         r1,c1 = cells[i]
#         r2,c2 = cells[i+1]
#         # collinear in grid if direction r0->r1 equals r1->r2
#         if (r1-r0, c1-c0) == (r2-r1, c2-c1):
#             continue
#         out.append((r1,c1))
#     out.append(cells[-1])
#     return out


# def draw_path_on_image(
#     image_path: str,
#     out_image_path: str,
#     cell_path: List[Tuple[int, int]],
#     grid: List[List[Cell]],
#     start_circle_px: Tuple[int, int],
#     end_circle_px: Tuple[int, int],
#     line_width: int = 5,
# ) -> None:
#     img = Image.open(image_path).convert("RGBA")
#     draw = ImageDraw.Draw(img)

#     poly: List[Tuple[int,int]] = []
#     poly.append(start_circle_px)
#     for (r,c) in cell_path:
#         poly.append(grid[r][c].center_px)
#     poly.append(end_circle_px)

#     draw.line(poly, width=line_width, fill=(255,0,0,255))
#     r_rad = max(6, line_width * 2)
#     g_rad = max(6, line_width * 2)
#     gx, gy = start_circle_px
#     rx, ry = end_circle_px
#     draw.ellipse((gx - g_rad, gy - g_rad, gx + g_rad, gy + g_rad), outline=(0,255,0,255), width=3)
#     draw.ellipse((rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad), outline=(255,0,0,255), width=3)

#     img.save(out_image_path)


# def write_path_json(
#     out_json_path: str,
#     cell_path: List[Tuple[int, int]],
#     grid: List[List[Cell]],
#     start_rc: Tuple[int, int],
#     end_rc: Tuple[int, int],
#     start_circle_px: Tuple[int, int],
#     end_circle_px: Tuple[int, int],
# ) -> None:
#     path_cells_expanded = [
#         {
#             "row": r,
#             "col": c,
#             "value": grid[r][c].value,
#             "center_px": [grid[r][c].center_px[0], grid[r][c].center_px[1]],
#         }
#         for (r, c) in cell_path
#     ]
#     pixel_poly: List[List[int]] = []
#     pixel_poly.append([start_circle_px[0], start_circle_px[1]])
#     for (r, c) in cell_path:
#         x, y = grid[r][c].center_px
#         pixel_poly.append([x, y])
#     pixel_poly.append([end_circle_px[0], end_circle_px[1]])

#     out = {
#         "start_cell": {"row": start_rc[0], "col": start_rc[1]},
#         "end_cell": {"row": end_rc[0], "col": end_rc[1]},
#         "start_circle_px": [start_circle_px[0], start_circle_px[1]],
#         "end_circle_px": [end_circle_px[0], end_circle_px[1]],
#         "path_cells": path_cells_expanded,
#         "path_pixels": pixel_poly,
#         "moves": max(0, len(cell_path) - 1),
#         "notes": "A* path with deterministic tie-breaking; start/end may be 0 if snap-to-free is 0.",
#     }
#     with open(out_json_path, "w", encoding="utf-8") as f:
#         json.dump(out, f, indent=2)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--json_path", default="part_2_maze_solution/result.json", help="Input maze JSON path.")
#     ap.add_argument("--start", default="red", help="Start: 'green' (default), 'red', or 'row,col'")
#     ap.add_argument("--end",   default="green",   help="End:   'red' (default), 'green', or 'row,col'")
#     ap.add_argument("--snap-to-free", type=int, default=0, help="Snap start/end to nearest value==1 cell (0/1).")
#     ap.add_argument("--smooth", type=int, default=0, help="Simplify path by removing collinear cells (0/1).")
#     ap.add_argument("--out-image", default="part_2_maze_solution/solution_overlay.png")
#     ap.add_argument("--out-json",  default="part_2_maze_solution/solution_path_points.json")
#     ap.add_argument("--line-width", type=int, default=5)
#     args = ap.parse_args()

#     data = load_maze(args.json_path)
#     rows = int(data["grid_rows"])
#     cols = int(data["grid_cols"])
#     grid = cells_to_grid(data["cells"], rows, cols)

#     start_rc, end_rc, start_circle_px, end_circle_px = parse_start_end(grid, data, args.start, args.end)

#     if args.snap_to_free:
#         start_rc = snap_to_nearest_free(grid, start_rc)
#         end_rc   = snap_to_nearest_free(grid, end_rc)

#     path = astar_single_path(
#         grid,
#         start_rc,
#         end_rc,
#         allow_start_on_zero=(args.snap_to_free == 0),
#         allow_end_on_zero=(args.snap_to_free == 0),
#     )
#     if path is None:
#         raise SystemExit("No feasible path found.")

#     if args.smooth:
#         path = smooth_cells_polyline(path)

#     image_path = data["input"]
#     if not Path(image_path).exists():
#         raise SystemExit(f"Input image not found at: {image_path}")

#     draw_path_on_image(
#         image_path=image_path,
#         out_image_path=args.out_image,
#         cell_path=path,
#         grid=grid,
#         start_circle_px=start_circle_px,
#         end_circle_px=end_circle_px,
#         line_width=args.line_width,
#     )

#     write_path_json(
#         out_json_path=args.out_json,
#         cell_path=path,
#         grid=grid,
#         start_rc=start_rc,
#         end_rc=end_rc,
#         start_circle_px=start_circle_px,
#         end_circle_px=end_circle_px,
#     )

#     print(f"Done.\n - Path image: {args.out_image}\n - Path JSON: {args.out_json}")


# if __name__ == "__main__":
#     main()
