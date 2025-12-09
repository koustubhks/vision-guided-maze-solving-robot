#!/usr/bin/env python3
"""
unwarp_and_overlay_path.py

Takes path points defined in the warped image coordinates and maps them
back (inverse perspective) to the ORIGINAL image, then overlays the polyline.

Inputs:
  1) Corners JSON (same one used by maze_warp_from_json.py)
  2) Path JSON (produced by the solver; supports 'path_pixels' or 'path_cells')

What it does:
  - Reads corners (TL,TR,BR,BL) and the original image path from the corners JSON.
  - Determines the warped output size (W,H) from --warped-image (recommended) or default name.
  - Computes the inverse perspective transform and applies it to the path points.
  - Draws the path on the original image and saves it.
  - Writes a new JSON with unwarped path pixel points (including start & end).

Examples:
  python unwarp_and_overlay_path.py corners.json solution_path_points.json \
    --warped-image maze_warp.png \
    --out-image original_with_path.png \
    --out-json solution_path_points_unwarped.json

  # If your solver JSON only has path_cells (no path_pixels):
  python unwarp_and_overlay_path.py corners.json solution_path_points.json --from cells

Notes:
  - We treat the path as a polyline:
      [start_circle_px] + cell centers + [end_circle_px]    (when available)
  - If 'path_pixels' is present we prefer it (already includes start/end); you can force
    using 'path_cells' via --from cells.
"""

import argparse
import json
import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ensure TL, TR, BR, BL order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def read_corners_json(path: str) -> Tuple[np.ndarray, str]:
    """
    Returns: (corners TL,TR,BR,BL as float32 4x2), original_image_path
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "corners" in data:
            tl = data["corners"]["TL"]
            tr = data["corners"]["TR"]
            br = data["corners"]["BR"]
            bl = data["corners"]["BL"]
            corners = np.array([tl, tr, br, bl], dtype=np.float32)
        elif all(k in data for k in ("TL", "TR", "BR", "BL")):
            corners = np.array([data["TL"], data["TR"], data["BR"], data["BL"]], dtype=np.float32)
        else:
            raise ValueError("Corners JSON must contain 'corners':{TL,TR,BR,BL} or root-level TL/TR/BR/BL.")

        img_path = data.get("input", "")
        if not isinstance(img_path, str) or not img_path:
            raise ValueError("Corners JSON must include original image path in 'input'.")
    else:
        raise ValueError("Invalid corners JSON root object.")

    return order_corners(corners), img_path


def read_path_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_polyline_pixels(
    path_data: Dict[str, Any],
    mode: str = "auto"
) -> List[List[float]]:
    """
    Returns a polyline as [[x,y], ...] in WARPED coordinates.
    Priority:
      - if mode=='auto': use 'path_pixels' if present; otherwise build from 'path_cells' (center_px)
      - if mode=='pixels': use 'path_pixels'
      - if mode=='cells': build [start_circle_px] + each cell.center_px + [end_circle_px]
    Ensures start & end are included when available.
    """
    def has_pixels():
        return isinstance(path_data.get("path_pixels"), list) and len(path_data["path_pixels"]) > 0

    def has_cells():
        return isinstance(path_data.get("path_cells"), list) and len(path_data["path_cells"]) > 0

    if mode == "pixels" or (mode == "auto" and has_pixels()):
        # Already includes start/end (as produced by the solver script)
        return [[float(x), float(y)] for x, y in path_data["path_pixels"]]

    if mode == "cells" or (mode == "auto" and not has_pixels() and has_cells()):
        poly: List[List[float]] = []
        # Start dot if present
        if "start_circle_px" in path_data:
            sx, sy = path_data["start_circle_px"]
            poly.append([float(sx), float(sy)])
        # Cell centers in order
        for c in path_data["path_cells"]:
            cx, cy = c["center_px"]
            poly.append([float(cx), float(cy)])
        # End dot if present
        if "end_circle_px" in path_data:
            ex, ey = path_data["end_circle_px"]
            poly.append([float(ex), float(ey)])

        if not poly:
            raise ValueError("No points found in 'path_cells' and no start/end circle pixels provided.")
        return poly

    raise ValueError("Path JSON must contain either 'path_pixels' or 'path_cells'.")


def main():
    ap = argparse.ArgumentParser(description="Unwarp path points and overlay them on the original image.")
    ap.add_argument("--corners_json", default= "part_2_maze_solution/camera_capture_corners.json",help="Corners JSON used for warping.")
    ap.add_argument("--path_json", default= "part_2_maze_solution/solution_path_points.json", help="Path JSON produced by solver.")
    ap.add_argument("--warped-image", default= "part_2_maze_solution/maze_warp.png", help="Path to the warped image used when solving (to read W,H). If omitted, tries 'maze_warp.png'.")
    ap.add_argument("--from", dest="from_mode", default="auto", choices=["auto", "pixels", "cells"],
                    help="Whether to read 'path_pixels' or rebuild from 'path_cells'. Default: auto.")
    ap.add_argument("--out-image", default="part_2_maze_solution/original_with_path.png", help="Output overlay on the ORIGINAL image.")
    ap.add_argument("--out-json", default="part_2_maze_solution/solution_path_points_unwarped.json", help="Output JSON with unwarped points.")
    ap.add_argument("--line-width", type=int, default=5, help="Overlay path width.")
    args = ap.parse_args()

    # --- Load corners + original image path ---
    corners, orig_img_path = read_corners_json(args.corners_json)

    # Resolve original image path relative to the corners.json directory if needed
    if not os.path.isabs(orig_img_path):
        cj_dir = os.path.dirname(os.path.abspath(args.corners_json))
        orig_img_path = os.path.join(cj_dir, orig_img_path)

    orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise SystemExit(f"Could not read original image: {orig_img_path}")

    # --- Determine the warped image size (W,H) that was used for solving ---
    warped_image_path = args.warped_image or "maze_warp.png"
    warped_img = cv2.imread(warped_image_path, cv2.IMREAD_COLOR)
    if warped_img is None:
        raise SystemExit(
            f"Could not read warped image '{warped_image_path}'. "
            f"Pass --warped-image to point at the warped PNG used by the solver."
        )
    H, W = warped_img.shape[:2]

    # --- Build inverse perspective transform: rect(W,H) -> original quad(corners) ---
    src_rect = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)   # warped space
    dst_quad = corners.astype(np.float32)                                                     # original image space
    M_inv = cv2.getPerspectiveTransform(src_rect, dst_quad)  # maps warped -> original

    # --- Load path JSON and collect a warped-space polyline ---
    path_data = read_path_json(args.path_json)
    poly_warped = np.array(collect_polyline_pixels(path_data, mode=args.from_mode), dtype=np.float32).reshape(-1, 1, 2)

    # --- Transform to original-image coordinates ---
    poly_orig = cv2.perspectiveTransform(poly_warped, M_inv).reshape(-1, 2)

    # --- Draw the unwarped path on the original image ---
    img_overlay = orig_img.copy()
    # Convert to list of ints for drawing and for JSON
    poly_orig_int = [(int(round(x)), int(round(y))) for x, y in poly_orig]

    # Draw the polyline
    for i in range(len(poly_orig_int) - 1):
        cv2.line(img_overlay, poly_orig_int[i], poly_orig_int[i + 1], color=(0, 0, 255), thickness=args.line_width)  # red BGR
    # Mark endpoints for clarity
    if len(poly_orig_int) >= 1:
        cv2.circle(img_overlay, poly_orig_int[0], max(6, args.line_width * 2), (0, 255, 0), thickness=3)  # start (green)
    if len(poly_orig_int) >= 2:
        cv2.circle(img_overlay, poly_orig_int[-1], max(6, args.line_width * 2), (0, 0, 255), thickness=3)  # end (red)

    ok = cv2.imwrite(args.out_image, img_overlay)
    if not ok:
        print("Warning: failed to save overlay image.", flush=True)

    # --- Save unwarped points JSON ---
    out = {
        "source_corners_json": os.path.abspath(args.corners_json),
        "source_path_json": os.path.abspath(args.path_json),
        "original_image": os.path.abspath(orig_img_path),
        "warped_image": os.path.abspath(warped_image_path),
        "unwarped_path_pixels": [[int(x), int(y)] for (x, y) in poly_orig_int],
        "notes": "Polyline points are in ORIGINAL image pixel coordinates. Start and end included if present in input."
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Done.")
    print(f" - Overlay image (original space): {args.out_image}")
    print(f" - Unwarped path JSON: {args.out_json}")


if __name__ == "__main__":
    main()
