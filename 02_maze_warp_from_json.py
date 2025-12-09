#!/usr/bin/env python3
"""
maze_warp_from_json.py
Warp/crop a maze to be axis-aligned by reading corners from a JSON file produced by
maze_corners_blackbg_expand_json.py (or compatible).

JSON format supported:
{
  "input": "capture_0.png",
  "method": "...",
  "width_param": 1200,
  "expand_param": 10.0,
  "timestamp": "...",
  "corners": {
    "TL": [x,y],
    "TR": [x,y],
    "BR": [x,y],
    "BL": [x,y]
  }
}

Usage:
  python maze_warp_from_json.py corners.json --out warped.png
  # Optionally override the image path (defaults to "input" inside the JSON, resolved relative to JSON file)
  python maze_warp_from_json.py corners.json --image /path/to/capture_0.png --out warped.png
  # Optional controls:
  python maze_warp_from_json.py corners.json --pad 16 --target-size 1200x900
"""

import argparse
import json
import os
import sys
from typing import Tuple
import numpy as np
import cv2

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ensure TL, TR, BR, BL order."""
    pts = pts.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def parse_target_size(s: str) -> Tuple[int,int]:
    try:
        w_str, h_str = s.lower().split("x")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        raise ValueError("Invalid --target-size. Use WIDTHxHEIGHT (e.g., 1200x800).")

def infer_size_from_quad(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    W = int(round((width_top + width_bottom) / 2.0))
    H = int(round((height_left + height_right) / 2.0))
    return max(W,1), max(H,1)

def warp_perspective(img: np.ndarray, corners: np.ndarray, out_size: Tuple[int,int], pad: int = 0) -> np.ndarray:
    W, H = out_size
    src = corners.astype(np.float32)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    if pad > 0:
        warped = cv2.copyMakeBorder(warped, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    return warped

def read_corners_from_json(json_path: str) -> Tuple[np.ndarray, str]:
    """
    Returns (corners 4x2 float32 in TL,TR,BR,BL order, image_path_from_json_or_empty)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    img_from_json = ""
    if isinstance(data, dict):
        # Try nested dict under "corners"
        if "corners" in data and isinstance(data["corners"], dict):
            try:
                tl = data["corners"]["TL"]
                tr = data["corners"]["TR"]
                br = data["corners"]["BR"]
                bl = data["corners"]["BL"]
                pts = np.array([tl, tr, br, bl], dtype=np.float32)
            except KeyError as e:
                raise ValueError(f"Missing key in corners JSON: {e}")
        # Or root-level TL/TR/BR/BL
        elif all(k in data for k in ("TL","TR","BR","BL")):
            pts = np.array([data["TL"], data["TR"], data["BR"], data["BL"]], dtype=np.float32)
        # Or root-level list
        elif isinstance(data, list) and len(data) == 4:
            pts = np.array(data, dtype=np.float32)
        else:
            raise ValueError("JSON doesn't contain recognizable corners format.")

        if "input" in data and isinstance(data["input"], str):
            img_from_json = data["input"]

    else:
        raise ValueError("JSON root must be an object or a list of 4 [x,y] pairs.")

    # Ensure correct order
    pts = order_corners(pts)
    return pts, img_from_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="part_2_maze_solution/camera_capture_corners.json", help="Path to corners JSON")
    ap.add_argument("--image", help="Optional override for image path; defaults to 'input' inside JSON, relative to JSON file dir")
    ap.add_argument("--out", default="part_2_maze_solution/maze_warp.png", help="Output warped image path")
    ap.add_argument("--target-size", help="Force output size as WIDTHxHEIGHT (e.g., 1200x800). If omitted, inferred from quad")
    ap.add_argument("--pad", type=int, default=0, help="Uniform pixel padding around the warped output")
    args = ap.parse_args()

    # Read corners and image path from JSON
    try:
        corners, img_from_json = read_corners_from_json(args.json)
    except Exception as e:
        print(f"Error reading JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve image path
    if args.image:
        img_path = args.image
    else:
        if not img_from_json:
            print("No image path provided via --image and none found in JSON 'input' field.", file=sys.stderr)
            sys.exit(2)
        # If JSON includes a relative image name, resolve relative to JSON file directory
        json_dir = os.path.dirname(os.path.abspath(args.json))
        img_path = os.path.join(json_dir, img_from_json)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image: {img_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output size
    if args.target_size:
        try:
            W, H = parse_target_size(args.target_size)
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
    else:
        W, H = infer_size_from_quad(corners)

    warped = warp_perspective(img, corners, (W, H), pad=args.pad)
    ok = cv2.imwrite(args.out, warped)
    if not ok:
        print("Warning: failed to save warped output.", file=sys.stderr)

    print(f"Warp complete from: {img_path}")
    print(f"Size: {W}x{H}px  Pad: {args.pad}px")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
