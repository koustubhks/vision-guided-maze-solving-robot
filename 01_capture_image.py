#!/usr/bin/env python3
"""
camera_maze_corners_blackbg_expand_json.py

Camera-only version:
 - Opens a webcam preview.
 - Press SPACE (or 'c') to capture a frame and detect the largest quadrilateral
   on a black background (white maze).
 - Expands/shrinks corners by --expand px (in ORIGINAL image pixels).
 - Saves:
     - Raw frame: camera_capture.png (or --save-raw)
     - Overlay:   maze_corners_overlay.png (or --out)
     - JSON:      camera_capture_corners.json (or --json-out)

Keys:
  SPACE / 'c' : capture & detect
  'r'         : retake (go back to live preview)
  's'         : save again (overlay/json are already saved)
  'q' / ESC   : quit

Usage:
  python camera_maze_corners_blackbg_expand_json.py
  python camera_maze_corners_blackbg_expand_json.py --cam-index 1 --width 1200 --expand 10
"""

import argparse
import sys
import os
import json
from typing import Optional, Tuple
import numpy as np
import cv2
from datetime import datetime


# ---------------- Core helpers ----------------
def make_dir_if_needed(path: str):
    dirn = os.path.dirname(path)
    if dirn and not os.path.exists(dirn):
        os.makedirs(dirn, exist_ok=True)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def binarize_white_foreground(gray: np.ndarray) -> np.ndarray:
    """Threshold so white maze remains white (255) and black background is 0."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.9:  # if almost everything turned white, invert
        th = 255 - th
    return th

def morph_cleanup(bin_img: np.ndarray) -> np.ndarray:
    """Light open + close to reduce specks and bridge small gaps."""
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=1)
    return closed

def approx_to_quads(cnt: np.ndarray, max_iter: int = 25) -> Optional[np.ndarray]:
    """Try to approximate the contour to exactly 4 points by increasing epsilon."""
    peri = cv2.arcLength(cnt, True)
    for frac in np.linspace(0.01, 0.06, max_iter):
        approx = cv2.approxPolyDP(cnt, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None

def find_largest_quad(bin_img: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Find the largest-area quadrilateral among contours.
    Fallback to minAreaRect on the largest contour if needed.
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found. Check thresholding/lighting.")

    h, w = bin_img.shape[:2]
    min_area = max(1000.0, 0.001 * (h * w))  # at least 0.1% of image or 1000 px
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        candidates = [max(contours, key=cv2.contourArea)]

    best_quad, best_area = None, -1.0
    for c in candidates:
        quad = approx_to_quads(c)
        if quad is not None:
            area = cv2.contourArea(quad.astype(np.int32))
            if area > best_area:
                best_area = area
                best_quad = quad

    if best_quad is not None:
        return order_corners(best_quad), "largest_quadrilateral"

    # Fallback: oriented bounding box of the largest candidate
    largest = max(candidates, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(box), "minAreaRect_fallback"

def expand_corners(corners: np.ndarray, expand_px: float) -> np.ndarray:
    """
    Move each corner outward from the polygon centroid by expand_px pixels.
    Positive = outward, negative = inward.
    """
    if abs(expand_px) < 1e-9:
        return corners.astype(np.float32)

    center = corners.mean(axis=0)
    expanded = []
    for p in corners:
        v = p - center
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            expanded.append(p)
        else:
            scale = (norm + expand_px) / norm
            expanded.append(center + v * scale)
    return order_corners(np.array(expanded, dtype=np.float32))

def draw_overlay(base_bgr: np.ndarray, corners: np.ndarray, color_poly=(255,0,0)) -> np.ndarray:
    overlay = base_bgr.copy()
    c = corners.astype(int)
    labels = ["TL", "TR", "BR", "BL"]
    cv2.polylines(overlay, [c.reshape(-1,1,2)], True, color_poly, 3)
    for i, p in enumerate(c):
        cv2.circle(overlay, tuple(p), 8, (0,255,0), -1)
        cv2.putText(overlay, labels[i], tuple(p + np.array([10,-10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
    return overlay

def detect_corners_blackbg(img_bgr: np.ndarray, resize_width: Optional[int]) -> Tuple[np.ndarray, str, float]:
    """
    Detect corners at processing resolution; caller is responsible for upscaling to original.
    Returns (corners_at_processing_scale, method, scale).
    """
    if resize_width is not None and resize_width > 0 and img_bgr.shape[1] > resize_width:
        scale = resize_width / img_bgr.shape[1]
        img_proc = cv2.resize(img_bgr, (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale)), cv2.INTER_AREA)
    else:
        scale = 1.0
        img_proc = img_bgr

    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    bin_img = binarize_white_foreground(gray)
    bin_img = morph_cleanup(bin_img)

    corners, method = find_largest_quad(bin_img)
    return corners, method, scale


# ---------------- Camera utilities ----------------

def open_camera(cam_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cam_index}.")
    return cap

def camera_loop_and_capture(cam_index: int, window_name: str = "Camera - SPACE/c capture, q to quit") -> Optional[np.ndarray]:
    cap = open_camera(cam_index)
    last_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read from camera.")
                break
            last_frame = frame
            disp = frame.copy()
            h, w = disp.shape[:2]
            cv2.putText(disp, "SPACE/c: capture  |  q/ESC: quit", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):     # ESC or q
                last_frame = None
                break
            if key in (32, ord('c')):     # SPACE or c
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
    return last_frame


# ---------------- Pipeline & saving ----------------

def run_detection_and_save(frame: np.ndarray, args):
    # Detect (possibly at reduced width), then map corners back to ORIGINAL pixels
    corners_proc, method, scale = detect_corners_blackbg(frame, None if args.width == 0 else args.width)
    corners = corners_proc / scale if scale != 1.0 else corners_proc.copy()
    corners = expand_corners(corners, args.expand)

    # Draw overlay
    overlay = draw_overlay(frame, corners)

    # Decide filenames
    raw_path = args.save_raw or "camera_capture.png"
    out_overlay_path = args.out
    json_path = args.json_out or "camera_capture_corners.json"

    # Save raw frame (so JSON 'input' references a real file)
    ok = cv2.imwrite(raw_path, frame)
    if ok:
        print(f"Raw frame saved to: {raw_path}")
    else:
        print("Warning: failed to save raw frame.")

    # Save overlay
    ok = cv2.imwrite(out_overlay_path, overlay)
    if not ok:
        print("Warning: failed to save overlay image.")
    else:
        print(f"Overlay saved to: {out_overlay_path}")

    # Save corners JSON
    corners_list = corners.tolist()
    data = {
        "input": os.path.basename(raw_path),
        "method": method,
        "width_param": args.width,
        "expand_param": args.expand,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "corners": {
            "TL": corners_list[0],
            "TR": corners_list[1],
            "BR": corners_list[2],
            "BL": corners_list[3]
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Corners JSON saved to: {json_path}")

    # Show result and allow quick re-save/retake
    show = overlay.copy()
    h, w = show.shape[:2]
    cv2.putText(show, "Press 's' to save again, 'r' to retake, 'q' to quit",
                (10, max(24, h - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Detected corners (overlay)", show)
    return raw_path, out_overlay_path, json_path


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Camera-only maze corners detector.")
    ap.add_argument("--cam-index", type=int, default=1, help="Camera index to open (default: 0).")
    ap.add_argument("--out", default="part_2_maze_solution/maze_corners_overlay.png", help="Path to save overlay visualization")
    ap.add_argument("--width", type=int, default=1200, help="Processing resize width (0 to disable)")
    ap.add_argument("--expand", type=float, default=10.0, help="Outward expansion in pixels (negative shrinks inward)")
    ap.add_argument("--json-out", type=str, default="part_2_maze_solution/camera_capture_corners.json", help="Corners JSON output path (default: camera_capture_corners.json)")
    ap.add_argument("--save-raw", type=str, default="part_2_maze_solution/camera_capture.png", help="Optional path to save the raw captured frame (default: camera_capture.png)")
    args = ap.parse_args()

    # Live preview -> capture
    frame = camera_loop_and_capture(args.cam_index)
    if frame is None:
        print("No frame captured. Exiting.")
        sys.exit(0)

    # Detect & save once
    raw_path, overlay_path, json_path = run_detection_and_save(frame, args)

    # Post-capture loop (retake/resave/quit)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            # Re-save overlay/json from the same frame (already saved)
            print("Saved (overlay & JSON already written).")
        elif key == ord('r'):
            cv2.destroyAllWindows()
            frame = camera_loop_and_capture(args.cam_index)
            if frame is None:
                print("No frame captured. Exiting.")
                break
            raw_path, overlay_path, json_path = run_detection_and_save(frame, args)
        else:
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
