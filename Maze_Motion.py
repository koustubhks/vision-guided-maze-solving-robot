# import numpy as np
# import cv2  
# from pydobot.dobot import MODE_PTP
# import time
# import pydobot
# from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
# from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

# M = np.array([
#      [6.00650232e-03, -4.84214952e-01, 4.24653329e+02],
#      [-4.09079919e-01, 3.74996755e-03, 1.11349575e+02]
# ], dtype=np.float64)

# H = np.array([
#     [-2.44594058e-02, -4.75669460e-01,  3.67247188e+02],
#     [-4.34041615e-01,  5.08065338e-03,  1.20901686e+02],
#     [-5.98330506e-05 ,-7.62411614e-05 , 1.00000000e+00]]
#     , dtype=np.float64)

# def move_robot_point(device,M,u,v):
#     Xa, Ya = apply_affine(M, u, v) # Using Affine
#     # Xa, Ya = apply_homography(H, u, v) # Using Homography
#     print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
#     move_to_specific_position(device, x=Xa, y=Ya, z=-45)
#     time.sleep(1)

# def main():
#     device = pydobot.Dobot(port="COM15")
#     device.speed(50, 50)
#     move_to_home(device)
#     time.sleep(2)

#     # Example pixel coordinates from clicks
#     pixel_coords = [
#     (210, 173),
#     (216, 177),
#     (234, 159),
#     (251, 141),
#     (269, 158),
#     (287, 176),
#     (305, 193),
#     (288, 211),
#     (270, 229),
#     (288, 247),
#     (306, 264),
#     (289, 282),
#     (271, 300),
#     (290, 318),
#     (308, 335),
#     (325, 317),
#     (342, 299),
#     (360, 281),
#     (377, 263),
#     (395, 280),
#     (412, 297),
#     (407, 286)
# ]
    
#     for (u, v) in pixel_coords:
#         move_robot_point(device ,M, u, v) 

#     device.close() 
    
# if __name__ == "__main__":
#     main()




import numpy as np
import cv2  
from pydobot.dobot import MODE_PTP
import time
import pydobot
import json # <-- ADDED: Needed to read the solution file
from camera_utilities import apply_affine, fit_affine, apply_homography, fit_homography
from robot_utilities import move_to_home, move_to_specific_position, get_current_pose

# Define Transformation Matrices (Keep existing matrices)
M = np.array([
     [6.00650232e-03, -4.84214952e-01, 4.24653329e+02],
     [-4.09079919e-01, 3.74996755e-03, 1.11349575e+02]
], dtype=np.float64)

H = np.array([
    [-2.44594058e-02, -4.75669460e-01,  3.67247188e+02],
    [-4.34041615e-01,  5.08065338e-03,  1.20901686e+02],
    [-5.98330506e-05 ,-7.62411614e-05 , 1.00000000e+00]]
    , dtype=np.float64)

def move_robot_point(device,M,u,v):
    Xa, Ya = apply_affine(M, u, v) # Using Affine
    # Xa, Ya = apply_homography(H, u, v) # Using Homography
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-45)
    time.sleep(1)

def main():
    # --- 1. Load the solved path coordinates from the JSON file ---
    PATH_JSON_FILE = "part_2_maze_solution/solution_path_points_unwarped.json"
    
    try:
        with open(PATH_JSON_FILE, 'r') as f:
            data = json.load(f)
        
        # Extract the list of [x, y] coordinates
        pixel_coords_list = data["unwarped_path_pixels"] 
        
        # Convert the list of lists [[x, y], ...] into a list of tuples [(u, v), ...]
        # This structure matches the required format for the loop: [(210, 173), ...]
        pixel_coords = [tuple(p) for p in pixel_coords_list] 

    except FileNotFoundError:
        print(f"Error: Solution file not found at {PATH_JSON_FILE}. Ensure you ran file 5.")
        return
    except KeyError:
        print("Error: 'unwarped_path_pixels' key missing in the JSON file.")
        return
    # -----------------------------------------------------------
    
    device = pydobot.Dobot(port="COM15")
    device.speed(50, 50)
    move_to_home(device)
    time.sleep(2)

    # Robot moves along the path loaded from the unwarped JSON file
    for (u, v) in pixel_coords:
        move_robot_point(device ,M, u, v) 

    device.close() 
    
if __name__ == "__main__":
    main()