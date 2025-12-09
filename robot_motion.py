from pydobot.dobot import MODE_PTP
import pydobot
import time

# # List likely serial devices (won't error if missing)
# ls -l /dev/ttyACM* /dev/ttyUSB* /dev/serial/by-id 2>/dev/null || true

device = pydobot.Dobot(port="COM22")

def move_to_home(device):
    #_______________________________________________________
    # Moving the robot to a home and get home position
    # Code ##################################################
    print("Homing the robot...")
    device.home()  # Home the robot to the origin position
    time.sleep(2)
    (pose, joint) = device.get_pose()  # Get the current position and joint angles
    print(f"pose: {pose}, j: {joint}")
    # Code End ##############################################
    # ______________________________________________________

def move_to_specific_position(device,x,y,z,r = 0.0):
    #_______________________________________________________
    # Moving the robot to a specific position
    # Code ##################################################
    device.speed(50, 50)
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x = x, y = y , z = z , r= r)  # Move to position (x=250, y=0, z=50) with r=0
    time.sleep(2)
    # Code End ##############################################
    # ______________________________________________________

def get_current_pose(device):
    #_______________________________________________________
    # Getting the current pose
    # Code ##################################################
    time.sleep(1)  
    print("current pose")
    (pose, joint) = device.get_pose()  # Get the current position and joint angles
    print(f"pose: {pose}")
    return pose, joint
    # Code End ##############################################
    # ______________________________________________________

# Main execution

move_to_home(device)

# move_to_specific_position(x=327, y=62, z=-45)

# get_current_pose(device)
# move_to_specific_position(x=218, y=-41, z=-45)
# move_to_specific_position(x=245, y=-53, z=-45)


current_pose, current_joint = get_current_pose(device)
# print(f"Final pose: {current_pose}")

# Close the device connection
# device.close()