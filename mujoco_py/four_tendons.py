import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math

xml_path = '../mujoco_models/four_tendons.xml' #xml file (assumes this is in the same folder as this file)
simend = 100 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

data.qvel = [0] * len(data.qvel)

started_recording = False
start_time = 5

received_goal = False
threshold = 0.002

r = 0.025
lc = np.cos(np.pi/4) * r
C_4 = np.array([-1.154-lc,  1.404+lc, 3.22]) # anchor points
C_3 = np.array([+1.154+lc,  1.404+lc, 3.22])
C_2 = np.array([+1.154+lc, -1.404-lc, 3.22])
C_1 = np.array([-1.154-lc, -1.404-lc, 3.22])
box_x, box_y, box_z = 0.05, 0.05, 0.05

def compute_cables_lenghts(Ac_new):
    A_1 = Ac_new + np.array([-box_x*np.cos(np.pi/4),  box_x*np.cos(np.pi/4), box_z])
    A_2 = Ac_new + np.array([ box_x*np.cos(np.pi/4),  box_x*np.cos(np.pi/4), box_z])
    A_3 = Ac_new + np.array([ box_x*np.cos(np.pi/4), -box_x*np.cos(np.pi/4), box_z])
    A_4 = Ac_new + np.array([-box_x*np.cos(np.pi/4), -box_x*np.cos(np.pi/4), box_z])

    beta_1 = np.arctan2(A_1[1] - C_1[1], A_1[0] - C_1[0])
    beta_2 = np.arctan2(A_2[1] - C_2[1], A_2[0] - C_2[0])
    beta_3 = np.arctan2(A_3[1] - C_3[1], A_3[0] - C_3[0])
    beta_4 = np.arctan2(A_4[1] - C_4[1], A_4[0] - C_4[0])

    C_1_c = C_1 + np.array([ r*np.cos(beta_1),  r*np.sin(beta_1), 0])
    C_2_c = C_2 + np.array([ r*np.cos(beta_2),  r*np.sin(beta_2), 0])
    C_3_c = C_3 + np.array([ r*np.cos(beta_3),  r*np.sin(beta_3), 0])
    C_4_c = C_4 + np.array([ r*np.cos(beta_4),  r*np.sin(beta_4), 0])

    #
    L_1 = np.linalg.norm(A_1 - C_1_c)
    L_2 = np.linalg.norm(A_2 - C_2_c)
    L_3 = np.linalg.norm(A_3 - C_3_c)
    L_4 = np.linalg.norm(A_4 - C_4_c)

    eps_1 = np.arccos(r / L_1)
    eps_2 = np.arccos(r / L_2)
    eps_3 = np.arccos(r / L_3)
    eps_4 = np.arccos(r / L_4)

    delta_1 = np.arccos(np.sqrt((A_1[0] - C_1_c[0])**2 + (A_1[1] - C_1_c[1])**2) / L_1)
    delta_2 = np.arccos(np.sqrt((A_2[0] - C_2_c[0])**2 + (A_2[1] - C_2_c[1])**2) / L_2)
    delta_3 = np.arccos(np.sqrt((A_3[0] - C_3_c[0])**2 + (A_3[1] - C_3_c[1])**2) / L_3)
    delta_4 = np.arccos(np.sqrt((A_4[0] - C_4_c[0])**2 + (A_4[1] - C_4_c[1])**2) / L_4)

    gamma_1 = eps_1 - delta_1
    gamma_2 = eps_2 - delta_2
    gamma_3 = eps_3 - delta_3
    gamma_4 = eps_4 - delta_4

    B_1 = C_1_c + np.array([r*np.cos(gamma_1)*np.cos(beta_1), r*np.cos(gamma_1)*np.sin(beta_1), r*np.sin(gamma_1)])
    B_2 = C_2_c + np.array([r*np.cos(gamma_2)*np.cos(beta_2), r*np.cos(gamma_2)*np.sin(beta_2), r*np.sin(gamma_2)])
    B_3 = C_3_c + np.array([r*np.cos(gamma_3)*np.cos(beta_3), r*np.cos(gamma_3)*np.sin(beta_3), r*np.sin(gamma_3)])
    B_4 = C_4_c + np.array([r*np.cos(gamma_4)*np.cos(beta_4), r*np.cos(gamma_4)*np.sin(beta_4), r*np.sin(gamma_4)])

    new_L_1 = r * (np.pi - gamma_1) + np.linalg.norm(A_1 - B_1)
    new_L_2 = r * (np.pi - gamma_2) + np.linalg.norm(A_2 - B_2)
    new_L_3 = r * (np.pi - gamma_3) + np.linalg.norm(A_3 - B_3)
    new_L_4 = r * (np.pi - gamma_4) + np.linalg.norm(A_4 - B_4)

    return new_L_1, new_L_2, new_L_3, new_L_4
# A_init = np.array([0, 0, 0])
new_L2_l, new_L2_r = 0, 0
# print(data.qpos)
# print(data.qvel)
while not glfw.window_should_close(window):
    data.qvel = [0] * len(data.qvel)
    # print(data.qpos[4:7])
    Ac = data.qpos[4:7]
    cur_L_1, cur_L_2, cur_L_3, cur_L_4 = compute_cables_lenghts(Ac)
    # break
    # print(f"{Ac = }")
    # print(Ac)
    if data.time > 2 and not received_goal:
        print()
        # print(f"\nCurrent position: {Ac[0]:.3f} {Ac[2]:.3f}")
        print("Put new desired position for the center of box.")
        print("DATA MUST BE IN 'X Y Z' FORMAT")
        print("Coordinates must be in range (-0.5; 0.5)")
        ax_new, ay_new, az_new = list(map(float, input("Enter new EE coordinates: ").split()))
        received_goal = True
        Ac_new = np.array([ax_new, ay_new, az_new])

        new_L_1, new_L_2, new_L_3, new_L_4 = compute_cables_lenghts(Ac_new)
        # print(new_A_l, new_A_r)
        # print(new_L2_l, new_L2_r)
    # print(f"{abs(cur_L2_l-new_L2_l)}, {abs(cur_L2_r-new_L2_r)}")
    if received_goal:
        print(f"{(cur_L_1 - new_L_1):.3f}, {cur_L_2 - new_L_2:.3f}, {cur_L_3 - new_L_3:.3f}, {cur_L_4 - new_L_4:.3f}")
        data.qvel[0] = .5 * (cur_L_1 - new_L_1)
        data.qvel[1] = .5 * (cur_L_2 - new_L_2)
        data.qvel[2] = .5 * (cur_L_3 - new_L_3)
        data.qvel[3] = .5 * (cur_L_4 - new_L_4)

    if received_goal and \
    abs(cur_L_1 - new_L_1) < threshold and \
    abs(cur_L_2 - new_L_2) < threshold and \
    abs(cur_L_3 - new_L_3) < threshold and \
    abs(cur_L_4 - new_L_4) < threshold:
        print(f"Reached goal")
        received_goal = False

    time_prev = data.time
    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
