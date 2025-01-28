import numpy as np
import scipy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw

CDPR4_PARAMS = {
    'cables':4,
    'A1':np.array([-1.154, -1.404, 3.220], dtype=np.float64),
    'A2':np.array([1.154, -1.404, 3.220], dtype=np.float64),
    'A3':np.array([1.154, 1.404, 3.220], dtype=np.float64),
    'A4':np.array([-1.154, 1.404, 3.220], dtype=np.float64),
    "A": np.array(
        [
            [-1.154, -1.404, 3.220],
            [1.154, -1.404, 3.220],
            [1.154, 1.404, 3.220],
            [-1.154, 1.404, 3.220],
        ],
        dtype=np.float64,
    ),
    'l1':1000.0, # servos are 1m lower than anchor points
    'l2':1000.0, # the initial length of each cable is 1m
    'l3':1000.0,
    'l4':1000.0,
    'drums_r':0.05, # m
    'cable_gage':0.003, # m
    'drums_w':1, #m
    'initial_phis':np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64), # rads at [0,0,0] position of end-effector
    'initial_ls':np.array([3.697, 3.697, 3.697, 3.697], dtype=np.float64), # [mm], A_i to  end-effector initial length
    'box':np.array([0.5, 0.5, 0.5], dtype=np.float64),
    'r':0.025 # [mm] radius of pulley
}

g = 9.81
MAX_EPISODE_STEPS = 32
USE_TARGET_VELOCITY = False
MAX_SPEED = 2
MAX_FORCE = 15
TOLERANCE = 0.01

class CDPR4:
    def __init__(self, pos, CDPR4_PARAMS=CDPR4_PARAMS, approx=1, mass=1):
        self.CDPR4_PARAMS = CDPR4_PARAMS # anchor points
        self.approx = approx # check folder /maths_behind_cdpr for approximations description
        self.pos = pos
        self.m = mass # mass of a load
        self.dt = 0.005
        self.v = np.array([0,0,0], dtype=np.float64) # end effector velocity
        self.a = 0 # end effector acceleration
        self.Kp = 1
        self.Kd = 1
        self.t_f = 5 # sec
        self.g = 9.81  # Gravity
        
        
    def inverse_kinematics_1(self, ee_pos):
        ls = np.array([0,0,0,0], dtype=np.float64) # arr of distances from anchor points to the end-effector
        ls[0] = np.linalg.norm(CDPR4_PARAMS['A1'] - ee_pos)
        ls[1] = np.linalg.norm(CDPR4_PARAMS['A2'] - ee_pos)
        ls[2] = np.linalg.norm(CDPR4_PARAMS['A3'] - ee_pos)
        ls[3] = np.linalg.norm(CDPR4_PARAMS['A4'] - ee_pos)
        
        return ls
    
    def get_real_anchor_points(self, ee_pos):
        r = self.CDPR4_PARAMS['r']

        C_4 = self.CDPR4_PARAMS['A4'] # center of the pulley
        C_3 = self.CDPR4_PARAMS['A3']
        C_2 = self.CDPR4_PARAMS['A2']
        C_1 = self.CDPR4_PARAMS['A1']
        
        box_x, box_y, box_z = self.CDPR4_PARAMS['box']

        A_1 = ee_pos + np.array([-box_x*np.cos(np.pi/4),  box_x*np.cos(np.pi/4), box_z]) # box corners in World Coords
        A_2 = ee_pos + np.array([ box_x*np.cos(np.pi/4),  box_x*np.cos(np.pi/4), box_z])
        A_3 = ee_pos + np.array([ box_x*np.cos(np.pi/4), -box_x*np.cos(np.pi/4), box_z])
        A_4 = ee_pos + np.array([-box_x*np.cos(np.pi/4), -box_x*np.cos(np.pi/4), box_z])

        beta_1 = np.arctan2(A_1[1] - C_1[1], A_1[0] - C_1[0]) # cable tilt (corner to the center of the pulley)
        beta_2 = np.arctan2(A_2[1] - C_2[1], A_2[0] - C_2[0])
        beta_3 = np.arctan2(A_3[1] - C_3[1], A_3[0] - C_3[0])
        beta_4 = np.arctan2(A_4[1] - C_4[1], A_4[0] - C_4[0])

        C_1_c = C_1 + np.array([ r*np.cos(beta_1),  r*np.sin(beta_1), 0]) # [x, y] change due to pulley dimensions
        C_2_c = C_2 + np.array([ r*np.cos(beta_2),  r*np.sin(beta_2), 0])
        C_3_c = C_3 + np.array([ r*np.cos(beta_3),  r*np.sin(beta_3), 0])
        C_4_c = C_4 + np.array([ r*np.cos(beta_4),  r*np.sin(beta_4), 0])

        # cable length, top view projection
        L_1 = np.linalg.norm(A_1 - C_1_c)
        L_2 = np.linalg.norm(A_2 - C_2_c)
        L_3 = np.linalg.norm(A_3 - C_3_c)
        L_4 = np.linalg.norm(A_4 - C_4_c)
        # print(L_1, L_2, L_3, L_1)
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

        B_1 = C_1_c + np.array([r*np.cos(gamma_1)*np.cos(beta_1), r*np.cos(gamma_1)*np.sin(beta_1), r*np.sin(gamma_1)]) # real anchor point
        B_2 = C_2_c + np.array([r*np.cos(gamma_2)*np.cos(beta_2), r*np.cos(gamma_2)*np.sin(beta_2), r*np.sin(gamma_2)])
        B_3 = C_3_c + np.array([r*np.cos(gamma_3)*np.cos(beta_3), r*np.cos(gamma_3)*np.sin(beta_3), r*np.sin(gamma_3)])
        B_4 = C_4_c + np.array([r*np.cos(gamma_4)*np.cos(beta_4), r*np.cos(gamma_4)*np.sin(beta_4), r*np.sin(gamma_4)])

        # new_L_1 = r * (np.pi - gamma_1) + np.linalg.norm(A_1 - B_1)
        # new_L_2 = r * (np.pi - gamma_2) + np.linalg.norm(A_2 - B_2)
        # new_L_3 = r * (np.pi - gamma_3) + np.linalg.norm(A_3 - B_3)
        # new_L_4 = r * (np.pi - gamma_4) + np.linalg.norm(A_4 - B_4)
        # ls = np.array([new_L_1,new_L_2,new_L_3,new_L_4], dtype=np.float64)

        return np.array([B_1, B_2, B_3, B_4])
    
    def inverse_kinematics(self):
        if self.approx == 1: return self.inverse_kinematics_1(self.pos)
        if self.approx == 2: return self.inverse_kinematics_1(self.pos) # TODO: delete and refactor the code
        
    def jacobian(self):
        J = np.zeros((4,3))
        
        for i in range(4):
            c_ai = self.pos - self.CDPR4_PARAMS[f'A{i+1}']
            li = np.linalg.norm(c_ai)# len
            J[i, :] = c_ai/li
        return J
    
    def B(self):
        B = np.zeros((6,4))
        
        lower_rows = -(1/self.m)*self.jacobian().T
        
        B[3:6, :] = lower_rows
        return B

    def control_pd(self, desired_pos, desired_vel): # PD controller
        err = -desired_pos + self.pos.reshape((3,1))
        d_err = -desired_vel + self.v.reshape((3,1))
        Kp_matrix = self.Kp * self.jacobian() #np.ones((4,3))
        Kd_matrix = self.Kd * self.jacobian() #np.ones((4,3))
        
        Gravity = self.jacobian() @ np.array([0,0,-g]).reshape((3,1))
        
        # print(err)
        return Kp_matrix@err + Kd_matrix@d_err + Gravity
        
    def simulate(self, u, point, vel):
        point = point.reshape((3,1))
        vel = vel.reshape((3,1))
        positions = []
        velocities = []
        
        X = np.hstack((self.pos, self.v), dtype=np.float64).reshape((6,1)) 
        t = np.linspace(0, self.t_f, int(self.t_f/self.dt))

        # Simulation loop
        for time in t:
            # Calculate acceleration
            v_prev = self.v
            dXdt = self.B() @ u(point, vel) + np.array([0, 0, 0, 0, 0, -g]).reshape((6,1))

            # Update velocities (last 3 elements of X)
            X[3:] += dXdt[3:] * self.dt
            self.v = X[3:].flatten() 
            
            # Update positions (first 3 elements of X)
            X[:3] += v_prev.reshape((3,1)) * self.dt + 0.5*dXdt[3:]*self.dt**2
            self.pos = X[:3].flatten()

            # Store results
            positions.append(X[:3].flatten())
            velocities.append(X[3:].flatten())
        
        return positions, velocities

# author: Alexey Korshuk
# edited: Damir Nurtdinov
class CDPR4_env(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        approx=1,
        mass=1,
        max_steps=MAX_EPISODE_STEPS,
        is_continuous=False,
        num_discretized_actions=11,
    ):
        super().__init__()
        self.cdpr = CDPR4(pos=pos, CDPR4_PARAMS=CDPR4_PARAMS, approx=approx, mass=mass)
        self.desired_state = None  # Will be set in reset
        self.cur_state = None  # Will be set in reset
        self.max_speed = MAX_SPEED
        self.max_force = MAX_FORCE
        self.dt = 0.1
        self.max_episode_steps = max_steps
        self.elapsed_steps = 0
        self.is_continuous = is_continuous
        self.num_discretized_actions = num_discretized_actions,

        # Action space: Continuous forces for each cable, scaled between -1 and 1
        if self.is_continuous:
            self.action_space = spaces.Box(
                low=-np.ones(self.cdpr.CDPR4_PARAMS["cables"], dtype=np.float32),
                high=np.ones(self.cdpr.CDPR4_PARAMS["cables"], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.MultiDiscrete(
                [num_discretized_actions] * self.cdpr.CDPR4_PARAMS["cables"]
            )

        # Modify observation space based on USE_TARGET_VELOCITY flag
        if USE_TARGET_VELOCITY:
            # Current observation space (12 dimensions)
            self.observation_space = spaces.Box(
                low=np.array(
                    [
                        -1.154 + 0.15,
                        -1.404 + 0.15,
                        0.0 + 0.15,  # Position lower bounds
                        -self.max_speed,
                        -self.max_speed,
                        -self.max_speed,  # Velocity lower bounds
                        -1.154,
                        -1.404,
                        0.0,  # Target position lower bounds
                        -self.max_speed,
                        -self.max_speed,
                        -self.max_speed,  # Desired velocity lower bounds
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        1.154 - 0.15,
                        1.404 - 0.15,
                        3.220 - 0.15,  # Position upper bounds
                        self.max_speed,
                        self.max_speed,
                        self.max_speed,  # Velocity upper bounds
                        1.154,
                        1.404,
                        3.220,  # Target position upper bounds
                        self.max_speed,
                        self.max_speed,
                        self.max_speed,  # Desired velocity upper bounds
                    ],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        else:
            # Reduced observation space (9 dimensions, without desired velocity)
            self.observation_space = spaces.Box(
                low=np.array(
                    [
                        -1.154 + 0.15,
                        -1.404 + 0.15,
                        0.0 + 0.15,  # Position lower bounds
                        -self.max_speed,
                        -self.max_speed,
                        -self.max_speed,  # Velocity lower bounds
                        -1.154,
                        -1.404,
                        0.0,  # Target position lower bounds
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        1.154 - 0.15,
                        1.404 - 0.15,
                        3.220 - 0.15,  # Position upper bounds
                        self.max_speed,
                        self.max_speed,
                        self.max_speed,  # Velocity upper bounds
                        1.154,
                        1.404,
                        3.220,  # Target position upper bounds
                    ],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )

        # Visualization variables
        self.fig = None
        self.ax = None
        self.start_pos = None  # To store the initial position
        self.previous_position_error = None
        self.render_mode = "rgb_array"
        self.last_reward = None
        self.max_possible_distance = self._precomute_max_distance()
        
        # self.reset() # commented for evaluation

    def set_max_episode_steps(self, max_steps):
        self.max_episode_steps = max_steps

    def set_num_discretized_actions(self, num_actions):
        self.num_discretized_actions = num_actions

    def _precomute_max_distance(self):
        pos_low = self.observation_space.low[:3]
        pos_high = self.observation_space.high[:3]
        corners = np.array(
            [
                [pos_low[0], pos_low[1], pos_low[2]],
                [pos_low[0], pos_low[1], pos_high[2]],
                [pos_low[0], pos_high[1], pos_low[2]],
                [pos_low[0], pos_high[1], pos_high[2]],
                [pos_high[0], pos_low[1], pos_low[2]],
                [pos_high[0], pos_low[1], pos_high[2]],
                [pos_high[0], pos_high[1], pos_low[2]],
                [pos_high[0], pos_high[1], pos_high[2]],
            ]
        )

        # Calculate distances between all pairs of corners
        max_possible_distance = 0.0
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                dist = np.linalg.norm(corners[i] - corners[j])
                max_possible_distance = max(max_possible_distance, dist)
        return max_possible_distance

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding if necessary

        # Randomly sample a starting position within bounds
        current_state = self.observation_space.sample()
        current_location = current_state[:3].flatten()
        current_velocity = np.zeros(3, dtype=np.float32)  # Start with zero velocity
        self.cur_state = np.hstack((current_location, current_velocity))

        # Randomly sample a target position within bounds
        destination_state = self.observation_space.sample()
        destination_location = destination_state[6:9].flatten()
        desired_velocity = np.zeros(
            3, dtype=np.float32
        )  # Desired velocity is zero at the target
        self.desired_state = np.hstack((destination_location, desired_velocity))

        # Update CDPR's position and velocity
        self.cdpr.pos = self.cur_state[:3].astype(np.float64)
        self.cdpr.v = self.cur_state[3:6].astype(np.float64)
        self.elapsed_steps = 0

        # Store the starting position
        self.start_pos = self.cur_state[:3].copy()

        if USE_TARGET_VELOCITY:
            # Current behavior with desired velocity
            self.cur_state = np.hstack((self.cur_state, self.desired_state)).astype(
                np.float32
            )
        else:
            # Only include position and velocity states, and target position
            self.cur_state = np.hstack(
                (
                    self.cur_state[:6],  # Current position and velocity
                    self.desired_state[:3],  # Only target position
                )
            ).astype(np.float32)

        self.previous_position_error = np.linalg.norm(
            self.cur_state[:3] - self.desired_state[:3]
        )
        # self.last_reward = None

        return self.cur_state, {}

    def step(self, action):
        assert action.shape == (self.cdpr.CDPR4_PARAMS["cables"],)
        if not self.is_continuous:
            action = action.astype(np.float32) / (self.num_discretized_actions - 1)
        else:
        # Ensure action is within bounds
            action = (np.clip(action, -1.0, 1.0) + 1)/2
        action = np.clip(action, 0.0, 1.0)

        # Scale actions to max_force
        u = self.max_force * action.reshape((self.cdpr.CDPR4_PARAMS["cables"], 1))

        # Current state
        pos = self.cur_state[:3].astype(np.float64)
        vel = self.cur_state[3:6].astype(np.float64)
        target_pos = self.cur_state[6:9].astype(np.float64)
        desired_velocity = self.cur_state[9:12].astype(np.float64)

        # State-space formulation for physics
        X = np.hstack((pos, vel))
        dXdt = (
            np.hstack((vel, np.zeros(3))).reshape((6, 1))
            + self.cdpr.B() @ u
            + np.array([0.0, 0.0, 0.0, 0.0, 0.0, -self.cdpr.g]).reshape((6, 1))
        )
        X_new = X + self.dt * dXdt.flatten()

        # Extract new position and velocity
        new_pos = X_new[:3]
        new_vel = X_new[3:]
        new_vel = np.clip(new_vel, -self.max_speed, self.max_speed)

        # Calculate distances to target before and after the action
        dist_before = np.linalg.norm(pos - target_pos)
        dist_after = np.linalg.norm(new_pos - target_pos)

        # Rest of the step function remains the same...
        reward = 0.0

        # Progress reward: Normalize the improvement relative to max possible distance
        distance_improvement = dist_before - dist_after
        normalized_improvement = distance_improvement / self.max_possible_distance
        reward += normalized_improvement * 50.0

        # Distance-based reward: Reward being close to target
        normalized_distance = dist_after / self.max_possible_distance
        proximity_reward = 1.0 - normalized_distance
        reward += proximity_reward * 5.0

        # # Action smoothness penalty: Penalize large actions
        # action_magnitude = np.linalg.norm(action)
        # reward -= 0.1 * action_magnitude

        # reward -= np.linalg.norm(pos - target_pos) * 10.0
        # reward -= 0.5 * np.linalg.norm(vel - desired_velocity)

        # Initialize termination flags
        terminated = False
        truncated = False

        # Check termination conditions based on position error and step count
        if np.allclose(new_pos, target_pos, atol=TOLERANCE, rtol=0):
            terminated = True
            reward += 10000.0
            if USE_TARGET_VELOCITY:
                current_velocity_magnitude = np.linalg.norm(new_vel)
                normalized_velocity = current_velocity_magnitude / self.max_speed
                velocity_reward = 1.0 - normalized_velocity
                reward += velocity_reward * 100.0

        # Check if the new position is out of bounds
        pos_low = self.observation_space.low[:3]
        pos_high = self.observation_space.high[:3]
        out_of_bounds = np.any(new_pos < pos_low) or np.any(new_pos > pos_high)

        if out_of_bounds and not terminated:
            # reward -= 1000.0
            terminated = True
            info = {
                "terminated": terminated,
                "truncated": truncated,
                # "position_error": current_position_error,
                "reason": "Out of Bounds",
            }
            self.last_reward = reward
            return self.cur_state, reward, terminated, truncated, info

        if self.elapsed_steps >= self.max_episode_steps and not terminated:
            truncated = True
            # reward -= 1000.0

        reward -= 5.0

        # Update state
        self.cdpr.pos = new_pos
        self.cdpr.v = new_vel
        self.cur_state = np.hstack(
            (new_pos, new_vel, target_pos, desired_velocity)
        ).astype(np.float32)
        self.elapsed_steps += 1

        # Additional info
        info = {
            "terminated": terminated,
            "truncated": truncated,
            # "position_error": current_position_error,
        }
        self.last_reward = reward
        return self.cur_state, reward, terminated, truncated, info

    def render(self, reward=None, mode=None):
        reward = reward if reward is not None else self.last_reward
        if mode is None:
            mode = self.render_mode

        # Render the environment and return an RGB array if needed
        if self.fig is None or self.ax is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")
            plt.ion()  # Interactive mode on

        self.ax.clear()

        # Plot anchor points
        A = self.cdpr.CDPR4_PARAMS["A"]
        self.ax.scatter(A[:, 0], A[:, 1], A[:, 2], c="r", marker="o", label="Anchors")

        # Plot starting point
        start_pos = self.start_pos  # Stored starting position
        self.ax.scatter(
            start_pos[0],
            start_pos[1],
            start_pos[2],
            c="g",
            marker="s",
            label="Start Point",
        )

        # Plot target point
        target_pos = self.desired_state[:3]
        self.ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            c="b",
            marker="*",
            s=100,
            label="Target Point",
        )

        # Plot end effector
        end_effector_pos = self.cur_state[:3]
        self.ax.scatter(
            end_effector_pos[0],
            end_effector_pos[1],
            end_effector_pos[2],
            c="b",
            marker="^",
            label="End Effector",
        )

        current_velocity = self.cur_state[3:6]
        desired_velosity = self.desired_state[3:6]

        # Draw cables
        for i in range(self.cdpr.CDPR4_PARAMS["cables"]):
            self.ax.plot(
                [A[i, 0], end_effector_pos[0]],
                [A[i, 1], end_effector_pos[1]],
                [A[i, 2], end_effector_pos[2]],
                "k-",
            )

        str_reward = f"Reward: {reward:.2f}" if reward is not None else "Reward: N/A"
        if USE_TARGET_VELOCITY:
            title_str = (
                f"Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) | "
                f"Current: ({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f}) | "
                f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})\n"
                f"Current Velocity: ({current_velocity[0]:.2f}, {current_velocity[1]:.2f}, {current_velocity[2]:.2f}) | "
                f"Desired Velocity: ({desired_velosity[0]:.2f}, {desired_velosity[1]:.2f}, {desired_velosity[2]:.2f})\n"
                f"Step: {self.elapsed_steps} | {str_reward}"
            )
        else:
            title_str = (
                f"Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}) | "
                f"Current: ({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f}, {end_effector_pos[2]:.2f}) | "
                f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})\n"
                f"Current Velocity: ({current_velocity[0]:.2f}, {current_velocity[1]:.2f}, {current_velocity[2]:.2f})\n"
                f"Step: {self.elapsed_steps} | {str_reward}"
            )

        # Set the suptitle
        self.fig.suptitle(title_str, fontsize=12)

        # Set plot limits
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 4])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.legend()

        plt.draw()
        plt.pause(0.001)

        if mode == "rgb_array":
            # Convert the current figure to an RGB array
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

class CDPR4_MuJoCo_env(gym.Env):
    def __init__(self):
        super(CDPR4_MuJoCo_env, self).__init__()
        
        # Load MuJoCo model
        xml_path = '../mujoco_models/four_tendons.xml' 
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Reduced observation space (9 dimensions, without desired velocity)
        self.observation_space = spaces.Box(
                low=np.array(
                    [
                        -1.154 + 0.15,
                        -1.404 + 0.15,
                        0.0 + 0.15,  # Position lower bounds
                        -self.max_speed,
                        -self.max_speed,
                        -self.max_speed,  # Velocity lower bounds
                        -1.154,
                        -1.404,
                        0.0,  # Target position lower bounds
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        1.154 - 0.15,
                        1.404 - 0.15,
                        3.220 - 0.15,  # Position upper bounds
                        self.max_speed,
                        self.max_speed,
                        self.max_speed,  # Velocity upper bounds
                        1.154,
                        1.404,
                        3.220,  # Target position upper bounds
                    ],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        
        # Initialize state
        self.cur_state = np.zeros(12)
        self.max_episode_steps = 1000
        self.elapsed_steps = 0
        
        # Rendering
        self.fig = None
        self.ax = None
        
    def step(self, action):
        # Apply action to actuators
        self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
        
        # Step the simulation
        mj.mj_step(self.model, self.data)
        
        # Update state
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        target_pos = self.cur_state[6:9]
        # desired_velocity = self.cur_state[9:12] # no desired velocity
        
        self.cur_state = np.hstack((pos, vel, target_pos)) # no desired velocity
        
        # Calculate reward, termination, etc.
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.elapsed_steps >= self.max_episode_steps
        
        self.elapsed_steps += 1
        
        return self.cur_state, reward, terminated, truncated, {}
    
    def reset(self):
        # Reset the simulation
        mj.mj_resetData(self.model, self.data)
        
        # Reset state
        self.cur_state = np.zeros(12)
        self.elapsed_steps = 0
        
        return self.cur_state
    
    def render(self, mode='human'):
        if mode == 'human':
            if self.fig is None or self.ax is None:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                plt.ion()
            
            self.ax.clear()
            
            # Plot end effector
            pos = self.cur_state[:3]
            self.ax.scatter(pos[0], pos[1], pos[2], c='b', marker='^', label='End Effector')
            
            # Plot target
            target_pos = self.cur_state[6:9]
            self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='r', marker='*', label='Target')
            
            # Set plot limits
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            self.ax.set_zlim([0, 4])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.legend()
            
            plt.draw()
            plt.pause(0.001)
        
        elif mode == 'rgb_array':
            # Use MuJoCo's renderer to generate an RGB array
            return mj.MjRenderContext(self.model, self.data).read_pixels()

    def _calculate_reward(self):
        # Implement your reward calculation here
        return 0.0

    def _check_termination(self):
        # Implement your termination condition here
        return False

    
if __name__ == '__main__':        
    robot1 = CDPR4(pos=np.array([0,0,1]))
    
    desired_p = np.array([1,1,2]) # desired point is 2 m above the ground
    desired_v = np.array([0,0,0])
    robot1.Kp = .005
    robot1.Kd = .007
    poses, vels = robot1.simulate(robot1.control_pd, desired_p, desired_v)
    # robot2 = CDPR4(approx=2)
    # print(robot2.inverse_kinematics(np.array([100,0,1000])))