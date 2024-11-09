import numpy as np
import scipy

params = {
    'cables':4,
    'A1':np.array([-1.154, -1.404, 3.220], dtype=np.float64),
    'A2':np.array([1.154, -1.404, 3.220], dtype=np.float64),
    'A3':np.array([1.154, 1.404, 3.220], dtype=np.float64),
    'A4':np.array([-1.154, 1.404, 3.220], dtype=np.float64),
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

class CDPR4:
    def __init__(self, pos, params=params, approx=1, mass=1):
        self.params = params # anchor points
        self.approx = approx # check folder /maths_behind_cdpr for approximations description
        self.pos = pos
        self.m = mass # mass of a load
        self.dt = 0.05
        self.v = np.array([0,0,0], dtype=np.float64) # end effector velocity
        self.a = 0 # end effector acceleration
        self.Kp = 1
        self.Kd = 1
        self.t_f = 5 # sec
        
        
    def inverse_kinematics_1(self, ee_pos):
        ls = np.array([0,0,0,0], dtype=np.float64) # arr of distances from anchor points to the end-effector
        ls[0] = np.linalg.norm(params['A1'] - ee_pos)
        ls[1] = np.linalg.norm(params['A2'] - ee_pos)
        ls[2] = np.linalg.norm(params['A3'] - ee_pos)
        ls[3] = np.linalg.norm(params['A4'] - ee_pos)
        
        return ls
    
    def inverse_kinematics_2(self, ee_pos):
        r = self.params['r']

        C_4 = self.params['A4'] # anchor points
        C_3 = self.params['A3']
        C_2 = self.params['A2']
        C_1 = self.params['A1']
        
        box_x, box_y, box_z = self.params['box']

        A_1 = ee_pos + np.array([-box_x*np.cos(np.pi/4),  box_x*np.cos(np.pi/4), box_z])
        A_2 = ee_pos + np.array([ box_x*np.cos(np.pi/4),  box_x*np.cos(np.pi/4), box_z])
        A_3 = ee_pos + np.array([ box_x*np.cos(np.pi/4), -box_x*np.cos(np.pi/4), box_z])
        A_4 = ee_pos + np.array([-box_x*np.cos(np.pi/4), -box_x*np.cos(np.pi/4), box_z])

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

        B_1 = C_1_c + np.array([r*np.cos(gamma_1)*np.cos(beta_1), r*np.cos(gamma_1)*np.sin(beta_1), r*np.sin(gamma_1)])
        B_2 = C_2_c + np.array([r*np.cos(gamma_2)*np.cos(beta_2), r*np.cos(gamma_2)*np.sin(beta_2), r*np.sin(gamma_2)])
        B_3 = C_3_c + np.array([r*np.cos(gamma_3)*np.cos(beta_3), r*np.cos(gamma_3)*np.sin(beta_3), r*np.sin(gamma_3)])
        B_4 = C_4_c + np.array([r*np.cos(gamma_4)*np.cos(beta_4), r*np.cos(gamma_4)*np.sin(beta_4), r*np.sin(gamma_4)])

        new_L_1 = r * (np.pi - gamma_1) + np.linalg.norm(A_1 - B_1)
        new_L_2 = r * (np.pi - gamma_2) + np.linalg.norm(A_2 - B_2)
        new_L_3 = r * (np.pi - gamma_3) + np.linalg.norm(A_3 - B_3)
        new_L_4 = r * (np.pi - gamma_4) + np.linalg.norm(A_4 - B_4)
        ls = np.array([new_L_1,new_L_2,new_L_3,new_L_4], dtype=np.float64)

        return ls
    
    def inverse_kinematics(self):
        if self.approx == 1: return self.inverse_kinematics_1(self.pos)
        if self.approx == 2: return self.inverse_kinematics_2(self.pos)
        
    def jacobian(self):
        J = np.zeros((4,3))
        
        for i in range(4):
            c_ai = self.pos - self.params[f'A{i+1}']
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
        
        X = np.hstack((self.pos, self.v), dtype=np.float64).reshape((6,1)) # TODO change 0 0 0 to initial velocities
        t = np.linspace(0, self.t_f, int(self.t_f/self.dt))
        # print(X)
        # Simulation loop
        for time in t:
            # Calculate acceleration
            v_prev = self.v
            dXdt = self.B() @ u(point, vel) + np.array([0, 0, 0, 0, 0, -g]).reshape((6,1))
            # print(dXdt)
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

    
if __name__ == '__main__':        
    robot1 = CDPR4(pos=np.array([0,0,1]))
    
    desired_p = np.array([1,1,2]) # desired point is 2 m above the ground
    desired_v = np.array([0,0,0])
    robot1.Kp = .005
    robot1.Kd = .007
    poses, vels = robot1.simulate(robot1.control_pd, desired_p, desired_v)
    # robot2 = CDPR4(approx=2)
    # print(robot2.inverse_kinematics(np.array([100,0,1000])))