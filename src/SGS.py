import copy

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import pdb

from src.PID import PID

class SGS:
    def __init__(self):
        self.K_0 = 273.15  # [K]

        self.T_ERROR = 1.5
        self.M_ERROR = 2

        self.ALPHA_1 = 2.0e-05  # [m^3]
        self.ALPHA_2 = 1.0e-06  # [m^3/K]

        self.HEIGHT = 3.  # [m]
        self.SIDE = 10.  # [m]
        self.AREA = self.SIDE ** 2  # [m^2]
        self.V_ROOM = self.AREA * self.HEIGHT  # [m^3]
        self.THICKNESS = 0.03  # [m]

        self.DEPTH = 0.2  # [m]
        self.Vs = self.AREA * self.DEPTH  # [m^3]

        self.RHO_A = 1.204  # [kg/m^3] density air
        self.RHO_W = 1  #997  # [kg/m^3] density water
        self.LAMBDA_G = 0.27  # [W/(mK)] thermal conductivity glass
        self.AREA_G = 4 * self.HEIGHT * self.SIDE + self.AREA  # [m^2] area glass
        self.Cp_A = 1005  # [J/(kgK)] specific heat air
        self.SIGMA = 5.67e-08  # [W/(m^2K^4)]
        self.EPSILON = 0.9  # [dimensionless]
        self.Cp_W = 4.196  # [J/(KgK)] specific heat air
        self.K = 1.08  # [dimensionless]
        self.BETA = self.V_ROOM * self.RHO_A * self.Cp_A  # [J/K]

        self.VARIANCE = 5
        self.SHIFT = 4
        self.DAY = 60*60*24
        self.T_BAR = self.K_0 - 20

        self.HORIZON = 3
        self.WEIGHT = 1

        self.DAYS = 1
        self.PERIOD = self.DAY * self.DAYS  # [s]

        self.simulate_M_target(self.DAYS)  # [dimensionless]
        self.simulate_V_target(self.DAYS)  # [m^3]
        self.simulate_T_target(self.DAYS)  # [K]

        self.T_w = self.K_0 + 5  # [K]
        self.T_0 = self.T_target_history[0]  # [K]
        self.V_0 = self.V_from_M(60)  # [m^3]

        self.V = self.V_0
        self.T = self.T_0
        self.M = 0
        self.update_M()

        self.min_P = -250
        self.max_P = 250
        self.min_F = 0  # [m^3]
        self.max_F = 0.01  # [m^3]

        self.timesteps = np.asarray(range(self.PERIOD))
        self.V_history = np.zeros(self.PERIOD)
        self.T_history = np.zeros(self.PERIOD)
        self.M_history = np.zeros(self.PERIOD)
        self.T_e_history = self.simulate_temperature()
        self.F_history = np.zeros(self.PERIOD)
        self.P_history = np.zeros(self.PERIOD)

        self.o_M_biggest = 0
        self.o_T_biggest = 0
        self.o_P_biggest = 0
        self.o_F_biggest = 0

        self.rise_time = None
        self.overshoot = None
        self.settling_time = None
        self.steady_state_error = None

        self.PID_V = PID("V")
        self.PID_T = PID("T")
        self.PID_V.set("V")
        self.PID_T.set("T")

        self.Q_matrix = np.diag([1e-2, 1e-4])  # covariance matrix for process noise
        self.R_matrix = np.diag([1e-4, 1e-2])  # covariance matrix for measurement noise
        self.P_matrix = np.diag([0.1, 0.1])
        self.P_k_1 = self.P_matrix

    def run(self):
        for t in range(self.PERIOD):
            self.T_e = self.T_e_history[t]

            V = self.get_V()
            T = self.get_T()
            M = self.get_M()

            u_F, u_P = self.control_PID(V, T, t)

            x = (self.V, self.T)
            u = (u_F, u_P)

            self.step(x, t, u)

        self.measures("T")

    def control_PID(self, V, T, t):

        self.V_target = self.V_target_history[t]
        self.T_target = self.T_target_history[t]

        u_F = self.PID_V.control(self.V_target, V)
        u_P = self.PID_T.control(self.T_target, T)

        u_F = self.check_F(u_F, t)
        u_P = self.check_P(u_P, t)

        self.P_history[t] = u_P
        self.F_history[t] = u_F

        return u_F, u_P

    def get_T(self):
        return self.T

    def get_V(self):
        return self.V

    def get_M(self):
        return self.M

    def update_M(self):
        self.M = self.V / (self.V + self.Vs) * 100

    def update_V(self, dVdt):
        self.V = max(0, self.V + dVdt)

    def update_T(self, dTdt):
        self.T = max(0, self.T + dTdt)

    def step_V(self, x, u):
        V, T = x
        F, P = u
        return self.flux(F) + self.evapotranspiration(T)

    def step_T(self, x, u):
        V, T = x
        F, P = u

        conduction = (1 / self.BETA) * self.conduction(T)
        water_conduction = (1 / self.BETA) * self.water_conduction(V, T)
        radiation = (1 / self.BETA) * self.radiation(T)
        air_conditioner = (1 / self.BETA) * self.air_conditioner(P, T)

        return air_conditioner + conduction + water_conduction + radiation

    def ODE(self, x, u):
        """
        Define ODEs of the system.
        """
        dVdt = self.step_V(x, u)
        dTdt = self.step_T(x, u)

        return dVdt, dTdt

    def step(self, x, t, u):
        """
        Do single discrete step.
        """
        dVdt, dTdt = self.ODE(x, u)

        self.update_V(dVdt)
        self.update_T(dTdt)
        self.update_M()

        self.V_history[t] = self.V
        self.T_history[t] = self.T
        self.M_history[t] = self.M

        return self.V, self.T

    def step_without_update(self, x, u):
        dVdt, dTdt = self.ODE(x, u)

        V = self.V + dVdt
        T = self.T + dTdt

        return V, T

    def flux(self, F):
        return F

    def conduction(self, T):
        return 1/self.THICKNESS * self.LAMBDA_G * self.AREA_G * (self.T_e - T)

    def radiation(self, T):
        term_e = (self.SIGMA**(-4) + self.EPSILON**(-4) * self.T_e)**4
        term_i = (self.SIGMA**(-4) + self.EPSILON**(-4) * T) ** 4
        return term_e - term_i

    def water_conduction(self, V, T):
        return self.Cp_W * V * self.RHO_W * (self.T_w - T)

    def air_conditioner(self, P, T):
        return P * self.K * self.T_target - T

    def evapotranspiration(self, T):
        return -(self.ALPHA_1 + self.ALPHA_2 * T)

    def check_F(self, u_F, t):
        u_F_previous = self.F_history[t - 1] if t > 0 else 0

        if u_F > self.max_F:
            u_F = self.max_F
        elif u_F < self.min_F:
            u_F = self.min_F

        if abs(u_F - u_F_previous) > self.o_F_biggest:
            self.o_F_biggest = abs(u_F - u_F_previous)

        return u_F

    def check_P(self, u_P, t):
        u_P_previous = self.P_history[t-1] if t > 0 else 0

        if u_P > self.max_P:
            u_P = self.max_P
        elif u_P < self.min_P:
            u_P = self.min_P

        if u_P - u_P_previous > 100:
            u_P = u_P_previous + 100
        if u_P_previous - u_P > 100:
            u_P = u_P_previous - 100

        if abs(u_P - u_P_previous) > self.o_P_biggest:
            self.o_P_biggest = abs(u_P - u_P_previous)

        return u_P

    def measures(self, variable):
        if variable == "T":
            history = self.T_history
            target = self.T_target_history
            lower_bound = self.T_lower_bound_history
            upper_bound = self.T_upper_bound_history
        if variable == "M":
            history = self.M_history
            target = self.M_target_history
            lower_bound = self.M_lower_bound_history
            upper_bound = self.M_upper_bound_history
        if variable == "V":
            history = self.V_history
            target = self.V_target_history
            lower_bound = self.V_lower_bound_history
            upper_bound = self.V_upper_bound_history

        print(f"Biggest o(t), F: {self.o_F_biggest}")
        print(f"Biggest o(t), P: {self.o_P_biggest}")
        self.robustness(history, lower_bound, upper_bound)

        #self.get_rise_time(history, target)
        #self.get_overshoot(history, target)
        #self.get_settling_time(history, min_oscillation)
        #self.get_steady_state_error(history, target)

    def robustness(self, history, lower_bound, upper_bound):
        """
        Checking divergence from bounds.
        """
        robustness = np.infty
        temp = 0
        for t in range(len(history)):
            if history[t] - lower_bound[t] < robustness:
                robustness = history[t] - lower_bound[t]
                temp = self.T_e_history[t]
            if upper_bound[t] - history[t] < robustness:
                robustness = upper_bound[t] - history[t]

    def print_measures(self, variable):
        print(f'Performance measures of {variable}:')
        print(f'------------------------------------')
        print(f'Rise time: {self.rise_time}')
        print(f'Overshoot: {self.overshoot}')
        print(f'Settling time: {self.settling_time}')
        print(f'Steady state error: {self.steady_state_error}')

    def get_rise_time(self, history, target):
        self.rise_time = None
        below = True if history[0] < target else False
        for t in self.timesteps:
            if below and history[t] > target:
                self.rise_time = t
                break
            elif not below and history[t] < target:
                self.rise_time = t
                break

    def get_overshoot(self, history, target):
        maximum = np.amax(history)
        self.overshoot = maximum - target

    def get_settling_time(self, history, min_oscillation):
        self.settling_time = None
        previous = history[0]
        for t in self.timesteps[1:]:
            difference = history[t] - previous
            previous = history[t]
            if abs(difference) < min_oscillation:
                self.settling_time = t
                break

    def get_steady_state_error(self, history, target):
        self.steady_state_error = None
        if self.settling_time is not None:
            self.steady_state_error = abs(history[self.settling_time] - target)

    def plot_variable(self, variable):
        """
        Function to plot chosen quantity.
        """

        hours_timesteps = np.array(self.timesteps)/3600

        if variable == "M":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))

            # Subplot for M
            ax1.plot(hours_timesteps, self.M_history)
            ax1.plot(hours_timesteps[6*60*60: 20*60*60], self.M_target_history[6*60*60: 20*60*60], color='orange', linestyle='--')
            ax1.plot(hours_timesteps[6*60*60: 20*60*60], self.M_lower_bound_history[6*60*60: 20*60*60], color='red', linestyle='--')
            ax1.plot(hours_timesteps[6*60*60: 20*60*60], self.M_upper_bound_history[6*60*60: 20*60*60], color='red', linestyle='--')

            ax1.set_xticklabels([f'{i:.0f}' for i in ax1.get_xticks()])

            ax1.set_ylabel('VWC')
            ax1.set_xlabel('Hours')
            ax1.set_title('Moisture')

            # Subplot for M
            ax2.plot(hours_timesteps, self.F_history)
            ax2.set_ylabel('$m^3$')
            ax2.set_xlabel('Hours')
            ax2.set_title('Flux')

            ax2.set_xticklabels([f'{i:.0f}' for i in ax2.get_xticks()])

        if variable == "T":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
            # Subplot for T
            ax1.plot(hours_timesteps, self.T_history)
            ax1.plot(hours_timesteps, self.T_target_history, color='orange', linestyle='--')
            ax1.plot(hours_timesteps, self.T_lower_bound_history, color='red', linestyle='--')
            ax1.plot(hours_timesteps, self.T_upper_bound_history, color='red', linestyle='--')
            ax1.set_ylim([self.K_0 - 20, self.K_0 + 30])

            ax1.set_ylabel('K')
            ax1.set_xlabel('Hours')
            ax1.set_title('Temperature')

            ax1.set_xticklabels([f'{i:.0f}' for i in ax1.get_xticks()])

            # Subplot for P
            ax2.plot(hours_timesteps, self.P_history)

            ax2.set_ylabel('W/K')
            ax2.set_xlabel('Hours')
            ax2.set_title('Power')

            ax2.set_xticklabels([f'{i:.0f}' for i in ax2.get_xticks()])


            '''
            # Subplot for T_e
            ax3.plot(self.timesteps, self.T_e_history)
            ax3.set_ylabel('K')
            ax3.set_xlabel('Seconds')
            ax3.set_title('Temperature external')
            '''

        plt.show()

    def simulate_temperature(self):
        """
        Function to simulate temperature over a given time interval.
        """
        self.T_3 = np.random.normal(0, 1)
        temperatures = [self.temperature(t) for t in self.timesteps]
        return temperatures

    def temperature(self, t):
        """
        Function to simulate temperature as a function of time.
        """
        T_1 = self.VARIANCE * np.sin(2 * np.pi * t / self.DAY + self.SHIFT)
        T_2 = self.T_BAR
        if t % 60 == 0:
            self.T_3 = np.random.normal(0, 1)
        T = T_1 + T_2 + self.T_3
        return T

    def EKF_observe(self, y, u, H_k, dt):
        """State Estimation by EKF given noisy observations of the system"""
        x_k_1 = copy.deepcopy(y)
        x_k, P_k = self.EKF_predict(x_k_1, u, dt)
        return self.update(x_k, y, P_k, H_k)

    def EKF_predict(self, x_k_1, u_k, dt):
        """Predict step of the EKF"""
        x_k = self.step_without_update(x_k_1, u_k)  # x_k|k-1: predicted state estimate
        F_k = self.evaluate_matrices(x_k[0], x_k[1], u_k)[0]  # Jacobian of the dynamics at the predicted state
        P_k = np.matmul(np.matmul(F_k, self.P_k_1),
                        np.transpose(F_k)) + self.Q_matrix  # P_k|k-1: predicted covariance estimate
        return x_k, P_k

    def evaluate_matrices(self, V, T, u):
        """Evaluate Jacobians at a given operating point"""
        A, B = self.get_matrices()
        A = copy.deepcopy(A)
        B = copy.deepcopy(B)

        F = sp.Symbol("F")
        P = sp.Symbol("P")
        T = sp.Symbol("T")
        V = sp.Symbol("V")

        A[0][0] = sp.lambdify([V, T, F], A[0][0], 'numpy')
        a00 = A[0][0](V, T, F)
        A[0][1] = sp.lambdify([V, T, F], A[0][1], 'numpy')
        a01 = A[0][1](V, T, F)
        A[1][0] = sp.lambdify([V, T, P],  A[1][0], 'numpy')
        a10 = A[1][0](V, T, P)
        A[1][1] = sp.lambdify([V, T, P], A[1][1], 'numpy')
        a11 = A[1][1](V, T, P)
        B[0][0] = sp.lambdify([V, T, F], B[1][0], 'numpy')
        b00 = B[0][0](V, T, F)
        B[1][0] = sp.lambdify([V, T, P], B[1][0], 'numpy')
        b01 = B[1][0](V, T, P)
        AA = np.array([[a00, a01], [a10, a11]])
        BB = np.array([[b00], [b01]])
        return AA, BB

    def symbolic_diff(self):

        F = sp.Symbol("F")
        P = sp.Symbol("P")
        T = sp.Symbol("T")
        V = sp.Symbol("V")

        dVdt = F - self.ALPHA_1 - self.ALPHA_2 * T
        dTdt = 1/self.THICKNESS * self.LAMBDA_G * self.AREA_G * (self.T_e - T) + \
                (self.SIGMA ** (-4) + self.EPSILON ** (-4) * self.T_e) ** 4 - \
                (self.SIGMA ** (-4) + self.EPSILON ** (-4) * T) ** 4 + \
                self.Cp_W * V * self.RHO_W * (self.T_w - T) + \
                P * self.K * (self.T_target - T)

        df1dV = dVdt.diff(V)
        df1dT = dVdt.diff(T)
        df2dV = dTdt.diff(V)
        df2dT = dTdt.diff(T)
        df1du = dVdt.diff(F)
        df2du = dTdt.diff(P)

        return [df1dV, df1dT, df2dV, df2dT, df1du, df2du]

    def get_matrices(self):
        """Get symbolic Jacobians of the dynamics of the CSTR"""
        df1dV, df1dT, df2dV, df2dT, df1du, df2du = self.symbolic_diff()
        A = np.array([[df1dV, df1dT], [df1dV, df2dT]])
        B = np.array([[df1du], [df2du]])  # u can be considered both P and F?
        return A, B

    def update(self, x_k, y_k, P_k, H_k):
        """Update step of the EKF"""
        y_k = np.array(y_k)
        x_k = np.array(x_k)
        z_k = y_k - x_k  # z_k: innovation
        S_k = self.R_matrix + np.matmul((np.matmul(H_k, P_k)), np.transpose(H_k))  # S_k: residual covariance
        K_k = np.matmul(np.matmul(P_k, np.transpose(H_k)), np.linalg.inv(S_k)) # K_k: near optimal Kalman gain
        x_kk = x_k + np.dot(K_k, z_k)  # x_k|k: updated state estimate
        P_kk = np.matmul((np.eye(2) - np.matmul(K_k, H_k)), P_k) # P_k|k: updated covariance estimate
        self.P_k_1 = copy.deepcopy(P_kk)
        return x_kk

    def simulate_T_target(self, DAYS):
        self.T_target_history = None
        self.T_lower_bound_history = None
        self.T_upper_bound_history = None

        T_DAY = self.K_0 + 20
        T_NIGHT = self.K_0 + 20

        for DAY in range(DAYS):
            # T
            T_0_4 = [T_NIGHT] * 60*60*4
            T_4_20 = [T_DAY] * 60*60*16
            T_20_0 = [T_NIGHT] * 60*60*4
            if self.T_target_history is None:
                self.T_target_history = T_0_4 + T_4_20 + T_20_0
            else:
                self.T_target_history += T_0_4 + T_4_20 + T_20_0

            # Upper bound
            T_0_6 = [T_NIGHT - self.T_ERROR] * 60 * 60 * 6
            T_6_20 = [T_DAY - self.T_ERROR] * 60 * 60 * 14
            T_20_0 = [T_NIGHT - self.T_ERROR] * 60 * 60 * 4
            if self.T_lower_bound_history is None:
                self.T_lower_bound_history = T_0_6 + T_6_20 + T_20_0
            else:
                self.T_lower_bound_history += T_0_6 + T_6_20 + T_20_0

            # Lower bound
            T_0_4 = [T_NIGHT + self.T_ERROR] * 60 * 60 * 4
            T_4_22 = [T_DAY + self.T_ERROR] * 60 * 60 * 18
            T_22_0 = [T_NIGHT + self.T_ERROR] * 60 * 60 * 2
            if self.T_upper_bound_history is None:
                self.T_upper_bound_history = T_0_4 + T_4_22 + T_22_0
            else:
                self.T_upper_bound_history += T_0_4 + T_4_22 + T_22_0

        self.T_target_history = np.array(self.T_target_history)
        self.T_lower_bound_history = np.array(self.T_lower_bound_history)
        self.T_upper_bound_history = np.array(self.T_upper_bound_history)

    def simulate_M_target(self, DAYS):
        self.M_target_history = None
        self.M_lower_bound_history = None
        self.M_upper_bound_history = None
        for DAY in range(DAYS):
            # M
            M_0_4 = [50] * 60 * 60 * 4
            M_4_20 = [70] * 60 * 60 * 16
            M_20_0 = [50] * 60 * 60 * 4
            if self.M_target_history is None:
                self.M_target_history = M_0_4 + M_4_20 + M_20_0
            else:
                self.M_target_history += M_0_4 + M_4_20 + M_20_0

            # Lower bound
            M_0_6 = [50 - self.M_ERROR] * 60 * 60 * 6
            M_6_20 = [70 - self.M_ERROR] * 60 * 60 * 14
            M_20_0 = [50 - self.M_ERROR] * 60 * 60 * 4

            if self.M_lower_bound_history is None:
                self.M_lower_bound_history = M_0_6 + M_6_20 + M_20_0
            else:
                self.M_lower_bound_history += M_0_6 + M_6_20 + M_20_0

            # Upper bound
            M_0_24 = [70 + self.M_ERROR] * 60 * 60 * 24

            if self.M_upper_bound_history is None:
                self.M_upper_bound_history = M_0_24
            else:
                self.M_upper_bound_history += M_0_24

        self.M_target_history = np.array(self.M_target_history)
        self.M_lower_bound_history = np.array(self.M_lower_bound_history)
        self.M_upper_bound_history = np.array(self.M_upper_bound_history)

    def simulate_V_target(self, DAYS):
        self.V_target_history = np.array([self.V_from_M(M) for M in self.M_target_history])

    def V_from_M(self, M):
        return (self.Vs * M) / (100 - M)

    '''
    def MPC_cost(self, x, t, u):
        J = 0
        u_F, u_P = u
        for t in range(self.HORIZON):
            V, T = self.step(x, t, u)
            J += (self.V_target-V)**2+(self.T_target-T)**2+self.WEIGHT*(u_F_previous-u_F)**2+(u_P_previous-u_P)**2
            x = V, T
        return J

    def MPC_control(self, x, t, u):
        y = np.zeros([2, self.HORIZON + 1])  # first row: V, second row: T
        y[:, 0] = x
        u_init = np.zeros(self.HORIZON + 1)
        solution = minimize(self.MPC_cost, u_init, args=(y, t, u_init))  # bounds?

        return solution.x[0]

    def mpc(self, ref, t):
        """Compute optimal input u"""
        x0 = [self.model.Ca0, self.model.T0]  # initial conditions
        ustar = np.ones(len(t) + self.horizon + 1)  # initializing control vector
        # reactor variables
        Ca = np.ones(len(t)+self.horizon+1) * self.model.Ca0
        T = np.ones(len(t)+self.horizon+1) * self.model.T0
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]  # current time step
            ustar[i] = self.control(x0, ref, dt)  # minimize cost function
            x0 = self.model.discrete_step(x0, dt, ustar[i], Ca, T, i)  # make a discrete step
            x0 = np.array(x0)
            # adding noise
            x0[0] += np.random.normal(0, 0.1)
            x0[1] += np.random.normal(0, 0.1)
            if self.observer is not None:
                # estimate state
                x0 = self.observer.observe(x0, ustar[i], np.eye(2), dt)
        return Ca, T, ustar

    def MPC(self):
    '''
