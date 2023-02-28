import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.PID import PID
from src.monitor import monitor, load_results


def calculate_oscillations(history):
    oscillations = [0] + [abs(history[t - 1] - history[t]) for t in range(1, len(history))]
    return oscillations


def calculate_steady_state_error(history, target_history):
    steady_state_error = [abs(history[t] - target_history[t]) for t in range(len(history))]
    return steady_state_error


class SGS:
    def __init__(self, T_BAR=None, AMPLITUDE=None):
        self.oT_threshold = 1
        self.oM_threshold = 0.6
        self.sT_threshold = 2
        self.sM_threshold = 2

        self.K_0 = 273.15  # [K]

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
        self.RHO_W = 1  # 997  # [kg/m^3] density water
        self.LAMBDA_G = 0.27  # [W/(mK)] thermal conductivity glass
        self.AREA_G = 4 * self.HEIGHT * self.SIDE + self.AREA  # [m^2] area glass
        self.Cp_A = 1005  # [J/(kgK)] specific heat air
        self.SIGMA = 5.67e-08  # [W/(m^2K^4)]
        self.EPSILON = 0.9  # [dimensionless]
        self.Cp_W = 4.196  # [J/(KgK)] specific heat air
        self.K = 1.08  # [dimensionless]
        self.BETA = self.V_ROOM * self.RHO_A * self.Cp_A  # [J/K]

        self.SHIFT = 4
        self.HOUR = 60 * 60
        self.DAY = self.HOUR * 24
        self.T_BAR = self.K_0 - 20 if T_BAR is None else T_BAR
        self.AMPLITUDE = 5 if AMPLITUDE is None else AMPLITUDE

        self.T_TARGET = self.K_0 + 20
        self.M_TARGET = 70

        self.PERIOD = self.HOUR * 2  # [s]

        self.simulate_M_target()  # [dimensionless]
        self.simulate_V_target()  # [m^3]
        self.simulate_T_target()  # [K]

        self.T_w = self.K_0 + 5  # [K]
        self.T_0 = self.K_0 + 10  # [K]
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

        self.rise_time = None
        self.overshoot = None
        self.settling_time = None
        self.steady_state_error = None

        self.PID_V = PID("V")
        self.PID_T = PID("T")
        self.PID_V.set("V")
        self.PID_T.set("T")

        self.filename = f'{self.T_BAR}_{self.AMPLITUDE}'

    def run(self):
        """
        Function to run simulation and write history of variables on file.
        """
        for t in range(self.PERIOD):
            self.T_e = self.T_e_history[t]

            V = self.get_V()
            T = self.get_T()
            M = self.get_M()

            u_F, u_P = self.control_PID(V, T, t)

            x = (self.V, self.T)
            u = (u_F, u_P)

            self.step(x, t, u)

        self.write_data()

    def control_PID(self, V, T, t):
        """
        Call PID control for current step.
        :param V: current volume
        :param T: current temperature
        :param t: current timestep
        """

        self.V_target = self.V_target_history[t]
        self.T_target = self.T_target_history[t]

        u_F = self.PID_V.control(self.V_target, V)
        u_P = self.PID_T.control(self.T_target, T)

        u_F = self.check_F(u_F)
        u_P = self.check_P(u_P)

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
        """
        Calculate rate of difference of volume given current state and control variables.
        :param x: current state
        :param u: control variables
        """
        V, T = x
        F, P = u
        return self.flux(F) + self.evapotranspiration(T)

    def step_T(self, x, u):
        """
        Calculate rate of difference of temperature given current state and control variables.
        :param x: current state
        :param u: control variables
        """
        V, T = x
        F, P = u

        conduction = (1 / self.BETA) * self.conduction(T)
        water_conduction = (1 / self.BETA) * self.water_conduction(V, T)
        radiation = (1 / self.BETA) * self.radiation(T)
        air_conditioner = (1 / self.BETA) * self.air_conditioner(P, T)

        return air_conditioner + conduction + water_conduction + radiation

    def ODE(self, x, u):
        """
        Calculate rates of change of volume and temperature differences.
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

    def flux(self, F):
        return F

    def conduction(self, T):
        return 1 / self.THICKNESS * self.LAMBDA_G * self.AREA_G * (self.T_e - T)

    def radiation(self, T):
        term_e = (self.SIGMA ** (-4) + self.EPSILON ** (-4) * self.T_e) ** 4
        term_i = (self.SIGMA ** (-4) + self.EPSILON ** (-4) * T) ** 4
        return term_e - term_i

    def water_conduction(self, V, T):
        return self.Cp_W * V * self.RHO_W * (self.T_w - T)

    def air_conditioner(self, P, T):
        return P * self.K * self.T_target - T

    def evapotranspiration(self, T):
        return -(self.ALPHA_1 + self.ALPHA_2 * T)

    def check_F(self, u_F):
        """
        Limit flux to physical requirements.
        :param u_F: flux control
        """
        if u_F > self.max_F:
            u_F = self.max_F
        elif u_F < self.min_F:
            u_F = self.min_F

        return u_F

    def check_P(self, u_P):
        """
        Limit power to physical requirements.
        :param u_P: power control
        """
        if u_P > self.max_P:
            u_P = self.max_P
        elif u_P < self.min_P:
            u_P = self.min_P

        return u_P

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
        Function to plot chosen history.
        """

        minutes = np.array(self.timesteps) / 60

        if variable == "M":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))

            # Subplot for M
            ax1.plot(minutes, self.M_history)
            ax1.plot(minutes, self.M_target_history,
                     color='orange', linestyle='--')
            ax1.plot(minutes, self.M_lower_bound_history,
                     color='red', linestyle='--')
            ax1.plot(minutes, self.M_upper_bound_history,
                     color='red', linestyle='--')

            #ax1.set_xticklabels([f'{i:.0f}' for i in ax1.get_xticks()])

            ax1.set_ylabel('VWC')
            ax1.set_xlabel('Minutes')
            ax1.set_title('Moisture')

            # Subplot for M
            ax2.plot(minutes, self.F_history)
            ax2.set_ylabel('$m^3$')
            ax2.set_xlabel('Minutes')
            ax2.set_title('Flux')

            #ax2.set_xticklabels([f'{i:.0f}' for i in ax2.get_xticks()])

        if variable == "T":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))
            # Subplot for T
            ax1.plot(minutes, self.T_history)
            ax1.plot(minutes, self.T_target_history, color='orange', linestyle='--')
            ax1.plot(minutes, self.T_lower_bound_history, color='red', linestyle='--')
            ax1.plot(minutes, self.T_upper_bound_history, color='red', linestyle='--')
            ax1.set_ylim([self.K_0 - 20, self.K_0 + 30])

            ax1.set_ylabel('K')
            ax1.set_xlabel('Minutes')
            ax1.set_title('Temperature')

            #ax1.set_xticklabels([f'{i:.0f}' for i in ax1.get_xticks()])

            # Subplot for P
            ax2.plot(minutes, self.P_history)

            ax2.set_ylabel('W')
            ax2.set_xlabel('Minutes')
            ax2.set_title('Power')

            #ax2.set_xticklabels([f'{i:.0f}' for i in ax2.get_xticks()])

        if variable == "Te":
            # Subplot for T
            plt.plot(minutes, self.T_e_history)

            plt.ylabel('K')
            plt.xlabel('Minutes')
            plt.title('Temperature')

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
        T_1 = self.AMPLITUDE * np.sin(2 * np.pi * t / self.DAY + self.SHIFT)
        T_2 = self.T_BAR
        if t % 60 == 0:
            self.T_3 = np.random.normal(0, 1)
        T = T_1 + T_2 + self.T_3
        return T

    def simulate_T_target(self):
        # T
        T_TARGET_LIST = [self.T_TARGET] * self.PERIOD
        self.T_target_history = T_TARGET_LIST

        # Lower bound
        self.T_lower_bound = self.T_TARGET - self.sT_threshold
        T_TARGET_LOWER_LIST = [self.T_lower_bound] * self.PERIOD
        self.T_lower_bound_history = T_TARGET_LOWER_LIST

        # Upper bound
        self.T_upper_bound = self.T_TARGET + self.sT_threshold
        T_TARGET_UPPER_LIST = [self.T_upper_bound] * self.PERIOD
        self.T_upper_bound_history = T_TARGET_UPPER_LIST

        self.T_target_history = np.array(self.T_target_history)
        self.T_lower_bound_history = np.array(self.T_lower_bound_history)
        self.T_upper_bound_history = np.array(self.T_upper_bound_history)

    def simulate_M_target(self):
        # M
        M_TARGET_LIST = [self.M_TARGET] * self.PERIOD
        self.M_target_history = M_TARGET_LIST

        # Lower bound
        self.M_lower_bound = self.M_TARGET - self.sM_threshold
        M_TARGET_LOWER_LIST = [self.M_lower_bound] * self.PERIOD
        self.M_lower_bound_history = M_TARGET_LOWER_LIST

        # Upper bound
        self.M_upper_bound = self.M_TARGET + self.sM_threshold
        M_TARGET_UPPER_LIST = [self.M_upper_bound] * self.PERIOD
        self.M_upper_bound_history = M_TARGET_UPPER_LIST

        self.M_target_history = np.array(self.M_target_history)
        self.M_lower_bound_history = np.array(self.M_lower_bound_history)
        self.M_upper_bound_history = np.array(self.M_upper_bound_history)

    def simulate_V_target(self):
        self.V_target_history = np.array([self.V_from_M(M) for M in self.M_target_history])

    def V_from_M(self, M):
        """
        Get Volume from Moisture
        :param M: moisture
        """
        return (self.Vs * M) / (100 - M)

    def write_data(self):
        oT_history = calculate_oscillations(self.T_history)
        oM_history = calculate_oscillations(self.M_history)
        sT_history = calculate_steady_state_error(self.T_history, self.T_target_history)
        sM_history = calculate_steady_state_error(self.M_history, self.M_target_history)

        data_dict = {"oT_history": oT_history,
                     "oM_history": oM_history,
                     "sT_history": sT_history,
                     "sM_history": sM_history,
                     "oT_threshold": self.oT_threshold,
                     "oM_threshold": self.oM_threshold,
                     "sT_threshold": self.sT_threshold,
                     "sM_threshold": self.sM_threshold}
        with open('data/data_' + self.filename + '.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

    def check_requirements(self):
        """
        Call monitoring routine to check requirements in quantitative way.
        Output written to file.
        """
        monitor(filename=self.filename)

    def load_requirements_results(self):
        """
        Load requirements results from file
        """
        return load_results(filename=self.filename)
