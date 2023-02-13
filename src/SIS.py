import numpy as np
from matplotlib import pyplot as plt
import pdb

from src.PID import PID

class SIS:
    def __init__(self):
        self.K_0 = 273.15  # [K]

        self.ALPHA_1 = 2.0e-05  # [m^3]
        self.ALPHA_2 = 1.0e-07  # [m^3/K]

        self.HEIGHT = 3  # [m]
        self.SIDE = 5  # [m]
        self.AREA = self.SIDE ** 2  # [m^2]
        self.V_ROOM = self.AREA * self.HEIGHT  # [m^3]

        self.DEPTH = 0.5  # [m]
        self.Vs = self.AREA * self.DEPTH  # [m^3]

        self.RHO_A = 1.204  # [kg/m^3]
        self.LAMBDA_G = 1  # [W/(mK)] specific heat glass
        self.AREA_G = 4 * self.HEIGHT * self.SIDE + self.AREA  # [m^2] area glass
        self.Cp_A = 1000  # [J/(kgK)] specific heat air
        self.SIGMA = 5.67e-08  # [W/(m^2K^4)]
        self.EPSILON = 0.9  # [dimensionless]
        self.C_W = 4187  # [J/Kg]
        self.K = 1.08  # [dimensionless]
        self.BETA = self.V_ROOM * self.RHO_A * self.Cp_A  # [K/J]

        self.PERIOD = 60 * 60  # [s]

        self.M_target = 70  # [dimensionless]
        self.V_target = (self.Vs * self.M_target) / (100 - self.M_target)  # [m^3]
        self.T_target = self.K_0 + 20  # [K]

        self.T_e_0 = self.K_0 + 30  # [K]
        self.T_w_0 = self.K_0 + 10  # [K]
        self.T_0 = self.K_0 + 10  # [K]
        self.V_w_0 = self.Vs / 5  # [m^3]

        self.T_e = self.T_e_0
        self.T_w = self.T_w_0

        self.V = self.V_w_0
        self.T = self.T_0
        self.M = 0
        self.update_M()

        self.min_oscillation_T = 0.1
        self.min_oscillation_M = 1

        self.min_P = -100
        self.max_P = 100
        self.min_F = 0  # [m^3]
        self.max_F = 0.01  # [m^3]

        self.timesteps = np.asarray(range(self.PERIOD))
        self.V_history = np.zeros(self.PERIOD)
        self.T_history = np.zeros(self.PERIOD)
        self.M_history = np.zeros(self.PERIOD)

        self.rise_time = None
        self.overshoot = None
        self.settling_time = None
        self.steady_state_error = None

        self.PID_V = PID("V")
        self.PID_T = PID("T")
        self.PID_V.reset("V")
        self.PID_T.reset("T")

    def run(self):
        for t in range(self.PERIOD):
            V = self.get_V()
            T = self.get_T()

            u_F = self.PID_V.control(self.V_target, V)
            u_P = self.PID_T.control(self.T_target, T)

            u_F = self.check_F(u_F)
            u_P = self.check_P(u_P)

            x = (self.V, self.T)
            u = (u_F, u_P)

            self.step(x, t, u)

    def get_T(self):
        return self.T

    def get_V(self):
        return self.V

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
        return (1 / self.BETA) * (self.conduction(T) + self.water_conduction(V, T) + self.air_conditioner(P, T))

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
        #self.update_T(dTdt)
        self.update_M()

        self.V_history[t] = self.V
        self.T_history[t] = self.T
        self.M_history[t] = self.M

        return self.V, self.T

    def flux(self, F):
        return F

    def conduction(self, T):
        return self.LAMBDA_G * self.AREA_G * (self.T_e - T)

    def radiation(self, T):
        return self.SIGMA * self.EPSILON * (self.T_e ** 4 - T ** 4)

    def water_conduction(self, V, T):
        return self.C_W * V * (self.T_w - T)

    def air_conditioner(self, P, T):
        return self.K * P * (self.T_target - T)

    def evapotranspiration(self, T):
        return -(self.ALPHA_1 + self.ALPHA_2 * T)

    def check_F(self, u_F):
        if u_F > self.max_F:
            return self.max_F
        elif u_F < self.min_F:
            return self.min_F
        else:
            return u_F

    def check_P(self, u_P):
        if u_P > self.max_P:
            return self.max_P
        elif u_P < self.min_P:
            return self.min_P
        else:
            return u_P

    def measures(self, variable):
        if variable == "T":
            history = self.T_history
            target = self.T_target
            min_oscillation = self.min_oscillation_T
        if variable == "M":
            history = self.M_history
            target = self.M_target
            min_oscillation = self.min_oscillation_M
        if variable == "V":
            history = self.V_history
            target = self.V_target
            min_oscillation = self.min_oscillation_M

        self.get_rise_time(history, target)
        self.get_overshoot(history, target)
        self.get_settling_time(history, min_oscillation)
        self.get_steady_state_error(history, target)

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
        fig, ax = plt.subplots()

        if variable == "M":
            y = self.M_history
            y_label = "VWC"
            plt.axhline(y=self.M_target, color='red', linestyle='--')
        if variable == "T":
            y = self.T_history
            y_label = "K"
            plt.axhline(y=self.T_target, color='red', linestyle='--')
        if variable == "V":
            y = self.V_history
            y_label = "$m^3$"
            plt.axhline(y=self.V_target, color='red', linestyle='--')

        ax.set_ylabel(y_label)
        ax.set_xlabel("Seconds")

        plt.plot(self.timesteps, y)

        plt.show()

