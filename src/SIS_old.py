import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random

np.random.seed(0)


def save_timesteps_temperatures(t_start, t_end):
    """
    Function to pre-calculate time and temperatures
    """
    timesteps = np.arange(t_start, t_end)
    temperatures = simulate_temperature(timesteps)

    with open("../times.pickle", "wb") as f:
        pkl.dump(timesteps, f)

    with open("../temperatures.pickle", "wb") as f:
        pkl.dump(temperatures, f)


def load_timesteps_temperatures():
    """
    Function to load the pre-calculated times and temperatures.
    """
    with open("../times.pickle", "rb") as f:
        timesteps = pkl.load(f)

    with open("../temperatures.pickle", "rb") as f:
        temperatures = pkl.load(f)

    return timesteps, temperatures


def simulate_temperature(timesteps):
    """
    Function to simulate temperature over a given time interval.
    """
    temperatures = [temperature(t) for t in timesteps]
    return temperatures


def simulate_timesteps(t_start, t_end):
    """
    Function to simulate timesteps over a given time interval.
    """
    timesteps = np.arange(t_start, t_end)
    return timesteps


def temperature(t):
    """
    Function to simulate temperature as a function of time.
    """
    T_1 = A_1 * np.sin(2 * np.pi * t / P_1)
    T_2 = A_2 * np.sin(2 * np.pi * t / P_2 + T_M)
    T_3 = T_BAR
    T_4 = np.random.normal(0, 1)
    T = T_1 + T_2 + T_3 + T_4
    return T


def simulate_rain(timesteps):
    """
    Function to simulate rain.
    """
    t_min = 10
    t_max = 30
    rain = np.zeros(len(timesteps))
    t = 0
    while t < len(timesteps):
        if random.random() < 0.05:
            rain_duration = random.randint(t_min, t_max)
            final_t = t + rain_duration
            if final_t < len(timesteps):
                rain[t:final_t] = [1] * rain_duration
                t = final_t
        else:
            t += 1
    return rain


def simulate_evapotranspiration(temperatures):
    """
    Function to simulate evapotranspiration.
    """
    e = [evapotranspiration(T) for T in temperatures]
    return e


def evapotranspiration(T):
    """
    Function that calculates evapotranspiration at temperature T
    """
    return epsilon_1 * T + epsilon_2


def calculate_volume_water(volume_water_previous, active, et_t, flux_minute, area):
    """
    Function to calculate volume of water given every parameter.
    """
    volume_water = volume_water_previous + - et_t * area

    if active:
        volume_water += flux_minute

    # check that volume of water is always at least 0
    volume_water = volume_water if volume_water > 0 else 0

    return volume_water


def calculate_moisture(timesteps, volume_water, volume_soil):
    """
    Function to calculate moisture quantities at every timestep.
    """
    moistures = [moisture(volume_water[t], volume_soil) for t in timesteps]

    return moistures


def moisture(volume_water, volume_soil):
    """
    Function to calculate moisture at timestep t.
    """
    return (volume_water / (volume_water + volume_soil)) * 100


def activate_shader(rain, temperature, temperature_shade):
    """
    Function to check whether to activate shader.
    """
    shader = True if temperature > temperature_shade else False
    return shader


def activate_sprinkler(shade, previous_moisture, max_moisture):
    """
    Function to check whether to activate sprinkler.
    """
    sprinkler = True if not shade and previous_moisture <= max_moisture else False
    return sprinkler


class SmartIrrigationSystem:

    def __init__(self,
                 temperature_shade,
                 moisture_min,
                 moisture_max,
                 initial_volume_water,
                 flux_minute,
                 area,
                 volume_soil):
        # set constants
        self.temperature_shade = temperature_shade
        self.moisture_min = moisture_min
        self.moisture_max = moisture_max
        self.initial_volume_water = initial_volume_water
        self.flux_minute = flux_minute
        self.area = area
        self.volume_soil = volume_soil

        self.timesteps = simulate_timesteps(t_start, t_end)
        self.temperatures = simulate_temperature(self.timesteps)
        self.rain = simulate_rain(self.timesteps)
        self.volume_water = []
        self.moistures = []
        self.shade = []
        self.active = []

    def activate(self):
        """
        Function that turns on the Smart Irrigation System.
        """
        self.volume_water = [self.initial_volume_water]
        self.moistures = [self.moisture_min]
        self.shade = [True if self.temperatures[0] > self.temperature_shade else False]
        self.active = [False if self.shade[0] else True]

        for t in self.timesteps[1:]:
            temperature_t = self.temperatures[t]
            rain_t = self.rain[t]
            shade_t = activate_shader(rain_t, temperature_t, self.temperature_shade)
            active_t = activate_sprinkler(shade_t, self.moistures[t-1], self.moisture_max)

            if shade_t or rain_t:
                temperature_t -= 10

            et_t = evapotranspiration(temperature_t)
            volume_water_t = calculate_volume_water(self.volume_water[t - 1], active_t, et_t, self.flux_minute, self.area)
            moisture_t = moisture(volume_water_t, self.volume_soil)

            self.temperatures.append(temperature_t)
            self.volume_water.append(volume_water_t)
            self.moistures.append(moisture_t)
            self.shade.append(shade_t)
            self.active.append(active_t)
    def plot_quantity(self, quantity):
        """
        Function to plot chosen quantity.
        """

        if quantity == "Moisture":
            y = self.moistures
            y_label = "VWC"
        if quantity == "Temperatures":
            y = self.temperatures
            y_label = "Kelvin"
        if quantity == "Volume water":
            y = self.volume_water
            y_label = "Liters"

        fig, ax = plt.subplots()

        x_ticks = np.arange(0, max(self.timesteps) + 1, 60 * 24)
        x_labels = [str(i // (60 * 24) + 1) for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        ax.set_ylabel(y_label)
        ax.set_xlabel("Days")

        plt.plot(self.timesteps, y)

        if quantity == "Moisture":
            plt.axhline(y=moisture_min, color='red', linestyle='--')
            plt.axhline(y=moisture_max, color='red', linestyle='--')

        plt.show()

    def ODE(self, x, u):
        """
        Define ODEs of the system.
        """
        V, T = x
        Fs, P, Te = u

        dVdt = Fs - self.evapotranspiration(T)
        dTdt = self.conduction(T, Te) + self.radiation(T, Te) + self.water_conduction(T, V) + self.air_conditioner(P)

        return dVdt, dTdt

    def step(self, x, u):
        """
        Do single discrete step.
        """
        dVdt, dTdt = self.ODE(x, u)
        V, T = x
        V += dVdt
        T += dTdt

        return V, T

    def conduction(self, T, Te):
        pass

    def radiation(self, T, Te):
        pass

    def water_conduction(self, T, V):
        pass

    def air_conditioner(self, P):
        pass

    def evapotranspiration(self, T):
        return self.alpha0 + self.alpha1 * T


if __name__ == '__main__':
    K_0 = 273.15

    t_day = 60 * 24
    t_start = 0
    t_end = t_day * 30

    A_1 = 5
    A_2 = 15
    T_BAR = 20 + K_0
    P_1 = t_day
    P_2 = t_day * 365
    T_M = t_day * 30 * 6  # indicates month

    epsilon_1 = 0.0022
    epsilon_2 = 0

    moisture_min = 60
    moisture_max = 80

    area = 25
    depth = 0.5
    volume_soil = area * depth * 1000  # convert to liters

    initial_volume_water = (volume_soil * moisture_min) / (100 - moisture_min)

    flux_hour = 1000
    flux_minute = flux_hour / 60

    temperature_shade = 35 + K_0

    SIS = SmartIrrigationSystem(temperature_shade=temperature_shade,
                                moisture_min=moisture_min,
                                moisture_max=moisture_max,
                                initial_volume_water=initial_volume_water,
                                flux_minute=flux_minute,
                                area=area,
                                volume_soil=volume_soil)

    SIS.activate()

    SIS.plot_quantity("Moisture")
