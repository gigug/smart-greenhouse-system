import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def save_timesteps_temperatures(t_start, t_end):
    """
    Function to pre-calculate time and temperatures
    """
    timesteps = np.arange(t_start, t_end)
    temperatures = simulate_temperature(timesteps)

    with open("times.pickle", "wb") as f:
        pkl.dump(timesteps, f)

    with open("temperatures.pickle", "wb") as f:
        pkl.dump(temperatures, f)


def load_timesteps_temperatures():
    """
    Function to load the pre-calculated times and temperatures.
    """
    with open("times.pickle", "rb") as f:
        timesteps = pkl.load(f)

    with open("temperatures.pickle", "rb") as f:
        temperatures = pkl.load(f)

    return timesteps, temperatures


def plot_quantity(timesteps, quantity, y_label, moisture_min, moisture_max, shade, active):
    """
    Function to plot chosen quantity.
    """
    fig, ax = plt.subplots()

    x_ticks = np.arange(0, max(timesteps) + 1, 60 * 24)
    x_labels = [str(i // (60 * 24) + 1) for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    ax.set_ylabel(y_label)
    ax.set_xlabel("Days")

    plt.plot(timesteps, quantity)

    if y_label == "VWC":
        plt.axhline(y=moisture_min, color='red', linestyle='--')
        plt.axhline(y=moisture_max, color='red', linestyle='--')

    # colors = np.array(['red' if not act else 'green' for act in active])
    # ax.bar(timesteps, [100]*len(timesteps), color=colors, alpha=0.5)
    # plt.savefig('test.png', dpi=100)
    plt.show()


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
        self.shade = [True if temperatures[0] > self.temperature_shade else False]
        self.active = [False if shade[0] else True]

        for t in timesteps[1:]:
            # check if shading system needs to be activated
            shade_t = True if temperatures[t] > self.temperature_shade else False
            # check if sprinkler can be activated
            active_t = False if shade_t or moistures[t - 1] > self.temperature_shade else True

            temperature_t = temperatures[t] if not shade_t else temperatures[t] - 10
            et_t = evapotranspiration(temperature_t)
            volume_water_t = calculate_volume_water(volume_water[t - 1], active_t, et_t, flux_minute, area)
            moisture_t = moisture(volume_water_t, volume_soil)

            volume_water.append(volume_water_t)
            moistures.append(moisture_t)
            shade.append(shade_t)
            active.append(active_t)

        return volume_water, moistures, shade, active


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

    timesteps = simulate_timesteps(t_start, t_end)

    SIS = SmartIrrigationSystem(temperature_shade=temperature_shade,
                                moisture_min=moisture_min,
                                moisture_max=moisture_max,
                                initial_volume_water=initial_volume_water,
                                flux_minute=flux_minute,
                                area=area,
                                volume_soil=volume_soil)

    SIS.activate()

    plot_quantity(timesteps,
                  moistures,
                  "VWC",
                  moisture_min,
                  moisture_max,
                  shade,
                  active)
