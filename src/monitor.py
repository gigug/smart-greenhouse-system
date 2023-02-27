import pdb
import pickle
import numpy as np

from src.moonlight import *


def write_results(filename, phi_1, phi_2, phi_3, phi_4, phi_5):
    """
    Write results to file.
    :param filename: portion of filename that identifies file.
    """
    results_dict = {"phi_1": phi_1[0][1],
                    "phi_2": phi_2[0][1],
                    "phi_3": phi_3[0][1],
                    "phi_4": phi_4[0][1],
                    "phi_5": phi_5[0][1]}

    with open('robustness/results_' + filename + '.pickle', 'wb') as f:
        pickle.dump(results_dict, f)


def load_results(filename):
    """
    Load results from file.
    :param filename: portion of filename that identifies file.
    """
    with open('robustness/results_' + filename + '.pickle', 'rb') as f:
        results_dict = pickle.load(f)

    return results_dict


def monitor(filename):
    """
    Load histories from file and check requirements.
    :param filename: portion of filename that identifies file.
    """
    with open('data/data_' + filename + '.pickle', 'rb') as f:
        data_dict = pickle.load(f)

    hour = 60 * 60 - 1
    hour_2 = 2 * 60 * 60 - 1

    oT_history = data_dict["oT_history"]
    oM_history = data_dict["oM_history"]
    sT_history = data_dict["sT_history"]
    sM_history = data_dict["sM_history"]
    oT_threshold = data_dict["oT_threshold"]
    oM_threshold = data_dict["oM_threshold"]
    sT_threshold = data_dict["sT_threshold"]
    sM_threshold = data_dict["sM_threshold"]

    oT_history = [[i] for i in oT_history]
    oM_history = [[i] for i in oM_history]
    sT_history = [[i] for i in sT_history]
    sM_history = [[i] for i in sM_history]

    timesteps = list(range(len(oT_history)))

    phi_1_2 = f"""
    signal {{real oT;}}
    domain minmax; 
    formula phi_1 = eventually {{ globally [0, {hour}] (oT < {oT_threshold}) }};
    formula phi_2 = globally [{hour}, {hour_2}] (oT < {oT_threshold/2});
    """

    phi_3 = f"""
    signal {{real oM;}}
    domain minmax; 
    formula phi_3 = globally [0, {hour_2}] (oM < {oM_threshold});
    """

    phi_4 = f"""
    signal {{real sT;}}
    domain minmax; 
    formula phi_4 = globally [{hour}, {hour_2}] (sT < {sT_threshold});
    """

    phi_5 = f"""
    signal {{real sM;}}
    domain minmax; 
    formula phi_5 = globally [{hour}, {hour_2}] (sM < {sM_threshold});
    """

    phi_1_2_script = ScriptLoader.loadFromText(phi_1_2)
    phi_3_script = ScriptLoader.loadFromText(phi_3)
    phi_4_script = ScriptLoader.loadFromText(phi_4)
    phi_5_script = ScriptLoader.loadFromText(phi_5)

    phi_1_monitor = phi_1_2_script.getMonitor("phi_1")
    phi_2_monitor = phi_1_2_script.getMonitor("phi_2")
    phi_3_monitor = phi_3_script.getMonitor("phi_3")
    phi_4_monitor = phi_4_script.getMonitor("phi_4")
    phi_5_monitor = phi_5_script.getMonitor("phi_5")

    phi_1_monitor_result = np.array(phi_1_monitor.monitor(timesteps, oT_history))
    phi_2_monitor_result = np.array(phi_2_monitor.monitor(timesteps, oT_history))
    phi_3_monitor_result = np.array(phi_3_monitor.monitor(timesteps, oM_history))
    phi_4_monitor_result = np.array(phi_4_monitor.monitor(timesteps, sT_history))
    phi_5_monitor_result = np.array(phi_5_monitor.monitor(timesteps, sM_history))

    write_results(filename,
                  phi_1_monitor_result,
                  phi_2_monitor_result,
                  phi_3_monitor_result,
                  phi_4_monitor_result,
                  phi_5_monitor_result)
