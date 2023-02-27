import pickle

import numpy as np

from src.SGS import SGS
from src.monitor import load_results


def write_falsification_results(robustness_dict):
    """
    Write falsification results to file.
    :param robustness_dict: dictionary that identifies conditions and robustness.
    """
    with open('robustness/falsification_results.pickle', 'wb') as f:
        pickle.dump(robustness_dict, f)


def load_falsification_results():
    """
    Load falsification results from file.
    :param robustness_dict: dictionary that identifies conditions and robustness.
    """
    with open('robustness/falsification_results.pickle', 'rb') as f:
        robustness_dict = pickle.load(f)

    return robustness_dict


def falsification():
    T_BAR_LIST = range(255, 310, 5)
    AMPLITUDES = [0.5, 1, 5, 10]

    robustness_dict = {}
    for T_BAR in T_BAR_LIST:
        robustness_dict[T_BAR] = {}
        for AMPLITUDE in AMPLITUDES:
            filename = f'{T_BAR}_{AMPLITUDE}'

            min_robustness = np.infty

            for N in range(10):
                sgs = SGS(T_BAR=T_BAR,
                          AMPLITUDE=AMPLITUDE)
                sgs.run()
                sgs.check_requirements()

                result_dict = load_results(filename)
                robustness = result_dict["phi_4"]

                if robustness < min_robustness:
                    min_robustness = robustness

                if robustness < 0:
                    break

            print(f'T_BAR: {T_BAR}, AMPLITUDE: {AMPLITUDE}, ROBUSTNESS: {min_robustness}')

            robustness_dict[T_BAR][AMPLITUDE] = min_robustness

    write_falsification_results(robustness_dict)

