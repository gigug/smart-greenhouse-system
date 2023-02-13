import pdb

import numpy as np


class PID:

    def __init__(self, flag):
        self.KP = 0
        self.TI = 0
        self.TD = 0
        self.reset(flag)

        self.e_previous = 0
        self.e_integral = 0

    def reset(self, flag):
        if flag == "V":
            self.KP = 1.2
            self.TI = 1.1
        if flag == "T":
            self.KP = 1.2
            self.TI = 1.1

    def control(self, target, current):
        e = target - current
        e_derivative = e - self.e_previous
        self.e_previous = e
        self.e_integral += e
        u = self.KP * (e + 1 / self.TI * self.e_integral + self.TD * e_derivative)
        return u
