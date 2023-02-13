import pdb

class PID:

    def __init__(self, flag):
        self.KP = 0
        self.TI = 0
        self.TD = 0
        self.reset(flag)

        self.e_previous = 0
        self.e_integral = 0
        self.e_derivative = 0

    def reset(self, flag):
        if flag == "V":
            self.KP = 2
            self.TI = None
            self.TD = None
        if flag == "T":
            self.KP = 1.0e04
            self.TI = 200
            self.TD = None

    def control(self, target, current):
        e = target - current
        self.e_derivative = e - self.e_previous
        self.e_previous = e
        self.e_integral += e

        P = self.KP * e
        I = 0
        D = 0
        if self.TI is not None:
            I = self.KP * 1 / self.TI * self.e_integral
        if self.TD is not None:
            D = self.KP * self.TD * self.e_derivative
        u = P + I + D
        return u
