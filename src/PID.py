import pdb

class PID:

    def __init__(self, flag):
        self.KP = 0
        self.TI = 0
        self.TD = 0
        self.set(flag)

        self.e_previous = 0
        self.e_integral = 0
        self.e_derivative = 0

        self.t = 0

    def set(self, flag):
        if flag == "V":
            self.KP = 2
            self.TI = None
            self.TD = None
        if flag == "T":
            self.KP = 50
            self.TI = 1
            self.TD = None

    def reset(self):
        self.e_integral = 0

    def control(self, target, current):
        if self.t % 60 * 60 == 0:
            self.reset()

        self.t += 1

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
