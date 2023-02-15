import numpy as np

class MPC:
    def __init__(self):
        self.horizon = 3
        self.R = 1

    def cost(self, V_target, T_target, V, T):
        for t in range(self.horizon):

        cost = np.sqrt((V_target - V)**2 + (T_target - T)**2) + self.R * [(u_F_previous - u_F)**2 + (u_F_previous - u_F)**2]
        return cost
