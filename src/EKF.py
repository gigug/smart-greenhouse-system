class EKF:
    """Extended Kalman Filter"""

    def __init__(self, model, R, Q, P):
        self.model = model
        self.A = self.get_matrices()[0]
        self.B = self.get_matrices()[1]
        self.Q_k = Q
        self.R_k = R
        self.P_k_1 = P