# PID controller class
class pid:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.error = 0
        self.error_prev = 0
        self.error_sum = 0
        self.output = 0

    def update(self, error):
        self.error = error
        self.error_sum += self.error
        self.output = self.kp * self.error + self.ki * self.error_sum * self.dt + self.kd * (self.error - self.error_prev) / self.dt
        self.error_prev = self.error
        return self.output
