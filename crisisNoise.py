import numpy as np


class crisisNoise:
    def __init__(self, intensity_sigma: float, intensity_mu: float, action_size: int, duration: int = 1000):
        self.intensity_sigma = intensity_sigma
        self.intensity_mu = intensity_mu
        self.action_size = action_size
        self.duration = duration
        self.t = 0
        self.reset()

    def noise(self) -> np.ndarray:
        x = self.intensity_mu + self.intensity_sigma * np.random.randn(self.action_size) * np.exp(-self.t / self.duration)
        self.t += 1
        return x

    def reset(self):
        self.t = 0