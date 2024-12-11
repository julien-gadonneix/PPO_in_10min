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
        x = self.intensity_mu + self.intensity_sigma * np.random.randn(self.action_size)
        self.t += 1
        return x * np.random.choice([-1, 1], size=self.action_size, p=[0.5, 0.5])

    def reset(self):
        self.t = 0