import numpy as np


class GAE:
    """
    The GAE (Generalized Advantage Estimation) class computes the advantage estimates for a sequence of states, actions, and rewards.
    GAE helps in reducing the variance of the advantage estimates, leading to more stable training.

    Parameters:
    -----------
    N : int
        The number of parallel actors.
    T : int
        The number of time steps.
    gamma : float
        The discount factor for future rewards.
    lambda_ : float
        The GAE parameter that controls the trade-off between bias and variance in the advantage estimates.
    """

    def __init__(self, N: int, T: int, gamma: float, lambda_: float):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.T = T
        self.N = N


    def __call__(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Computes the advantage estimates for the given inputs.

        Parameters:
        -----------
        done : np.ndarray, shape=[N, T]
            A binary array indicating whether each actor is done at each time step.
        rewards : np.ndarray, shape=[N, T]
            The rewards obtained at each time step for each actor.
        values : np.ndarray, shape=[N, T]
            The estimated values of the states at each time step for each actor.

        Returns:
        --------
        advantages : np.ndarray, shape=[N, T]
            The computed advantage estimates for each time step and actor.
        """
        
        # Advantages table
        advantages = np.zeros((self.N, self.T), dtype=np.float32)
        last_advantage = 0

        # V(s_{t+1})
        last_value = values[:, -1]

        for t in reversed(range(self.T)):
            # Mask if episode completed after step t
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            # delta_t (TD error at time t)
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            # A_t = delta_t + gamma * lambda * A_{t+1}
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage

            advantages[:, t] = last_advantage

            last_value = values[:, t]

        # A_t
        return advantages
    
