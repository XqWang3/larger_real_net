import numpy as np
import numpy.random as nr

class OrnsteinUhlenbeckNoise:
    """
        Temporal Noise Class
        modelling three factors (analogy to financial models):
            volatility
            reversion rate
            expected value

        Caveat:
            1. Implemented on numpy level, not TensorFlow or PyTorch level. Initialization params are supposed to be np.ndarray
            2. sigma is supposed to be diagonal (not covariance matrix), thus assuming independence between dimensions.
    """

    def __init__(self, mu, theta=0.2, sigma=0.2, x0=None):
        """
            params:
                mu: expected value
                theta: float, [0, 1]. reversion rate. 
                    0: Brownian noise
                    1: independent Gaussian noise
                sigma: volatility
                x0: initial level imposed
        """
        self.mu = np.copy(mu)
        self.theta = np.copy(theta)
        self.sigma = np.copy(sigma)
        self.x0 = np.copy(mu) if x0 is None else np.copy(x0)
        self.x = np.copy(self.x0)

    def reset(self):
        self.x = np.copy(self.x0)

    def __call__(self):
        self.x = self.x + self.theta * (self.mu - self.x) + self.sigma * np.random.normal(size=self.mu.shape)
        return self.x

    def __repr__(self):

        return "OrnsteinUhlenbeckNoise(mu={}, theta={}, sigma={}, x={}".format(self.mu, self.theta, self.sigma, self.x)

##################################################3



class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
        print(states[-1])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()