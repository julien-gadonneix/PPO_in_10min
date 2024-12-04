import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class NN_CartPole(nn.Module):
    """
    Model optimzed for CartPole-v1 environment.
    """

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(in_features=4, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=128)
        self.lin3 = nn.Linear(in_features=128, out_features=256)
        self.lin4 = nn.Linear(in_features=256, out_features=512)

        # A fully connected layer to get logits for pi
        self.pi_logits = nn.Linear(in_features=512, out_features=2)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

        self.activation = nn.ReLU()
        
    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin1(obs))
        h = self.activation(self.lin2(h))
        h = self.activation(self.lin3(h))
        h = self.activation(self.lin4(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value


class NN_Humanoid(nn.Module):
    """
    Model optimzed for Humanoid-v5 environment.
    """

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(in_features=348, out_features=4096)
        self.lin2 = nn.Linear(in_features=4096, out_features=2048)
        self.lin3 = nn.Linear(in_features=2048, out_features=1024)
        self.lin4 = nn.Linear(in_features=1024, out_features=512)

        # Fully connected layers to get logits for pi
        self.pi_loc = nn.Linear(in_features=512, out_features=17)
        self.pi_scale = nn.Parameter(torch.zeros(17))

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin1(obs))
        h = self.activation(self.lin2(h))
        h = self.activation(self.lin3(h))
        h = self.activation(self.lin4(h))

        mu = self.pi_loc(h)
        sigma = torch.exp(self.pi_scale).expand_as(mu)
        pi = Normal(loc=mu, scale=sigma)
        value = self.value(h).reshape(-1)

        return pi, value
    

class NN_Hopper(nn.Module):
    """
    Model optimzed for Hopper-v5 environment.
    """

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(in_features=11, out_features=1024)
        self.lin2 = nn.Linear(in_features=1024, out_features=512)
        self.lin3 = nn.Linear(in_features=512, out_features=256)
        self.lin4 = nn.Linear(in_features=256, out_features=512)
        self.lin5 = nn.Linear(in_features=512, out_features=256)

        # Fully connected layers to get logits for pi
        self.pi_loc = nn.Linear(in_features=256, out_features=3)
        self.pi_scale = nn.Parameter(torch.zeros(3))

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=256, out_features=1)

        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin1(obs))
        h = self.activation(self.lin2(h))
        h = self.activation(self.lin3(h))
        h = self.activation(self.lin4(h))
        h = self.activation(self.lin5(h))

        mu = self.pi_loc(h)
        sigma = torch.exp(self.pi_scale).expand_as(mu)
        pi = Normal(loc=mu, scale=sigma)
        value = self.value(h).reshape(-1)

        return pi, value