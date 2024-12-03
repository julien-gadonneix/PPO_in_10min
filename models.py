import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class NN_CartPole(nn.Module):
    """
    Model optimzed for CartPole-v1 environment.
    """

    def __init__(self, state_size: int, action_size: int, str_env: str):
        super().__init__()

        self.str_env = str_env

        self.lin1 = nn.Linear(in_features=state_size, out_features=512)
        self.lin2 = nn.Linear(in_features=512, out_features=1024)
        self.lin3 = nn.Linear(in_features=1024, out_features=2048)
        self.lin4 = nn.Linear(in_features=2048, out_features=512)

        # A fully connected layer to get logits for pi
        if str_env == "CartPole-v1":
            self.pi_logits = nn.Linear(in_features=512, out_features=action_size)
        else:
            self.pi_loc = nn.Linear(in_features=512, out_features=action_size)
            self.pi_scale = nn.Parameter(torch.zeros(action_size))

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

        self.activation = nn.ReLU()
        
    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin1(obs))
        h = self.activation(self.lin2(h))
        h = self.activation(self.lin3(h))
        h = self.activation(self.lin4(h))

        if self.str_env == "CartPole-v1":
            pi = Categorical(logits=self.pi_logits(h))
        else:
            mu = self.pi_loc(h)
            sigma = torch.exp(self.pi_scale).expand_as(mu)
            pi = Normal(loc=mu, scale=sigma)
        value = self.value(h).reshape(-1)

        return pi, value


class NN_Humanoid(nn.Module):
    """
    Model optimzed for Humanoid-v5 environment.
    """

    def __init__(self, state_size: int, action_size: int, str_env: str):
        super().__init__()

        self.str_env = str_env

        self.lin1 = nn.Linear(in_features=state_size, out_features=4096)
        self.lin2 = nn.Linear(in_features=4096, out_features=2048)
        self.lin3 = nn.Linear(in_features=2048, out_features=1024)
        self.lin4 = nn.Linear(in_features=1024, out_features=512)
        self.lin5 = nn.Linear(in_features=512, out_features=256)

        # A fully connected layer to get logits for pi
        if str_env == "CartPole-v1":
            self.pi_logits = nn.Linear(in_features=256, out_features=action_size)
        else:
            self.pi_loc = nn.Linear(in_features=256, out_features=action_size)
            self.pi_scale = nn.Parameter(torch.zeros(action_size))

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=256, out_features=1)

        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin1(obs))
        h = self.activation(self.lin2(h))
        h = self.activation(self.lin3(h))
        h = self.activation(self.lin4(h))
        h = self.activation(self.lin5(h))

        if self.str_env == "CartPole-v1":
            pi = Categorical(logits=self.pi_logits(h))
        else:
            mu = self.pi_loc(h)
            sigma = torch.exp(self.pi_scale).expand_as(mu)
            pi = Normal(loc=mu, scale=sigma)
        value = self.value(h).reshape(-1)

        return pi, value