import torch
import torch.nn as nn


class ClippedPPOLoss(nn.Module):
    """
    The ClippedPPOLoss function computes the clipped surrogate objective for PPO algorithms.
    This objective helps in stabilizing the training of RL agents by limiting the change in the policy update.
    """

    def __init__(self):
        super().__init__()


    def forward(self, log_pi: torch.Tensor, sampled_log_pi: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
        """
        Loss calculations.

        Parameters:
        -----------
        log_pi : torch.Tensor, shape=[batch_size, action_size]
            The log probabilities of the actions taken under the current policy.
        sampled_log_pi : torch.Tensor, shape=[batch_size, action_size]
            The log probabilities of the actions taken under the old policy (used to collect the data).
        advantage : torch.Tensor, shape=[batch_size, 1]
            The advantage estimates, which measure the relative benefit of taking a particular action compared to the average action.
        clip : float
            The clipping parameter, which defines the range within which the ratio of the new policy to the old policy is clipped.
            This helps in preventing large policy updates.

        Returns:
        --------
        -policy_reward.mean() : torch.Tensor, shape=[]
            The negative mean of the clipped surrogate objective. This is the loss value that is minimized during training.
        clip_fraction : torch.Tensor, shape=[]
            The fraction of samples that are clipped, which provides an indication of how often the clipping mechanism is activated.
            This can be useful for monitoring the training process.
        """

        r_theta = torch.exp(log_pi - sampled_log_pi)
        clipped_r_theta = r_theta.clamp(min=1.0 - clip, max=1.0 + clip)
        policy_reward = torch.min(r_theta * advantage, clipped_r_theta * advantage)

        clip_fraction = (abs((r_theta - 1.0)) > clip).to(torch.float).mean()

        return -policy_reward.mean(), clip_fraction


class ClippedValueFunctionLoss(nn.Module):
    """
    The ClippedValueFunctionLoss function computes the clipped value function loss for value estimation.
    This loss helps in stabilizing the training of the value function by limiting the change in the value updates.
    """

    def forward(self, value: torch.Tensor, sampled_value: torch.Tensor, sampled_return: torch.Tensor, clip: float):
        """
        Loss calculations.

        Parameters:
        -----------
        value : torch.Tensor, shape=[batch_size]
            The predicted values from the current value function.
        sampled_value : torch.Tensor, shape=[batch_size]
            The predicted values from the old value function (the value function used to collect the data).
        sampled_return : torch.Tensor, shape=[batch_size]
            The actual returns (cumulative rewards) observed during the data collection.
        clip : float
            The clipping parameter, which defines the range within which the difference between the new value and the old value is clipped.
            This helps in preventing large value updates.

        Returns:
        --------
        0.5 * vf_loss.mean() : torch.Tensor, shape=[]
            The mean of the clipped value function loss, scaled by 0.5 to obtain the TD error.
            This is the loss value that is minimized during training.
        """
            
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)

        return 0.5 * vf_loss.mean()

