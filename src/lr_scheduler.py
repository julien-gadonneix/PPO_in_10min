import torch


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    """
    This scheduler adjusts the learning rate of each parameter group using an exponential decay, but ensures that the learning rate does not fall below a
    specified minimum value.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scheduled.
    lr_decay : float
        The decay factor for the learning rate.
    min_lr : float, optional
        The minimum learning rate allowed. Default is 1e-6.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, min_lr: float = 1e-6):
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)


    def get_lr(self):
        """
        Compute the learning rate using the chainable form of the scheduler.

        Returns:
        --------
        List[float]
            A list of learning rates for each parameter group, ensuring that none fall below the minimum learning rate.
        """
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]