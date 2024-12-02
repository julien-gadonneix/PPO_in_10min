from typing import Dict
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from GAE import GAE
from losses import ClippedPPOLoss, ClippedValueFunctionLoss
from worker import Worker
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class Trainer:
    """
    The Trainer class manages the training process for a reinforcement learning model using PPO.
    It handles the configuration of training parameters, initialization of worker processes, and the training loop.

    Parameters:
    -----------
    updates : int
        The number of updates to perform during training.
    epochs : int
        The number of epochs to train the model with sampled data.
    N : int
        The number of worker processes.
    T : int
        The number of steps to run on each process for a single update.
    batches : int
        The number of mini-batches.
    value_loss_coef : float
        The coefficient for the value loss in the overall loss function.
    entropy_bonus_coef : float
        The coefficient for the entropy bonus in the overall loss function.
    clip_range : float
        The clipping range for the PPO loss function.
    learning_rate : float
        The learning rate for the optimizer.
    model : nn.Module
        The neural network model to be trained.
    device : str
        The device on which to perform the training.
    """

    def __init__(self, *, updates: int, epochs: int, N: int, T: int, batches: int, value_loss_coef: float, entropy_bonus_coef: float, clip_range: float,
                 learning_rate: float, model: nn.Module, device: str):
        self.updates = updates
        self.epochs = epochs
        self.N = N
        self.T = T
        self.batches = batches
        self.batch_size = self.N * self.T
        self.mini_batch_size = self.batch_size // self.batches
        assert (self.batch_size % self.batches == 0)

        self.value_loss_coef = value_loss_coef
        self.entropy_bonus_coef = entropy_bonus_coef

        self.clip_range = clip_range
        self.learning_rate = learning_rate

        # Create workers
        self.workers = [Worker(47+i) for i in range(self.N)]

        # Initialize tensors for observations
        # self.action_size = 2 # for CartPole
        self.action_size = 17 # for Humanoid
        # self.state_size = 4 # for CartPole
        self.state_size = 348 # for Humanoid
        self.obs = np.zeros((self.N, self.state_size), dtype=np.float32)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()[0]

        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)
        gamma = 0.99
        lambda_ = 0.95
        self.gae = GAE(self.N, self.T, gamma, lambda_, self.action_size)
        self.ppo_loss = ClippedPPOLoss()
        self.value_loss = ClippedValueFunctionLoss()

        # Monitoring
        self.policy_loss_list = []
        self.value_loss_list = []
        self.entropy_bonus_list = []
        self.kl_divergence_list = []
        self.clip_fraction_list = []

        self.episode_length = []
        self.episode_reward = []


    def sample(self) -> Dict[str, torch.Tensor]:
        """
        This method collects data from the environment using the current policy. It interacts with the worker processes to gather observations, actions, rewards,
        and other relevant information over a specified number of time steps. The collected data is then used to compute advantages using GAE.

        Returns:
        --------
        samples_flat : Dict[str, torch.Tensor]
            A dictionary containing the flattened samples of observations, actions, values, log probabilities, and advantages, ready for training.
        """

        rewards = np.zeros((self.N, self.T), dtype=np.float32)
        # actions = np.zeros((self.N, self.T), dtype=np.int32) # for CartPole
        actions = np.zeros((self.N, self.T, self.action_size), dtype=np.int32) # for Humanoid
        done = np.zeros((self.N, self.T), dtype=bool)
        obs = np.zeros((self.N, self.T, self.state_size), dtype=np.float32)
        # log_pis = np.zeros((self.N, self.T), dtype=np.float32) # for CartPole
        log_pis = np.zeros((self.N, self.T, self.action_size), dtype=np.float32) # for Humanoid
        values = np.zeros((self.N, self.T + 1), dtype=np.float32)

        # Reset workers; sometimes needed TODO
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()[0]

        with torch.no_grad():
            # Sample T from each worker
            for t in range(self.T):
                # self.obs keeps track of the last observation from each worker, which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # Sample actions from pi_{theta_{OLD}} for each worker; this returns arrays of size N
                pi, v = self.model(torch.tensor(self.obs, dtype=torch.float32, device=self.device))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

                # Run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w, t]))

                for w, worker in enumerate(self.workers):
                    # Get results after executing the actions
                    self.obs[w], rewards[w, t], done[w, t], _, _ = worker.child.recv()
                    if done[w, t] and not done[w, t - 1]:
                        self.episode_length.append(t)
                        self.episode_reward.append(rewards[w, :t].sum())
                    elif t == self.T - 1 and not done[w, t]:
                        self.episode_length.append(self.T)
                        self.episode_reward.append(rewards[w].sum())

            # Get value of after the final step
            _, v = self.model(torch.tensor(self.obs, dtype=torch.float32, device=self.device))
            values[:, self.T] = v.cpu().numpy()

        # Calculate advantages
        advantages = self.gae(done, rewards, values)

        samples = {
            'obs': obs,
            'actions': actions,
            'values': values[:, :-1],
            'log_pis': log_pis,
            'advantages': advantages
        }

        # Camples are currently in [N, T] table, we should flatten it for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            samples_flat[k] = torch.tensor(v, device=self.device)

        return samples_flat
    

    def train(self, samples: Dict[str, torch.Tensor]):
        """
        This method trains the model using the provided samples. It performs multiple epochs of training, shuffling the data for each epoch and dividing it into
        mini-batches. For each mini-batch, it calculates the loss, updates the learning rate, computes the gradients, clips them to prevent exploding gradients,
        and updates the model parameters.

        Parameters:
        -----------
        samples : Dict[str, torch.Tensor]
            A dictionary containing the samples of observations, actions, values, log probabilities, and advantages, used for training the model.
        """

        # It learns faster with a higher number of epochs, but becomes a little unstable; that is, the average episode reward does not monotonically increase
        # over time. May be reducing the clipping range might solve it.
        for _ in range(self.epochs):
            # Shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # For each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # Get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                loss = self._calc_loss(mini_batch)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()


    @staticmethod
    def _normalize(adv: torch.Tensor):
        """
        This static method normalizes the advantage values by subtracting the mean and dividing by the standard deviation.
        Normalization helps in stabilizing the training process by ensuring that the advantage values have a mean of zero and a standard deviation of one.

        Parameters:
        -----------
        adv : torch.Tensor
            The advantage values to be normalized.

        Returns:
        --------
        torch.Tensor
            The normalized advantage values.
        """

        return (adv - adv.mean()) / (adv.std() + 1e-8)


    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This method calculates the total loss for training the model based on the provided samples. It computes the policy loss, value function loss, and entropy
        bonus, and combines them to form the total loss. Additionally, it normalizes the advantages.

        Parameters:
        -----------
        samples : Dict[str, torch.Tensor]
            A dictionary containing the samples of observations, actions, values, log probabilities, and advantages, used for calculating the loss.

        Returns:
        --------
        loss : torch.Tensor
            The total loss value used for updating the model parameters.
        """

        # R_t returns sampled from pi_{theta_{OLD}}
        # sampled_return = samples['values'] + samples['advantages'] # for CartPole
        sampled_return = samples['values'] + samples['advantages'][:, 0] # for Humanoid


        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get pi_theta(a_t|s_t) and V^{pi_theta}(s_t); we are treating observations as state
        pi, value = self.model(samples['obs'])

        # log pi_theta(a_t|s_t), a_t are actions sampled from pi_{theta_{OLD}}
        log_pi = pi.log_prob(samples['actions'])

        # Calculate policy loss
        policy_loss, clipped_fraction = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range)

        # Calculate Entropy Bonus
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate value function loss
        value_loss = self.value_loss(value, samples['values'], sampled_return, self.clip_range)

        loss = (policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_bonus_coef * entropy_bonus)

        # For monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        self.policy_loss_list.append(-policy_loss.item())
        self.value_loss_list.append(value_loss.item())
        self.entropy_bonus_list.append(entropy_bonus.item())
        self.kl_divergence_list.append(approx_kl_divergence.item())
        self.clip_fraction_list.append(clipped_fraction.item())

        return loss
    

    def run_training_loop(self):
        """
        This method runs the training loop for the model. It samples data using the current policy and trains the model using the sampled data.
        The training loop continues for a specified number of updates.
        """

        for _ in tqdm(range(self.updates)):
            # Sample with current policy
            samples = self.sample()

            # Train the model
            self.train(samples)


    def destroy(self):
        """
        This method stops all the worker processes. It sends a "close" command to each worker to terminate their execution.
        """
        
        for worker in self.workers:
            worker.child.send(("close", None))


    def plot(self):
        """
        This method plots the training metrics such as policy loss, value loss, entropy bonus, KL divergence, and clip fraction.
        """

        _, axs = plt.subplots(2, 3, figsize=(10, 5))
        axs[0, 0].plot(self.policy_loss_list)
        axs[0, 0].set_title("Policy Loss")
        axs[0, 0].set_xlabel("Updates")
        axs[0, 0].set_ylabel("Loss")

        axs[0, 1].plot(self.value_loss_list)
        axs[0, 1].set_title("Value Loss")
        axs[0, 1].set_xlabel("Updates")
        axs[0, 1].set_ylabel("Loss")

        axs[0, 2].plot(self.entropy_bonus_list)
        axs[0, 2].set_title("Entropy Bonus")
        axs[0, 2].set_xlabel("Updates")
        axs[0, 2].set_ylabel("Loss")

        axs[1, 0].plot(self.kl_divergence_list)
        axs[1, 0].set_title("KL Divergence")
        axs[1, 0].set_xlabel("Updates")
        axs[1, 0].set_ylabel("Loss")

        axs[1, 1].plot(self.clip_fraction_list)
        axs[1, 1].set_title("Clip Fraction")
        axs[1, 1].set_xlabel("Updates")
        axs[1, 1].set_ylabel("Fraction")

        plt.tight_layout()
        plt.savefig("results/training_metrics.png")

        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(self.episode_length)
        axs[0].set_title("Episode Length")
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Length")

        axs[1].plot(self.episode_reward)
        axs[1].set_title("Episode Reward")
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Reward")

        plt.tight_layout()
        plt.savefig("results/training_episodes.png")

    
    def log_video(self):
        """
        This method logs a video of the trained model's performance in the environment.
        """

        # env = gym.make("CartPole-v1", render_mode="rgb_array") # for CartPole
        env = gym.make("Humanoid-v5", render_mode="rgb_array") # for Humanoid

        video_file ="results/learned_dynamics.mp4"
        video_recorder = VideoRecorder(env, video_file, enabled=True)

        obs, _ = env.reset()
        done = False
        state = torch.tensor(obs, dtype=torch.float32, device=self.device)
        while not done:
            video_recorder.capture_frame()
            # Sample an action
            with torch.no_grad():
                pi, _ = self.model(state)
                action = pi.sample()
            # Step the environment
            obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            done = terminated or truncated

        video_recorder.capture_frame()
        video_recorder.close()
        video_recorder.enabled = False
        env.close()