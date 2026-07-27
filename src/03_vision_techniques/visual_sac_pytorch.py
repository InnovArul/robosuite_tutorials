import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal

import copy

# import spatial softmax and augmentation modules from other files
from frame_stack_spatial_softmax import SpatialSoftmaxEncoder
from random_translation_shift import RandomShiftsAug

class SquashedGaussianActor(nn.Module):
    """
    Squashed Gaussian Actor for continuous action spaces.
    Outputs actions in the range [-1, 1] using tanh squashing.
    """
    def __init__(self, keypoint_dim, proprio_dim, action_dim, hidden_dim=256, log_std_bounds=(-10, 2)):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.keypoint_dim = keypoint_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

        in_dim = keypoint_dim + proprio_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)


    def forward(self, keypoints, proprio):
        x = torch.cat([keypoints, proprio], dim=-1)
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, *self.log_std_bounds)
        # deduce std from log_std
        std = torch.exp(log_std)

        # Create a Normal distribution
        dist = Normal(mean, std)
        u = dist.rsample()  # Reparameterization trick
        action = torch.tanh(u)  # Squash to [-1, 1]

        # deduce log probability of the action
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)

        # Apply correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob

class DoubleQCritic(nn.Module):
    """Dual Critic evaluate Q(z, a) to prevent overestimation bias."""
    def __init__(self, keypoint_dim, proprio_dim, action_dim, hidden_dim=256):
        super().__init__()
        in_dim = keypoint_dim + proprio_dim + action_dim

        self.Q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, proprio, action):
        x = torch.cat([z, proprio, action], dim=-1)
        return self.Q1(x), self.Q2(x)


class DrQv2SACAgent:
    """Unified DrQ-v2 Visual SAC Agent."""
    def __init__(self, action_dim, proprio_dim=32, device="cuda"):
        self.device = torch.device(device)
        self.aug = RandomShiftsAug(pad=4).to(self.device)

        # 1. Encoder & Target Encoder
        self.encoder = SpatialSoftmaxEncoder(input_channels=9, num_filters=32).to(self.device)
        self.encoder_target = copy.deepcopy(self.encoder)

        # 2. Actor & Critic Networks
        keypoint_dim = 32 * 2  # 32 channels * 2D coordinates = 64
        self.actor = SquashedGaussianActor(keypoint_dim, proprio_dim, action_dim).to(self.device)
        self.critic = DoubleQCritic(keypoint_dim, proprio_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # 3. Optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

    def update(self, batch, gamma=0.99, tau=0.01):
        """
        Update the SAC agent using a batch of transitions.
        Args:
            batch (dict): A batch of transitions containing:
                - "pixels": Tensor of shape (B, 9, 84, 84)
                - "proprio": Tensor of shape (B, proprio_dim)
                - "actions": Tensor of shape (B, action_dim)
                - "rewards": Tensor of shape (B, 1)
                - "next_pixels": Tensor of shape (B, 9, 84, 84)
                - "dones": Tensor of shape (B, 1)
            gamma (float): Discount factor.
            tau (float): Soft update coefficient for target networks.
        """
        pixels, proprio, actions, rewards, next_pixels, next_proprio, dones = [
                        b.to(self.device) for b in batch
        ]

        # --- A. Augment Observations ---
        aug_pixels = self.aug(pixels)
        aug_next_pixels = self.aug(next_pixels)

        # --- B. Update Critic & Encoder ---
        with torch.no_grad():
            next_z = self.encoder_target(aug_next_pixels)
            next_a, next_log_prob = self.actor(next_z, next_proprio)
            target_q1, target_q2 = self.critic_target(next_z, next_proprio, next_a)
            target_q = torch.min(target_q1, target_q2) - 0.1 * next_log_prob
            y = rewards + gamma * (1 - dones) * target_q

        z = self.encoder(aug_pixels)
        q1, q2 = self.critic(z, proprio, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        self.encoder_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # --- C. Update Actor (Freeze Encoder Gradient) ---
        z_no_grad = z.detach()  # Stop gradient flowing to visual encoder
        a, log_prob = self.actor(z_no_grad, proprio)
        q1_actor, q2_actor = self.critic(z_no_grad, proprio, a)
        actor_loss = (0.1 * log_prob - torch.min(q1_actor, q2_actor)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- D. Polyak Target Update ---
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.copy_(tau * p.data + (1 - tau) * p_targ.data)
        for p, p_targ in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            p_targ.data.copy_(tau * p.data + (1 - tau) * p_targ.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
