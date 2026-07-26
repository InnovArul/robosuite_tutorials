import collections
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobosuitePixelWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for Robosuite that handles frame-stacking (k=3),
    channel-first conversion (CHW), OpenGL inversion, and proprioception integration.
    """
    def __init__(self, env, num_stack=3, camera_name="agentview_image"):
        super().__init__(env)
        self.num_stack = num_stack
        self.camera_name = camera_name
        self.frame_buffer = collections.deque(maxlen=num_stack)

    def _process_frame(self, raw_img):
        # Fix OpenGL vertical flip and convert to CHW float32
        flipped = np.flipud(raw_img)
        chw = np.transpose(flipped, (2, 0, 1)).copy()
        return chw.astype(np.float32) / 255.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        raw_img = obs[self.camera_name]
        processed_frame = self._process_frame(raw_img)

        # Clear buffer and fill with the initial frame
        self.frame_buffer.clear()
        for _ in range(self.num_stack):
            self.frame_buffer.append(processed_frame)

        stacked_pixels = np.concatenate(list(self.frame_buffer), axis=0) # Shape: (3*C, H, W)
        
        return {
            "pixels": torch.from_numpy(stacked_pixels).float(),
            "proprio": torch.from_numpy(obs["robot0_proprio-state"]).float()
        }

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed_frame = self._process_frame(obs[self.camera_name])
        self.frame_buffer.append(processed_frame)

        stacked_pixels = np.concatenate(list(self.frame_buffer), axis=0)
        
        processed_obs = {
            "pixels": torch.from_numpy(stacked_pixels).float(),
            "proprio": torch.from_numpy(obs["robot0_proprio-state"]).float()
        }
        return processed_obs, reward, done, info

class SpatialSoftmaxEncoder(nn.Module):
    """
    Shallow CNN Encoder with Spatial Softmax for extraction of explicit 2D spatial keypoints.
    """
    def __init__(self, input_channels=9, num_filters=32, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        # Shallow 4-layer CNN (Standard Nature-CNN inspired architecture for Visual RL)
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        # Precompute grid coordinates for spatial softmax [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 42),
            torch.linspace(-1.0, 1.0, 42),
            indexing="ij"
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, x):
        # x shape: (B, C_stacked, H, W) e.g., (B, 9, 84, 84)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))  # Shape: (B, num_filters, 42, 42)

        B, C, H, W = h.shape
        features = h.view(B, C, H * W)
        
        # Softmax over spatial dimensions
        softmax_attention = F.softmax(features / self.temperature, dim=-1)

        # Compute expected 2D spatial keypoint coordinates
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=-1, keepdim=True)

        # Concatenate (x, y) keypoints for each channel
        keypoints = torch.cat([expected_x, expected_y], dim=-1).view(B, C * 2)
        return keypoints  # Output dimension: B x (num_filters * 2) = B x 64