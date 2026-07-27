import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    """
    Random Shift Data Augmentation module for Visual RL (DrQ-v2 style).
    Executes vectorized zero-padding and grid crop operations entirely on GPU.
    """
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
                              Expected float32 normalized in range [0, 1].
        Returns:
            torch.Tensor: Augmented image tensor of shape (B, C, H, W).
        """
        n, c, h, w = x.size()
        assert h == w, "Input image must be square (e.g., 84x84)."
        
        # 1. Pad image edges zero-filling along spatial dimensions (H, W)
        padding = (self.pad, self.pad, self.pad, self.pad)
        x_padded = F.pad(x, padding, mode="replicate") # Replicate boundary pixels

        # 2. Generate normalized crop grid coordinates
        eps = 1.0 / (h + 2 * self.pad)
        
        # Linear sampling grid from -1 to 1
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        
        arange = arange.unsqueeze(0).repeat(h, 1) # Shape: (H, H)
        
        # Base grid coordinates (x, y) stacked along last dimension -> Shape: (1, H, H, 2)
        grid = torch.stack([arange, arange.T], dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)

        # 3. Generate random integer offset translations per batch item
        delta = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        delta = delta * (2.0 / (h + 2 * self.pad))

        # 4. Apply translation offset to sampling grid
        grid = grid + delta

        # 5. Perform grid sampling (bilinear interpolation)
        augmented_x = F.grid_sample(
            x_padded,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        
        return augmented_x


# --- Execution Diagnostic Test ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate a mini-batch of stacked Robosuite observations (B=32, Stacked Channels=9, H=84, W=84)
    dummy_obs_batch = torch.rand(32, 9, 84, 84, device=device)
    
    # Instantiate augmentation module
    aug = RandomShiftsAug(pad=4).to(device)
    
    # Apply data augmentation
    augmented_obs_batch = aug(dummy_obs_batch)
    
    print("=== DrQ-v2 Random Shift Diagnostic ===")
    print(f"Input Tensor Shape:      {dummy_obs_batch.shape}")
    print(f"Augmented Tensor Shape:  {augmented_obs_batch.shape}")
    print(f"Device Executed On:      {augmented_obs_batch.device}")
    
    # Check that tensor dimensions are preserved
    assert dummy_obs_batch.shape == augmented_obs_batch.shape
    print("Augmentation test passed successfully!")
