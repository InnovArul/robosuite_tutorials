import numpy as np
import robosuite as suite
import torch
from robosuite.controllers import load_composite_controller_config


def make_visual_env(env_name="Lift", robot_name="Panda", controller_name=None):
    """
    Creates a robosuite environment with pixel observations.
    """
    controller_config = load_composite_controller_config(controller=controller_name, robot=robot_name)

    # environment configuration
    env = suite.make(
        env_name=env_name,
        robots=robot_name,
        controller_configs=controller_config,
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=True,  # off-screen renderer is required for pixel observations
        use_camera_obs=True,  # use camera observations
        use_object_obs=False,  # do not use object observations
        camera_names=["agentview", "robot0_eye_in_hand"],  # use "agentview" camera
        camera_heights=84,
        camera_widths=84,
        reward_shaping=True,
        control_freq=20,  # control should happen fast enough so that the simulation looks smooth
    )
    return env


def process_image_to_tensor(image):
    """
    Converts a numpy image array to a PyTorch tensor and normalizes it to [0, 1].
    """
    # Flip the image vertically to fix openGL rendering issue (if needed)
    image = np.flipud(image)

    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Convert to PyTorch tensor and permute dimensions to (C, H, W)
    tensor_image = torch.from_numpy(image).permute(2, 0, 1)
    return tensor_image


if __name__ == "__main__":
    env = make_visual_env()
    obs = env.reset()

    # extract pixel observations from the observation dictionary
    pixel_obs = obs["agentview_image"]   # shape: (84, 84, 3)
    wrist_pixel_obs = obs["robot0_eye_in_hand_image"]  # shape: (84, 84, 3)
    proprio_obs = obs["robot0_proprio-state"]  # shape: (9,) Joint positions, velocities, and gripper state

    # process image observations to be suitable for PyTorch (C, H, W)
    tensor_agentview = process_image_to_tensor(pixel_obs)
    tensor_wrist = process_image_to_tensor(wrist_pixel_obs)

    print("=== Robosuite Pixel Observation Diagnostic ===")
    print(f"Raw Image Shape:         {pixel_obs.shape} | Dtype: {pixel_obs.dtype}")
    print(f"PyTorch Image Tensor:    {tensor_agentview.shape} | Range: [{tensor_agentview.min():.2f}, {tensor_agentview.max():.2f}]")
    print(f"Proprioception Tensor:   {proprio_obs.shape} | Range: [{proprio_obs.min():.2f}, {proprio_obs.max():.2f}]")
    print(f"Action Space Dimension:  {env.action_dim}")
    
    env.close()