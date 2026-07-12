"""
How to Execute:
- Run from the src/ directory:
  python 01_getting_started/visualize_camera.py         # Runs simulation, visualizes camera obs, and saves a video

- Run from inside this folder:
  python visualize_camera.py                            # Runs simulation, visualizes camera obs, and saves a video
"""

import logging
import os

# Suppress initialization warnings if needed
logging.getLogger("robosuite_logs").disabled = True

import cv2
import imageio
import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.wrappers.gym_wrapper import GymWrapper as SuiteGymWrapper
from torchrl.envs import GymWrapper as TorchRLGymWrapper
from torchrl.envs.utils import step_mdp

# Set the image convention to OpenCV (top-left origin) so images are right-side up
macros.IMAGE_CONVENTION = "opencv"


def make_env():
    # Instantiate the robosuite PickPlace env with Panda robot
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,  # No interactive Mujoco window needed, we render offscreen
        has_offscreen_renderer=True,  # Enable offscreen rendering for cameras
        use_camera_obs=True,  # Include camera observations in returned dict
        camera_names=["agentview", "robot0_eye_in_hand"],  # We want both views
        camera_heights=256,
        camera_widths=256,
        reward_shaping=True,
    )
    # Use flatten_obs=False to keep observations as a dictionary instead of a single flat vector.
    # This keeps the image tensors separate and intact.
    gym_compat_env = SuiteGymWrapper(env, flatten_obs=False)
    return TorchRLGymWrapper(gym_compat_env)


if __name__ == "__main__":
    print("Initializing environment...")
    env = make_env()

    # Create video writer using imageio to save the camera observations offline
    video_filename = "camera_observations.mp4"
    print(f"Frames will be recorded and saved to: {os.path.abspath(video_filename)}")
    writer = imageio.get_writer(video_filename, fps=20)

    tensordict = env.reset()

    # Function to extract and compile the visual frame
    def get_combined_frame(td):
        # Extract images from tensordict and convert PyTorch Tensors to NumPy arrays
        agentview_img = td["agentview_image"].cpu().numpy()
        eye_in_hand_img = td["robot0_eye_in_hand_image"].cpu().numpy()

        # Combine the two views side-by-side (256x256 each -> 256x512)
        combined = np.hstack((agentview_img, eye_in_hand_img))
        return combined

    num_steps = 150
    print(f"Running simulation for {num_steps} steps...")

    try:
        for step in range(num_steps):
            # Sample random action
            tensordict["action"] = env.action_spec.rand()
            next_tensordict = env.step(tensordict)

            # Get the current frame from the next state's observations
            frame = get_combined_frame(next_tensordict["next"])

            # Save the frame to video (imageio uses RGB)
            writer.append_data(frame)

            # Optional: Display the frame live if a GUI window is supported (OpenCV uses BGR)
            try:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Robosuite Camera Observations (Left: Agentview | Right: Eye-in-Hand)", bgr_frame)
                # Wait 50ms per frame (~20fps) and check for 'q' to quit early
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    print("Interactive visualization aborted by user.")
                    break
            except Exception:
                # If running headless or OpenCV GUI isn't supported, cv2.imshow will raise an error.
                # We catch it and proceed silently without GUI display.
                pass

            done = next_tensordict["next", "done"].any()
            if done:
                tensordict = env.reset()
            else:
                tensordict = step_mdp(next_tensordict)

            if step % 30 == 0:
                print(f"Step {step}/{num_steps} completed...")

    finally:
        writer.close()
        cv2.destroyAllWindows()
        env.close()

    print(f"Simulation complete! Video saved to {video_filename}")
