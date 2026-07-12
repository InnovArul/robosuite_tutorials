"""
How to Execute:
- Run from the src/ directory:
  python 01_getting_started/03_depth_and_semantics.py   # Runs simulation, visualizes RGB/depth/segmentation, and saves a video

- Run from inside this folder:
  python 03_depth_and_semantics.py                      # Runs simulation, visualizes RGB/depth/segmentation, and saves a video
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
from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.wrappers.gym_wrapper import GymWrapper as SuiteGymWrapper
from torchrl.envs import GymWrapper as TorchRLGymWrapper
from torchrl.envs.utils import step_mdp

# Set the image convention to OpenCV (top-left origin) so images are right-side up
macros.IMAGE_CONVENTION = "opencv"

# Predefined premium color palette for segmentation mapping (RGB format)
PRESET_COLORS = np.array([
    [244, 67, 54],    # Red
    [76, 175, 80],    # Green
    [33, 150, 243],   # Blue
    [255, 235, 59],   # Yellow
    [156, 39, 176],   # Purple
    [0, 188, 212],    # Cyan
    [255, 152, 0],    # Orange
    [233, 30, 99],    # Pink
    [139, 195, 74],   # Light Green
    [0, 150, 136],    # Teal
    [103, 58, 183],   # Deep Purple
    [255, 87, 34],    # Deep Orange
    [121, 85, 72],    # Brown
    [158, 158, 158],  # Grey
    [96, 125, 139],   # Blue Grey
], dtype=np.uint8)


class MinMaxScaler:
    """
    Min-Max Scaler to normalize depth maps to the [0.0, 1.0] range.
    This replicates the MinMaxScaler interface in a lightweight and robust way,
    handling potential NaNs or division by zero from simulator edge cases.
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def fit_transform(self, X):
        X_min = np.nanmin(X)
        X_max = np.nanmax(X)
        
        if X_max > X_min:
            X_std = (X - X_min) / (X_max - X_min)
        else:
            X_std = np.zeros_like(X)
        
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        # Clean up any NaNs or infinite values that might propagate from simulator instabilities
        return np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)


def make_env(width=256, height=256):
    # Instantiate the robosuite TwoArmHandover env with two Panda robots.
    # We enable:
    # 1. 'frontview' (global view of the environment)
    # 2. 'robot0_eye_in_hand' (camera mounted on Panda 0's gripper)
    # 3. 'robot1_eye_in_hand' (camera mounted on Panda 1's gripper)
    env = suite.make(
        "TwoArmHandover",
        robots=["Panda", "Panda"],
        has_renderer=False,            # No interactive Mujoco window needed, we render offscreen
        has_offscreen_renderer=True,    # Enable offscreen rendering for cameras
        use_camera_obs=True,            # Include camera observations in returned dict
        camera_names=["frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
        camera_heights=height,
        camera_widths=width,
        camera_depths=True,             # Enable depth map observation for all cameras
        # We pass a nested list of length 3 to work around the robosuite input2list broadcasting bug
        camera_segmentations=[["instance", "class"], ["instance", "class"], ["instance", "class"]],
        reward_shaping=True,
    )

    # Specify keys to include in the GymWrapper observation dictionary.
    keys = [
        "object-state",
        "frontview_image",
        "frontview_depth",
        "frontview_segmentation_instance",
        "frontview_segmentation_class",
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_depth",
        "robot0_eye_in_hand_segmentation_instance",
        "robot0_eye_in_hand_segmentation_class",
        "robot1_eye_in_hand_image",
        "robot1_eye_in_hand_depth",
        "robot1_eye_in_hand_segmentation_instance",
        "robot1_eye_in_hand_segmentation_class",
        "robot0_proprio-state",
        "robot1_proprio-state"
    ]

    # Use flatten_obs=False to keep observations as a dictionary instead of a single flat vector.
    gym_compat_env = SuiteGymWrapper(env, keys=keys, flatten_obs=False)
    return TorchRLGymWrapper(gym_compat_env)


def print_segmentation_mappings(instance_names, class_names):
    """Prints resolved class/instance names mapping tables on start."""
    print("\n--- Segmentation Mappings ---")
    print("Instance ID Map:")
    print("  0: Background")
    for idx, name in enumerate(instance_names):
        print(f"  {idx + 1}: {name}")

    print("\nClass ID Map:")
    print("  0: Background")
    for idx, name in enumerate(class_names):
        print(f"  {idx + 1}: {name}")
    print("-----------------------------\n")


def colorize_segmentation(seg_map, background_color=(30, 30, 30)):
    """
    Map the integer segmentation map to a beautiful, clean color image.
    Value 0 is background, and values >= 1 are mapped to distinct preset colors.
    """
    seg_map = np.squeeze(seg_map)
    H, W = seg_map.shape

    # Initialize with background color
    color_img = np.full((H, W, 3), background_color, dtype=np.uint8)

    # Find unique non-zero IDs
    unique_ids = np.unique(seg_map)
    unique_ids = unique_ids[unique_ids > 0]

    for uid in unique_ids:
        mask = (seg_map == uid)
        color = PRESET_COLORS[(uid - 1) % len(PRESET_COLORS)]
        color_img[mask] = color

    return color_img


def process_camera_view(td_state, cam_name, seg_type, sim, scaler, instance_names, class_names, width, height):
    """
    Extracts, colorizes, and annotates RGB, depth, and segmentation maps for a single camera.
    """
    # Extract data and make copies to avoid memory corruption on shared buffers
    rgb = td_state[f"{cam_name}_image"].cpu().numpy().copy()
    raw_depth = td_state[f"{cam_name}_depth"].cpu().numpy().copy()
    raw_depth = np.clip(raw_depth, 0.0, 1.0)
    
    # Convert raw depth map to actual distance in meters
    real_depth = get_real_depth_map(sim, raw_depth)
    
    # Colorize depth map using MinMaxScaler
    depth_scaled = scaler.fit_transform(raw_depth.reshape(-1, 1)).reshape(height, width)
    depth_uint8 = ((1.0 - depth_scaled) * 255.0).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    
    # Extract and colorize segmentation map
    seg_map = td_state[f"{cam_name}_segmentation_{seg_type}"].cpu().numpy().copy()
    seg_colored = colorize_segmentation(seg_map)
    
    # Query center pixel metrics
    center_x = width // 2
    center_y = height // 2
    center_distance = real_depth[center_y, center_x, 0]
    center_id = seg_map[center_y, center_x, 0]
    
    names_list = instance_names if seg_type == "instance" else class_names
    center_obj_name = names_list[center_id - 1] if center_id > 0 else "Background"
    
    return rgb, depth_colored, seg_colored, center_distance, center_obj_name


def annotate_grid_row(grid_img, cam_label, seg_label, center_distance, center_obj_name, row_idx, width, height):
    """
    Draws text labels, metric depth info, and crosshair overlays onto a row of the compiled grid.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    color_white = (255, 255, 255)
    color_yellow = (0, 255, 255)
    thickness = 1
    
    center_x = width // 2
    center_y = height // 2
    y_offset = row_idx * height

    # Col 1: RGB
    cv2.putText(grid_img, cam_label, (10, y_offset + 20), font, font_scale, color_white, thickness)
    
    # Col 2: Depth
    cv2.putText(grid_img, f"Depth (Center: {center_distance:.3f}m)", (width + 10, y_offset + 20), font, font_scale, color_white, thickness)
    cv2.drawMarker(grid_img, (width + center_x, y_offset + center_y), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)
    
    # Col 3: Segmentation
    cv2.putText(grid_img, seg_label, (2 * width + 10, y_offset + 20), font, font_scale, color_white, thickness)
    cv2.putText(grid_img, f"Center: {center_obj_name}", (2 * width + 10, y_offset + 40), font, font_scale, color_yellow, thickness)
    cv2.drawMarker(grid_img, (2 * width + center_x, y_offset + center_y), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)


if __name__ == "__main__":
    # Define camera resolution parameters
    width = 256
    height = 256

    print(f"Initializing environment with camera resolution {width}x{height}...")
    env = make_env(width=width, height=height)

    # Get underlying raw environment to query model information and sim
    raw_env = env.unwrapped
    sim = raw_env.sim

    # Initialize our MinMaxScaler for depth map visualization
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    # Retrieve instance and class lists to resolve integer IDs to human-readable names
    instance_names = list(raw_env.model.instances_to_ids.keys())
    class_names = list(raw_env.model.classes_to_ids.keys())

    # Print ID mapping tables
    print_segmentation_mappings(instance_names, class_names)

    # Create video writer to save the compiled frame grid
    video_filename = "depth_and_semantics.mp4"
    print(f"Frames will be recorded and saved to: {os.path.abspath(video_filename)}")
    writer = imageio.get_writer(video_filename, fps=20)

    tensordict = env.reset()

    num_steps = 150
    print(f"Running simulation for {num_steps} steps...")

    try:
        for step in range(num_steps):
            # Sample random action
            tensordict["action"] = env.action_spec.rand()
            next_tensordict = env.step(tensordict)

            td_state = next_tensordict["next"]

            # --- PROCESS CHANNELS ---
            # Frontview (Row 1) - Instance Segmentation
            f_rgb, f_depth_col, f_seg_col, f_dist, f_name = process_camera_view(
                td_state, "frontview", "instance", sim, scaler, instance_names, class_names, width, height
            )
            # Robot 0 Hand (Row 2) - Class Segmentation
            r0_rgb, r0_depth_col, r0_seg_col, r0_dist, r0_name = process_camera_view(
                td_state, "robot0_eye_in_hand", "class", sim, scaler, instance_names, class_names, width, height
            )
            # Robot 1 Hand (Row 3) - Instance Segmentation
            r1_rgb, r1_depth_col, r1_seg_col, r1_dist, r1_name = process_camera_view(
                td_state, "robot1_eye_in_hand", "instance", sim, scaler, instance_names, class_names, width, height
            )

            # --- COMPILE GRID VIEW (3 rows, 3 columns -> (3 * height) x (3 * width) pixels) ---
            row1 = np.hstack((f_rgb, f_depth_col, f_seg_col))
            row2 = np.hstack((r0_rgb, r0_depth_col, r0_seg_col))
            row3 = np.hstack((r1_rgb, r1_depth_col, r1_seg_col))
            combined_grid = np.vstack((row1, row2, row3))

            # --- ADD LABELS AND ANNOTATIONS ---
            annotate_grid_row(combined_grid, "RGB Camera (Frontview)", "Instance Segmentation", f_dist, f_name, 0, width, height)
            annotate_grid_row(combined_grid, "RGB Camera (Robot 0 Hand)", "Class Segmentation", r0_dist, r0_name, 1, width, height)
            annotate_grid_row(combined_grid, "RGB Camera (Robot 1 Hand)", "Instance Segmentation", r1_dist, r1_name, 2, width, height)

            # Save the frame (imageio uses RGB)
            writer.append_data(combined_grid)

            # Optional: Display live visualization if GUI window is supported (OpenCV uses BGR)
            try:
                bgr_grid = cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR)
                cv2.imshow("Robosuite Modalities Grid (Top: Frontview | Mid: Robot0 Hand | Bottom: Robot1 Hand)", bgr_grid)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    print("Interactive visualization aborted by user.")
                    break
            except Exception:
                pass

            done = next_tensordict["next", "done"].any()
            if done:
                tensordict = env.reset()
            else:
                tensordict = step_mdp(next_tensordict)

            if step % 30 == 0:
                print(f"Step {step}/{num_steps} completed...")
                print(f"  Frontview - Center depth: {f_dist:.3f}m | Obj: {f_name}")
                print(f"  Robot 0 Hand - Center depth: {r0_dist:.3f}m | Obj: {r0_name}")
                print(f"  Robot 1 Hand - Center depth: {r1_dist:.3f}m | Obj: {r1_name}")

    finally:
        writer.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()

    print(f"\nSimulation complete! Video saved to {os.path.abspath(video_filename)}")
