# Robosuite Tutorials

This repository contains tutorials and examples for working with the [robosuite](https://robosuite.ai/) robot simulation framework and reinforcement learning.

## Setup

Ensure your Conda environment is active. For example:
```bash
conda activate hf
```

## Tutorials

### 1. Getting Started (`src/01_getting_started/`)

*   **[01_hello_robosuite.py](file:///Users/innov/codebases/robosuite_tutorials/src/01_getting_started/01_hello_robosuite.py)**: A basic hello-world style tutorial using the interactive PyTorch/TorchRL wrapper to control a Panda robot using random actions in a PickPlace environment.
*   **[02_visualize_camera.py](file:///Users/innov/codebases/robosuite_tutorials/src/01_getting_started/02_visualize_camera.py)**: Demonstrates offscreen rendering from multiple cameras (agentview, eye-in-hand) and saving the recorded frames to a video file (`camera_observations.mp4`).
*   **[03_depth_and_semantics.py](file:///Users/innov/codebases/robosuite_tutorials/src/01_getting_started/03_depth_and_semantics.py)**: Demonstrates advanced rendering features, including:
    *   Retrieving **depth maps** and converting raw normalized depth to actual metric distance (meters) using `get_real_depth_map`.
    *   Retrieving **instance-level** and **class-level semantic segmentation maps**.
    *   Resolving segmentations back to human-readable object class and instance names (e.g., `Milk`, `Panda`, `bin1`).
    *   Compiling all four modalities (RGB, Depth, Instance, and Class) into a 2x2 grid video (`depth_and_semantics.mp4`).

To run the depth and semantics tutorial:
```bash
# Run from the src/ directory
python 01_getting_started/03_depth_and_semantics.py
```
