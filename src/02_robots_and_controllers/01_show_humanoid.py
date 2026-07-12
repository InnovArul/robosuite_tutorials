"""
How to Execute:
- Run from the src/ directory:
  python 02_robots_and_controllers/show_humanoid.py     # Headless Mode
  mjpython 02_robots_and_controllers/show_humanoid.py   # Visualizer Mode (macOS GUI)

- Run from inside this folder:
  python show_humanoid.py                               # Headless Mode
  mjpython show_humanoid.py                             # Visualizer Mode (macOS GUI)
"""

import logging

# Suppress robosuite init warnings
logging.getLogger("robosuite_logs").disabled = True

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper as SuiteGymWrapper
from torchrl.envs import GymWrapper as TorchRLGymWrapper
from torchrl.envs.utils import step_mdp


def make_env():
    # Instantiate the H1 humanoid robot (H1FixedLowerBody)
    # The G1FixedLowerBody robot is also available and works without additional solvers
    env = suite.make(
        "Lift",
        robots="H1FixedLowerBody",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    gym_compat_env = SuiteGymWrapper(env)
    return TorchRLGymWrapper(gym_compat_env)


if __name__ == "__main__":
    print("Initializing H1 Humanoid Lift Environment...")
    env = make_env()
    tensordict = env.reset()
    print("Environment reset successful. Running simulation steps...")

    for i in range(300):
        # Sample random actions for the humanoid
        tensordict["action"] = env.action_spec.rand()
        next_tensordict = env.step(tensordict)

        done = next_tensordict["next", "done"].any()
        if done:
            tensordict = env.reset()
            print("Episode finished. Resetting...")
        else:
            tensordict = step_mdp(next_tensordict)

    env.close()
    print("Simulation complete.")
