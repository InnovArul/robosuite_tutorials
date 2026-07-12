"""
How to Execute:
- Run from the src/ directory:
  python 01_getting_started/hello_robosuite.py          # Headless Mode
  mjpython 01_getting_started/hello_robosuite.py        # Visualizer Mode (macOS GUI)

- Run from inside this folder:
  python hello_robosuite.py                             # Headless Mode
  mjpython hello_robosuite.py                           # Visualizer Mode (macOS GUI)
"""

import logging

# Suppress initialization warnings if needed
logging.getLogger("robosuite_logs").disabled = True

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper as SuiteGymWrapper
from torchrl.envs import GymWrapper as TorchRLGymWrapper
from torchrl.envs.utils import step_mdp


def make_env():
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    gym_compat_env = SuiteGymWrapper(env)
    return TorchRLGymWrapper(gym_compat_env)


if __name__ == "__main__":
    env = make_env()
    tensordict = env.reset()

    for _ in range(100):  # Run for a short sequence of steps
        tensordict["action"] = env.action_spec.rand()
        next_tensordict = env.step(tensordict)

        done = next_tensordict["next", "done"].any()
        if done:
            tensordict = env.reset()
        else:
            tensordict = step_mdp(next_tensordict)

    env.close()
