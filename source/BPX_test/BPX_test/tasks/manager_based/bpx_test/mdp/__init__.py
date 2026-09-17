# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Locomotion-specific reward terms (not part of the core mdp module).
# feet_slide is also reused by the dedicated standing task to suppress zero-command foot drift.
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import feet_air_time, feet_slide  # noqa: F401

from .curriculums import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .velocity_commands import *  # noqa: F401, F403
