"""Offline forward / inverse kinematics for the Astribot S1 humanoid.

A self-contained, pure-Python package that bundles the official Astribot S1
URDF (torso + dual arm, 18 DoF) and exposes both FK and IK through pinocchio.
No ROS / SDK runtime is required.

Quick start
-----------
>>> from astribot_kinematics import AstribotFK, AstribotIK
>>>
>>> fk = AstribotFK()
>>> q = fk.build_q(
...     torso=[0.275, -0.55, 0.275, 0.0],
...     arm_left=[-0.09622, -0.4218, -1.1273, 1.6168, -0.4149, 0.0645, 0.4225],
...     arm_right=[0.09622, -0.4218, 1.1273, 1.6168, 0.4149, 0.0645, -0.4225],
... )
>>> fk.eef_left(q)                         # (7,) xyzquat
>>> fk.manipulability(q, "astribot_arm_left")
>>>
>>> ik = AstribotIK(fk=fk, chains=["astribot_torso", "astribot_arm_left"])
>>> target = fk.eef_left(q)                # reproduce the same pose
>>> q_sol, ok, info = ik.solve({"astribot_arm_left": target})
"""

from .constants import (
    ARM_LEFT_JOINTS,
    ARM_RIGHT_JOINTS,
    BUNDLED_URDF,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_JOINTS,
    CHAIN_TORSO,
    EEF_LINKS,
    FULL_JOINT_ORDER,
    TORSO_JOINTS,
    default_urdf_path,
)
from .fk import AstribotFK
from .ik import AstribotIK

__all__ = [
    "AstribotFK",
    "AstribotIK",
    "FULL_JOINT_ORDER",
    "TORSO_JOINTS",
    "ARM_LEFT_JOINTS",
    "ARM_RIGHT_JOINTS",
    "CHAIN_JOINTS",
    "CHAIN_TORSO",
    "CHAIN_ARM_LEFT",
    "CHAIN_ARM_RIGHT",
    "EEF_LINKS",
    "BUNDLED_URDF",
    "default_urdf_path",
]
__version__ = "0.1.0"
