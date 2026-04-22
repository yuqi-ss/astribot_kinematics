"""Canonical joint / link names of Astribot S1 (torso + dual arm)."""

from __future__ import annotations

import os


TORSO_JOINTS = [f"astribot_torso_joint_{i}" for i in range(1, 5)]         # 4 DoF
ARM_LEFT_JOINTS = [f"astribot_arm_left_joint_{i}" for i in range(1, 8)]    # 7 DoF
ARM_RIGHT_JOINTS = [f"astribot_arm_right_joint_{i}" for i in range(1, 8)]  # 7 DoF

FULL_JOINT_ORDER = TORSO_JOINTS + ARM_LEFT_JOINTS + ARM_RIGHT_JOINTS       # 18 DoF

# SDK-style chain keys (consistent with astribot_sdk Astribot.{torso,arm_left,arm_right}_name)
CHAIN_TORSO = "astribot_torso"
CHAIN_ARM_LEFT = "astribot_arm_left"
CHAIN_ARM_RIGHT = "astribot_arm_right"

CHAIN_JOINTS = {
    CHAIN_TORSO: TORSO_JOINTS,
    CHAIN_ARM_LEFT: ARM_LEFT_JOINTS,
    CHAIN_ARM_RIGHT: ARM_RIGHT_JOINTS,
}

# World -> torso_base weld transform. The URDF's root link
# ``astribot_torso_base`` sits at the origin, but the Astribot SDK applies an
# extra ``weld_to_base_pose = [0, 0, 0.097, 0, 0, 0]`` from astribot_torso.yaml
# when reporting Cartesian poses, so we need to replay the same offset to
# stay bit-level consistent with the SDK's get_forward_kinematics output.
WORLD_TO_TORSO_BASE_XYZ = (0.0, 0.0, 0.097)
WORLD_TO_TORSO_BASE_RPY = (0.0, 0.0, 0.0)

# Frames exposed by AstribotFK. Keys mirror astribot_sdk naming: the
# ``astribot_arm_*`` keys resolve to the URDF link returned by the SDK's
# ``get_forward_kinematics`` (which already bakes the tool offset
# xyz=(0, -0.15, 0) into ``astribot_arm_*_ee_joint``). ``*_tool`` points to the
# wrist flange just before that tool offset.
EEF_LINKS = {
    CHAIN_TORSO: "astribot_torso_end_effector",
    CHAIN_ARM_LEFT: "astribot_arm_left_end_effector",
    CHAIN_ARM_RIGHT: "astribot_arm_right_end_effector",
    "astribot_arm_left_tool": "astribot_arm_left_tool_link",
    "astribot_arm_right_tool": "astribot_arm_right_tool_link",
}

# Bundled URDF shipped with this package; the module is fully self-contained.
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
BUNDLED_URDF = os.path.join(_PACKAGE_ROOT, "assets", "astribot_whole_body.urdf")


def default_urdf_path() -> str:
    """Return the URDF bundled with this package."""
    if not os.path.isfile(BUNDLED_URDF):
        raise FileNotFoundError(
            f"Bundled URDF missing at {BUNDLED_URDF}. "
            "Reinstall the package or pass an explicit `urdf_path=`."
        )
    return BUNDLED_URDF
