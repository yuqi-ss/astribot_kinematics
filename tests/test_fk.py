"""Smoke tests for astribot_kinematics.AstribotFK.

Run with:  pytest -q tests/test_fk.py
"""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from astribot_kinematics import (
    ARM_LEFT_JOINTS,
    ARM_RIGHT_JOINTS,
    AstribotFK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
    FULL_JOINT_ORDER,
    TORSO_JOINTS,
)


@pytest.fixture(scope="module")
def fk() -> AstribotFK:
    return AstribotFK()


def test_joint_layout(fk):
    assert fk.nq == 18
    assert fk.joint_names == list(FULL_JOINT_ORDER)
    assert fk.joint_names[:4] == TORSO_JOINTS
    assert fk.joint_names[4:11] == ARM_LEFT_JOINTS
    assert fk.joint_names[11:] == ARM_RIGHT_JOINTS


def test_joint_limits_shape(fk):
    lower, upper = fk.joint_limits()
    assert lower.shape == (18,)
    assert upper.shape == (18,)
    assert np.all(upper >= lower)


def test_frame_keys(fk):
    for key in (
        CHAIN_TORSO,
        CHAIN_ARM_LEFT,
        CHAIN_ARM_RIGHT,
        "astribot_arm_left_tool",
        "astribot_arm_right_tool",
    ):
        assert key in fk.frame_keys


def test_forward_neutral(fk):
    q = np.zeros(18)
    out = fk.forward(q, format="xyzquat")

    assert set(out.keys()) >= {CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT}
    for pose in out.values():
        assert pose.shape == (7,)
        assert np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-6)

    left = out[CHAIN_ARM_LEFT][:3]
    right = out[CHAIN_ARM_RIGHT][:3]
    assert np.allclose(left[[0, 2]], right[[0, 2]], atol=5e-3)
    assert np.sign(left[1]) != np.sign(right[1])


def test_matrix_vs_xyzquat(fk):
    q = np.zeros(18)
    mat = fk.forward(q, format="matrix")[CHAIN_ARM_LEFT]
    pose = fk.forward(q, format="xyzquat")[CHAIN_ARM_LEFT]
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat[:3, 3], pose[:3], atol=1e-10)
    quat = pin.Quaternion(pose[6], pose[3], pose[4], pose[5])  # w, x, y, z
    np.testing.assert_allclose(quat.toRotationMatrix(), mat[:3, :3], atol=1e-8)


def test_dict_input_matches_flat(fk):
    q_flat = fk.build_q(
        torso=[0.275, -0.55, 0.275, 0.0],
        arm_left=[-0.09622, -0.4218, -1.1273, 1.6168, -0.4149, 0.0645, 0.4225],
        arm_right=[0.09622, -0.4218, 1.1273, 1.6168, 0.4149, 0.0645, -0.4225],
    )
    q_dict = {
        CHAIN_TORSO: [0.275, -0.55, 0.275, 0.0],
        CHAIN_ARM_LEFT: [-0.09622, -0.4218, -1.1273, 1.6168, -0.4149, 0.0645, 0.4225],
        CHAIN_ARM_RIGHT: [0.09622, -0.4218, 1.1273, 1.6168, 0.4149, 0.0645, -0.4225],
    }
    p_flat = fk.forward(q_flat)
    p_dict = fk.forward(q_dict)
    for k in p_flat:
        np.testing.assert_allclose(p_flat[k], p_dict[k], atol=1e-12)


def test_batch_matches_single(fk):
    rng = np.random.default_rng(0)
    lower, upper = fk.joint_limits()
    q_batch = rng.uniform(lower, upper, size=(4, 5, 18))

    batched = fk.forward_batch(q_batch)
    for k, arr in batched.items():
        assert arr.shape == (4, 5, 7)
        for i in range(4):
            for j in range(5):
                single = fk.forward(q_batch[i, j], links=[k])[k]
                np.testing.assert_allclose(arr[i, j], single, atol=1e-10)


def test_invalid_input_raises(fk):
    with pytest.raises(ValueError):
        fk.forward(np.zeros(17))
    with pytest.raises(KeyError):
        fk.forward(np.zeros(18), links=["not_a_link"])
    with pytest.raises(ValueError):
        fk.forward(np.zeros(18), format="bogus")


def test_jacobian_shape_and_finite_difference(fk):
    """Validate the user-layout Jacobian against numerical differentiation."""
    rng = np.random.default_rng(42)
    lower, upper = fk.joint_limits()
    q = rng.uniform(lower * 0.3, upper * 0.3)  # stay well within limits

    link = CHAIN_ARM_LEFT
    J = fk.jacobian(q, link=link, reference="local_world_aligned")
    assert J.shape == (6, 18)

    eps = 1e-6
    J_num = np.zeros((3, 18))
    for i in range(18):
        qp = q.copy(); qp[i] += eps
        qm = q.copy(); qm[i] -= eps
        pp = fk.forward(qp, links=[link])[link][:3]
        pm = fk.forward(qm, links=[link])[link][:3]
        J_num[:, i] = (pp - pm) / (2 * eps)

    np.testing.assert_allclose(J[:3], J_num, atol=1e-5, rtol=1e-4)


def test_manipulability_nonnegative(fk):
    rng = np.random.default_rng(123)
    lower, upper = fk.joint_limits()
    for _ in range(5):
        q = rng.uniform(lower * 0.3, upper * 0.3)
        m = fk.manipulability(q, link=CHAIN_ARM_LEFT, translation_only=True)
        assert m >= 0.0
