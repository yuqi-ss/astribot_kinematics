"""Smoke tests for astribot_kinematics.AstribotIK.

Strategy: generate a random feasible q*, run FK to obtain a target pose,
perturb the initial guess, and verify IK recovers a configuration whose
FK pose matches the target within tight tolerances.

Run with:  pytest -q tests/test_ik.py
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from astribot_kinematics import (
    AstribotFK,
    AstribotIK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)


@pytest.fixture(scope="module")
def fk() -> AstribotFK:
    return AstribotFK()


def _pose_error(a: np.ndarray, b: np.ndarray) -> float:
    """Translation error between two (7,) xyzquat poses."""
    return float(np.linalg.norm(a[:3] - b[:3]))


def test_single_arm_ik_recovers_fk(fk):
    rng = np.random.default_rng(7)
    lower, upper = fk.joint_limits()

    q_star = fk.build_q(
        torso=rng.uniform(lower[:4] * 0.3, upper[:4] * 0.3),
        arm_left=rng.uniform(lower[4:11] * 0.3, upper[4:11] * 0.3),
    )
    target = fk.eef_left(q_star)

    ik = AstribotIK(fk=fk, chains=[CHAIN_ARM_LEFT])
    q_init = fk.build_q(
        torso=q_star[:4],                     # keep torso same (inactive chain)
        arm_left=np.zeros(7),
    )

    q_sol, ok, info = ik.solve(
        {CHAIN_ARM_LEFT: target},
        q_init=q_init,
        max_iters=200,
        tol_pos=1e-5,
        tol_rot=1e-4,
    )
    assert ok, f"IK did not converge: {info}"
    achieved = fk.eef_left(q_sol)
    assert _pose_error(achieved, target) < 1e-4


def test_torso_plus_arm_left(fk):
    rng = np.random.default_rng(17)
    lower, upper = fk.joint_limits()

    q_star = fk.build_q(
        torso=rng.uniform(lower[:4] * 0.4, upper[:4] * 0.4),
        arm_left=rng.uniform(lower[4:11] * 0.4, upper[4:11] * 0.4),
    )
    target = fk.eef_left(q_star)

    ik = AstribotIK(fk=fk, chains=[CHAIN_TORSO, CHAIN_ARM_LEFT])
    q_sol, ok, info = ik.solve(
        {CHAIN_ARM_LEFT: target},
        q_init=np.zeros(18),
        max_iters=300,
    )
    # Over-determined DoFs vs 6-D task, so this should converge comfortably.
    assert ok, f"IK did not converge: {info}"
    np.testing.assert_allclose(fk.eef_left(q_sol)[:3], target[:3], atol=5e-4)


def test_bimanual(fk):
    rng = np.random.default_rng(31)
    lower, upper = fk.joint_limits()

    q_star = fk.build_q(
        torso=rng.uniform(lower[:4] * 0.3, upper[:4] * 0.3),
        arm_left=rng.uniform(lower[4:11] * 0.3, upper[4:11] * 0.3),
        arm_right=rng.uniform(lower[11:] * 0.3, upper[11:] * 0.3),
    )
    tgt_l = fk.eef_left(q_star)
    tgt_r = fk.eef_right(q_star)

    ik = AstribotIK(fk=fk, chains=[CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT])
    q_sol, ok, info = ik.solve_bimanual(
        tgt_l, tgt_r, q_init=np.zeros(18), max_iters=400
    )
    assert ok, f"IK did not converge: {info}"
    assert _pose_error(fk.eef_left(q_sol), tgt_l) < 5e-4
    assert _pose_error(fk.eef_right(q_sol), tgt_r) < 5e-4


def test_position_only_mode(fk):
    """3-DoF position task with a 7-DoF arm should always solve."""
    rng = np.random.default_rng(99)
    lower, upper = fk.joint_limits()

    q_star = fk.build_q(
        arm_left=rng.uniform(lower[4:11] * 0.3, upper[4:11] * 0.3),
    )
    target = fk.eef_left(q_star)

    ik = AstribotIK(fk=fk, chains=[CHAIN_ARM_LEFT])
    q_sol, ok, info = ik.solve(
        {CHAIN_ARM_LEFT: target},
        position_only=True,
        max_iters=200,
        tol_pos=1e-5,
    )
    assert ok, info
    np.testing.assert_allclose(fk.eef_left(q_sol)[:3], target[:3], atol=5e-4)


def test_limits_respected(fk):
    """An unreachable target should not drive joints outside URDF limits."""
    ik = AstribotIK(fk=fk, chains=[CHAIN_ARM_LEFT])
    impossible = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 1.0])
    q_sol, _, _ = ik.solve(
        {CHAIN_ARM_LEFT: impossible},
        max_iters=50,
        respect_limits=True,
    )
    lower, upper = fk.joint_limits()
    assert np.all(q_sol >= lower - 1e-9)
    assert np.all(q_sol <= upper + 1e-9)


