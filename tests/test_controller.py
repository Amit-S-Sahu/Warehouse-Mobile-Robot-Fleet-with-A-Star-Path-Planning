import math

import pytest

from warehouse_fleet.config import GAINS, ROBOT
from warehouse_fleet.controller import WaypointFollower, diff_drive_ik
from warehouse_fleet.geometry import Pose2D, wrap_angle, yaw_from_matrix
from warehouse_fleet.sim_interface import MockSim


def test_wrap_angle():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(2 * math.pi + 0.3) == pytest.approx(0.3)
    assert wrap_angle(-2 * math.pi - 0.4) == pytest.approx(-0.4)
    assert -math.pi <= wrap_angle(3.5) < math.pi


def test_yaw_from_matrix():
    for angle in (-2.1, -0.5, 0.0, 0.8, 2.9):
        c, s = math.cos(angle), math.sin(angle)
        m = [c, -s, 0.0, 1.0,
             s,  c, 0.0, 2.0,
             0.0, 0.0, 1.0, 0.0]
        assert yaw_from_matrix(m) == pytest.approx(angle)


def test_diff_drive_ik_straight_and_spin():
    wl, wr = diff_drive_ik(0.2, 0.0)
    assert wl == pytest.approx(wr)
    assert wl * ROBOT.wheel_radius == pytest.approx(0.2)

    wl_spin, wr_spin = diff_drive_ik(0.0, 1.0)
    assert wl_spin == pytest.approx(-wr_spin)
    assert wr_spin > 0.0


def test_diff_drive_ik_saturates_preserving_ratio():
    wl, wr = diff_drive_ik(10.0, 10.0)
    assert max(abs(wl), abs(wr)) == pytest.approx(ROBOT.max_wheel_speed)


def test_rotate_in_place_when_heading_error_large():
    ctrl = WaypointFollower()
    ctrl.set_path([(0.0, 2.0)])  # 90 deg left of +x
    out = ctrl.step(Pose2D(0.0, 0.0, 0.0), dt=0.05)
    assert out.v == pytest.approx(0.0)
    assert out.omega > 0.0
    assert out.omega_right > out.omega_left


def test_mock_sim_reaches_multi_waypoint_path():
    sim = MockSim(dt=0.05, initial_robots={"Robot_0": Pose2D(-4.0, -4.0, 0.0)})
    sim.start()
    waypoints = [(-4.0, -4.0), (-2.0, -4.0), (-2.0, -2.5)]
    ctrl = WaypointFollower(GAINS, ROBOT)
    ctrl.set_path(waypoints, current_pose=sim.get_pose("Robot_0"))

    for _ in range(600):
        pose = sim.get_pose("Robot_0")
        out = ctrl.step(pose, sim.dt)
        sim.set_wheel_speeds("Robot_0", out.omega_left, out.omega_right)
        if out.goal_reached:
            break
        sim.step()

    assert ctrl.goal_reached
    final_pose = sim.get_pose("Robot_0")
    assert final_pose.distance_to(waypoints[-1]) <= GAINS.goal_tolerance + 1e-3
