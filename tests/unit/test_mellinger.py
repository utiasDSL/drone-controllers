from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest

from drone_controllers import parametrize
from drone_controllers.drones import Drones
from drone_controllers.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)
from drone_controllers.mellinger.params import AttitudeParams, ForceTorqueParams, StateParams

if TYPE_CHECKING:
    from drone_controllers._typing import Array  # To be changed to array_api_typing later


def create_rnd_states(shape: tuple[int, ...] = ()) -> tuple[Array, Array, Array, Array]:
    x = np.random.randn(*shape, 3 + 4 + 3 + 3)
    return x[..., :3], x[..., 3:7], x[..., 7:10], x[..., 10:13]


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_state2attitude(drone_model: Drones):
    # Manually parametrize the controller
    params = StateParams.load(drone_model)
    controller = partial(state2attitude, ctrl_freq=100, **params._asdict())
    # Single input
    pos, quat, vel, ang_vel = create_rnd_states()
    rpyt, pos_err_i = controller(pos, quat, vel, ang_vel, np.ones(13), ctrl_freq=100)
    assert rpyt.shape == (4,)
    assert pos_err_i.shape == (3,)
    # Batch input
    pos, quat, vel, ang_vel = create_rnd_states((5, 4))
    rpyt, pos_err_i = controller(pos, quat, vel, ang_vel, np.ones((5, 4, 13)), ctrl_freq=100)
    assert rpyt.shape == (5, 4, 4)
    assert pos_err_i.shape == (5, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_attitude2force_torque(drone_model: Drones):
    # Manually parametrize the controller
    params = AttitudeParams.load(drone_model)
    controller = partial(attitude2force_torque, ctrl_freq=500, **params._asdict())
    # Single input
    pos, quat, vel, ang_vel = create_rnd_states()
    rpyt_cmd = np.array([0.1, 0.1, 0.1, 1.0])  # roll, pitch, yaw, thrust command
    force_des, torque_des, r_int_error = controller(pos, quat, vel, ang_vel, rpyt_cmd)
    assert force_des.shape == (1,)
    assert torque_des.shape == (3,)
    assert r_int_error.shape == (3,)
    # Batch input
    pos, quat, vel, ang_vel = create_rnd_states((5, 4))
    rpyt_cmd = np.random.randn(5, 4, 4)
    rpyt_cmd[..., 3] = np.abs(rpyt_cmd[..., 3])  # Ensure positive thrust
    force_des, torque_des, r_int_error = controller(pos, quat, vel, ang_vel, rpyt_cmd)
    assert force_des.shape == (5, 4, 1)
    assert torque_des.shape == (5, 4, 3)
    assert r_int_error.shape == (5, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_force_torque2rotor_vel(drone_model: Drones):
    # Manually parametrize the controller
    params = ForceTorqueParams.load(drone_model)
    controller = partial(force_torque2rotor_vel, **params._asdict())
    # Single input
    force = np.array([1.0])
    torque = np.array([0.1, 0.1, 0.1])
    rotor_vel = controller(force, torque)
    assert rotor_vel.shape == (4,)
    # Batch input
    force = np.ones((5, 4, 1))
    torque = np.random.randn(5, 4, 3) * 0.1
    rotor_vel = controller(force, torque)
    assert rotor_vel.shape == (5, 4, 4)


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_state2attitude_parametrize(drone_model: Drones):
    # Test the parametrize function with all available drones
    controller = parametrize(state2attitude, drone_model)
    # Single input test
    pos, quat, vel, ang_vel = create_rnd_states()
    rpyt, pos_err_i = controller(pos, quat, vel, ang_vel, np.ones(13))
    assert rpyt.shape == (4,)
    assert pos_err_i.shape == (3,)
    # Batch input test
    pos, quat, vel, ang_vel = create_rnd_states((3, 2))
    rpyt, pos_err_i = controller(pos, quat, vel, ang_vel, np.ones((3, 2, 13)))
    assert rpyt.shape == (3, 2, 4)
    assert pos_err_i.shape == (3, 2, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_attitude2force_torque_parametrize(drone_model: Drones):
    # Test the parametrize function with all available drones
    controller = parametrize(attitude2force_torque, drone_model)
    # Single input test
    pos, quat, vel, ang_vel = create_rnd_states()
    rpyt_cmd = np.array([0.1, 0.1, 0.1, 1.0])  # roll, pitch, yaw, thrust command
    force_des, torque_des, r_int_error = controller(pos, quat, vel, ang_vel, rpyt_cmd)
    assert force_des.shape == (1,)
    assert torque_des.shape == (3,)
    assert r_int_error.shape == (3,)
    # Batch input test
    pos, quat, vel, ang_vel = create_rnd_states((3, 2))
    rpyt_cmd = np.random.randn(3, 2, 4)
    rpyt_cmd[..., 3] = np.abs(rpyt_cmd[..., 3])  # Ensure positive thrust
    force_des, torque_des, r_int_error = controller(pos, quat, vel, ang_vel, rpyt_cmd)
    assert force_des.shape == (3, 2, 1)
    assert torque_des.shape == (3, 2, 3)
    assert r_int_error.shape == (3, 2, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_force_torque2rotor_vel_parametrize(drone_model: Drones):
    # Test the parametrize function with all available drones
    controller = parametrize(force_torque2rotor_vel, drone_model)
    # Single input test
    force = np.array([1.0])
    torque = np.array([0.1, 0.1, 0.1])
    rotor_vel = controller(force, torque)
    assert rotor_vel.shape == (4,)
    # Batch input test
    force = np.ones((3, 2, 1))
    torque = np.random.randn(3, 2, 3) * 0.1
    rotor_vel = controller(force, torque)
    assert rotor_vel.shape == (3, 2, 4)
