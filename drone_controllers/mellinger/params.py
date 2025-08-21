"""Parameters for the Mellinger controller."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from array_api_typing import Array


class StateParams(NamedTuple):
    """Parameters for the Mellinger state controller."""

    kp: Array
    kd: Array
    ki: Array
    int_err_max: Array
    mass: float
    gravity_vec: Array
    mass_thrust: float
    thrust_max: float
    pwm_max: float

    @staticmethod
    def load(drone_model: str) -> StateParams:
        """Load the parameters from the config file."""
        with open(Path(__file__).parent / "params.toml", "rb") as f:
            params = tomllib.load(f)
        if drone_model not in params:
            raise KeyError(f"Drone model `{drone_model}` not found in params.toml")
        params = params[drone_model]["state2attitude"] | params[drone_model]["core"]
        params = {k: np.asarray(v) for k, v in params.items() if k in StateParams._fields}
        return StateParams(**params)


class AttitudeParams(NamedTuple):
    """Parameters for the Mellinger attitude controller."""

    kR: Array
    kw: Array
    ki_m: Array
    kd_omega: Array
    int_err_max: Array
    torque_pwm_max: Array
    thrust_max: float
    pwm_min: float
    pwm_max: float
    L: float
    KF: float
    KM: float
    mixing_matrix: Array

    @staticmethod
    def load(drone_model: str) -> AttitudeParams:
        """Load the parameters from the config file."""
        with open(Path(__file__).parent / "params.toml", "rb") as f:
            params = tomllib.load(f)
        if drone_model not in params:
            raise KeyError(f"Drone model `{drone_model}` not found in params.toml")
        params = params[drone_model]["attitude2force_torque"] | params[drone_model]["core"]
        params = {k: np.asarray(v) for k, v in params.items() if k in AttitudeParams._fields}
        return AttitudeParams(**params)


class ForceTorqueParams(NamedTuple):
    """Parameters for the Mellinger force torque controller."""

    thrust_min: float
    thrust_max: float
    L: float
    KF: float
    KM: float
    mixing_matrix: Array

    @staticmethod
    def load(drone_model: str) -> ForceTorqueParams:
        """Load the parameters from the config file."""
        with open(Path(__file__).parent / "params.toml", "rb") as f:
            params = tomllib.load(f)
        if drone_model not in params:
            raise KeyError(f"Drone model `{drone_model}` not found in params.toml")
        params = params[drone_model]["core"]
        params = {k: np.asarray(v) for k, v in params.items() if k in ForceTorqueParams._fields}
        return ForceTorqueParams(**params)
