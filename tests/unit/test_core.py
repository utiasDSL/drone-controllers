from __future__ import annotations

import inspect
import tomllib
from pathlib import Path
from typing import Any, Callable

import array_api_strict
import pytest

from drone_controllers.core import load_params, parametrize
from drone_controllers.drones import Drones
from drone_controllers.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)

_MELLINGER_FNS = [state2attitude, attitude2force_torque, force_torque2rotor_vel]


@pytest.mark.unit
@pytest.mark.parametrize("fn", _MELLINGER_FNS, ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("drone_model", Drones)
def test_load_params_keys(fn: Callable[..., Any], drone_model: Drones) -> None:
    params = load_params(fn, drone_model)
    kwonly = {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    assert kwonly <= set(params.keys()), f"Missing keys: {kwonly - set(params.keys())}"


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_load_params_values(drone_model: Drones) -> None:
    params = load_params(state2attitude, drone_model)
    toml_path = Path(__file__).parents[2] / "drone_controllers/mellinger/params.toml"
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)
    expected_mass = raw[drone_model.value]["core"]["mass"]
    assert float(params["mass"]) == pytest.approx(expected_mass)


@pytest.mark.unit
def test_load_params_unknown_drone() -> None:
    with pytest.raises(KeyError, match="nonexistent_drone"):
        load_params(state2attitude, "nonexistent_drone")


@pytest.mark.unit
def test_parametrize_unknown_drone() -> None:
    with pytest.raises(KeyError):
        parametrize(state2attitude, "nonexistent_drone")


@pytest.mark.unit
@pytest.mark.parametrize("drone_model", Drones)
def test_parametrize_xp_namespace(drone_model: Drones) -> None:
    controller = parametrize(state2attitude, drone_model, xp=array_api_strict)
    xp_array_type = type(array_api_strict.asarray(0.0))
    assert all(isinstance(v, xp_array_type) for v in controller.keywords.values())
