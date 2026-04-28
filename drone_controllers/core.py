"""Core functionalities for controller parametrization."""

from __future__ import annotations

import inspect
import tomllib
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

P = ParamSpec("P")
R = TypeVar("R")


def parametrize(
    fn: Callable[P, R], drone_model: str, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a controller function with the default parameters for a drone model.

    Args:
        fn: The controller function to parametrize.
        drone_model: The drone model to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Example:
        >>> from drone_controllers import parametrize
        >>> from drone_controllers.mellinger import state2attitude
        >>> controller = parametrize(state2attitude, drone_model="cf2x_L250")
        >>> command_rpyt, int_pos_err = controller(
        ...     pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, cmd=cmd, ctrl_freq=100
        ... )

    Returns:
        The parametrized controller function with all keyword-only parameters filled in.
    """
    xp = np if xp is None else xp
    controller = fn.__module__.split(".")[-2]
    fn_name = fn.__name__
    try:
        sig = inspect.signature(fn)
        kwonly_params = [
            name
            for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        params = load_params(controller, fn_name, drone_model, xp=xp)
        params = {k: xp.asarray(v, device=device) for k, v in params.items() if k in kwonly_params}
    except KeyError as e:
        raise KeyError(
            f"Drone model `{drone_model}` not found for controller `{controller}.{fn_name}`"
        ) from e
    return partial(fn, **params)


def load_params(
    controller: str, fn_name: str, drone_model: str, xp: ModuleType | None = None
) -> dict[str, Any]:
    """Load and merge parameters for a controller function and drone model.

    Reads parameters from the controller's ``params.toml`` file, merging the
    shared ``[drone_model.core]`` section with the function-specific
    ``[drone_model.{fn_name}]`` section (if it exists).

    Args:
        controller: Name of the controller sub-package, e.g. ``"mellinger"``.
        fn_name: Name of the controller function, e.g. ``"state2attitude"``.
        drone_model: Name of the drone configuration, e.g. ``"cf2x_L250"``.
        xp: Array API module used to convert parameter values. If ``None``,
            NumPy is used.

    Returns:
        A flat dict mapping parameter names to arrays in the requested array namespace.

    Raises:
        KeyError: If ``drone_model`` is not found in the TOML file.
    """
    xp = np if xp is None else xp
    with open(Path(__file__).parent / f"{controller}/params.toml", "rb") as f:
        all_params = tomllib.load(f)
    if drone_model not in all_params:
        raise KeyError(f"Drone model `{drone_model}` not found in {controller}/params.toml")
    drone_params = all_params[drone_model]
    params = dict(drone_params.get("core", {}))
    params |= drone_params.get(fn_name, {})
    return {k: xp.asarray(v) for k, v in params.items()}
