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
    fn: Callable[P, R],
    drone_model: str,
    xp: ModuleType | None = None,
    device: str | None = None,
) -> Callable[P, R]:
    """Parametrize a controller function with the default controller parameters for a drone model.

    Args:
        fn: The controller function to parametrize.
        drone_model: The drone model to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If None, the device is inferred from the xp module.

    Example:
        >>> from drone_controllers.core import parametrize
        >>> from drone_controllers.mellinger import state2attitude
        >>> controller_fn = parametrize(state2attitude, drone_model="cf2x_L250")
        >>> command_rpyt, int_pos_err = controller_fn(
        ...     pos=pos,
        ...     quat=quat,
        ...     vel=vel,
        ...     ang_vel=ang_vel,
        ...     cmd=cmd,
        ...     ctrl_errors=(int_pos_err,),
        ...     ctrl_freq=100,
        ... )

    Returns:
        The parametrized controller function with all keyword argument only parameters filled in.
    """
    xp = np if xp is None else xp
    controller = fn.__module__.split(".")[-2]
    sig = inspect.signature(fn)
    kwonly_params = {
        name
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    }
    try:
        params = load_params(controller, fn.__name__, drone_model, xp=xp)
    except KeyError as e:
        raise KeyError(
            f"Controller `{controller}.{fn.__name__}` not found for drone `{drone_model}`"
        ) from e
    params = {k: xp.asarray(v, device=device) for k, v in params.items() if k in kwonly_params}
    return partial(fn, **params)


def load_params(
    controller: str, fn_name: str, drone_model: str, xp: ModuleType | None = None
) -> dict[str, Any]:
    """Load and merge controller parameters for a specific function.

    Reads ``drone_controllers/<controller>/params.toml`` and merges the
    ``[drone_model.core]`` section with the ``[drone_model.<fn_name>]`` section,
    with function-specific values taking precedence over core values.

    Args:
        controller: Name of the controller sub-package, e.g. ``"mellinger"``.
        fn_name: Name of the controller function, e.g. ``"state2attitude"``.
        drone_model: Name of the drone configuration, e.g. ``"cf2x_L250"``.
        xp: The array API module to use. If not provided, numpy is used.

    Returns:
        A flat dict mapping parameter names to arrays in the requested array namespace.

    Raises:
        KeyError: If ``drone_model`` is not found in the params.toml file.
    """
    xp = np if xp is None else xp
    with open(Path(__file__).parent / f"{controller}/params.toml", "rb") as f:
        params = tomllib.load(f)
    if drone_model not in params:
        raise KeyError(f"Drone model `{drone_model}` not found in {controller}/params.toml")
    model_params = params[drone_model]
    merged = model_params.get("core", {}) | model_params.get(fn_name, {})
    return {k: xp.asarray(v) for k, v in merged.items()}
