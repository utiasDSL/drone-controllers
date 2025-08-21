"""Core functionalities for controller parametrization and registration."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, ParamSpec, Protocol, TypeVar, runtime_checkable

P = ParamSpec("P")
R = TypeVar("R")


controller_parameter_registry: dict[str, type[ControllerParams]] = {}


def parametrize(fn: Callable[P, R], drone_model: str) -> Callable[P, R]:
    """Parametrize a controller function with the default controller parameters for a drone model.

    Args:
        fn: The controller function to parametrize.
        drone_model: The drone model to use.

    Example:
        >>> from drone_models.controller import parametrize
        >>> from drone_models.controller.mellinger import state2attitude
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
    controller_id = fn.__module__ + "." + fn.__name__
    try:
        params = controller_parameter_registry[controller_id].load(drone_model)
    except KeyError as e:
        raise KeyError(
            f"Controller `{controller_id}` does not exist in the parameter registry"
        ) from e
    except ValueError as e:
        raise ValueError(f"Drone model `{drone_model}` not supported for `{fn.__name__}`") from e
    return partial(fn, **params._asdict())


@runtime_checkable
class ControllerParams(Protocol):
    """Protocol for controller parameters."""

    @staticmethod
    def load(drone_model: str) -> ControllerParams:
        """Load the parameters from the config file."""

    def _asdict(self) -> dict[str, Any]:
        """Convert the parameters to a dictionary."""


def register_controller_parameters(
    params: ControllerParams | type[ControllerParams],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register the default controller parameters for this controller.

    Warning:
        The controller parameters **must** be a named tuple with a function `load` that takes in the
        drone model name and returns an instance of itself, or a class that implements the
        ControllerParams protocol.

    Args:
        params: The controller parameter type.

    Returns:
        A decorator function that registers the parameters and returns the function unchanged.
    """
    if not isinstance(params, ControllerParams):
        raise ValueError(f"{params} does not implement the ControllerParams protocol")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        controller_id = fn.__module__ + "." + fn.__name__
        if controller_id in controller_parameter_registry:
            raise ValueError(f"Controller `{controller_id}` already registered")
        controller_parameter_registry[controller_id] = params
        return fn

    return decorator
