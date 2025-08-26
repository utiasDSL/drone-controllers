# Core

::: drone_controllers.core

The core module provides the foundational functionality for controller parametrization and registration.

## Key Concepts

### Controller Parametrization

The `parametrize` function allows you to automatically configure controllers with parameters for specific drone models:

```python
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

# Get a controller configured for the Crazyflie 2.x
controller = parametrize(state2attitude, "cf2x_L250")

# Use the controller (all parameters are automatically filled in)
rpyt, pos_err = controller(pos, quat, vel, ang_vel, cmd)
```

### Parameter Registry

Controllers register their parameter types using the `@register_controller_parameters` decorator:

```python
@register_controller_parameters(MyControllerParams)
def my_controller(pos, vel, *, param1, param2, param3):
    # Controller implementation
    pass
```

### ControllerParams Protocol

All controller parameter classes must implement the `ControllerParams` protocol:

- `load(drone_model: str)` - Load parameters for a specific drone model
- `_asdict()` - Convert parameters to a dictionary

## Example Usage

```python
from functools import partial
from drone_controllers.mellinger.params import StateParams

# Manual parameter loading
params = StateParams.load("cf2x_L250")
controller = partial(state2attitude, **params._asdict())

# Equivalent to using parametrize
from drone_controllers import parametrize
controller = parametrize(state2attitude, "cf2x_L250")
```
