# Core

::: drone_controllers.core

The core module provides the foundational functionality for controller parametrization.

## Key Concepts

### Controller Parametrization

The `parametrize` function automatically configures a controller with parameters for a specific drone model by inspecting the function's keyword-only arguments and filling them from the corresponding TOML file:

```python
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

# Get a controller configured for the Crazyflie 2.x
controller = parametrize(state2attitude, "cf2x_L250")

# Use the controller (all parameters are automatically filled in)
rpyt, pos_err = controller(pos, quat, vel, cmd)
```

### Manual Parameter Loading

Use `load_params` to inspect or override parameters directly:

```python
from drone_controllers.core import load_params

params = load_params("mellinger", "state2attitude", "cf2x_L250")
print(params["mass"])   # 0.029
print(params["kp"])     # position gain array
```

### Array Namespace Support

Both `parametrize` and `load_params` accept an `xp` argument so that static parameters are placed in the correct array namespace before being bound to the function:

```python
import jax.numpy as jnp
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

controller = parametrize(state2attitude, "cf2x_L250", xp=jnp)
```
