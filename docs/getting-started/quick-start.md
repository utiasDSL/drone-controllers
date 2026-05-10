# Quick Start

This page walks through the minimal workflow: bind parameters to a drone, set up a state, and call the first stage of the Mellinger controller.

## State and command

Every controller stage shares the same state representation:

| Variable | Shape | Description |
|---|---|---|
| `pos` | `(3,)` | Position in world frame [m] |
| `quat` | `(4,)` | Attitude as unit quaternion, scalar-last `xyzw` |
| `vel` | `(3,)` | Linear velocity in world frame [m/s] |

The command `cmd` for `state2attitude` is a 13-element array:
`[x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]` in SI units and radians.

## Evaluate a controller

`parametrize` loads the physical parameters for a specific drone and returns a `functools.partial` with those parameters pre-filled. You then call it with just the state and command.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, drone_model="cf2x_L250")

# State: hovering at origin, upright, stationary.
pos  = np.zeros(3)
quat = np.array([0., 0., 0., 1.])  # xyzw, identity (no rotation)
vel  = np.zeros(3)

# Command: setpoint at origin, zero velocity and acceleration, yaw = 0.
cmd = np.zeros(13)

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
```

## Outputs

`state2attitude` returns two arrays:

| Return | Shape | Description |
|---|---|---|
| `rpyt` | `(4,)` | Attitude + thrust command: `[roll_rad, pitch_rad, yaw_rad, thrust_N]` |
| `int_pos_err` | `(3,)` | Position integral error; pass back next call to accumulate |

## Next steps

- [Controllers](../user-guide/controllers.md): all three Mellinger stages and how they chain
- [Mellinger](../user-guide/mellinger.md): input/output tables for every stage
- [Parametrize](../user-guide/parametrize.md): available drone models and array backends
- [Batching](../user-guide/batching.md): vectorized evaluation over many drones
- [Integral Errors](../user-guide/integral-errors.md): carrying state across timesteps
