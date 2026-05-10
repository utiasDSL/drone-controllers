# Examples

These examples build on each other; each one introduces one new idea on top of the previous. Start from the top if you're new, or jump to the section you need.

## Single stage with NumPy

The simplest starting point: use `parametrize` to bind the drone's physical parameters to `state2attitude`, then call it with a hovering state and a zero setpoint. The drone is at the origin, upright, with no velocity.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

pos  = np.zeros(3)
quat = np.array([0., 0., 0., 1.])  # xyzw — identity (no rotation)
vel  = np.zeros(3)
cmd  = np.zeros(13)                 # setpoint at origin, yaw = 0

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
```

`rpyt[:3]` is the attitude correction in radians; `rpyt[3]` is the collective thrust in Newtons. At the setpoint with no error, RPY is `[0, 0, 0]` and thrust is the hover thrust.

## Full pipeline

Chain all three stages to convert a full-state setpoint into individual motor speeds. Each stage feeds directly into the next.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)

state_ctrl = parametrize(state2attitude, "cf2x_L250")
att_ctrl   = parametrize(attitude2force_torque, "cf2x_L250")
rotor_ctrl = parametrize(force_torque2rotor_vel, "cf2x_L250")

pos     = np.array([0., 0., 1.])      # 1 m altitude
quat    = np.array([0., 0., 0., 1.])
vel     = np.zeros(3)
ang_vel = np.zeros(3)
cmd     = np.zeros(13)
cmd[:3] = np.array([0., 0., 1.])      # hover at 1 m

rpyt, _          = state_ctrl(pos, quat, vel, cmd)
force, torque, _ = att_ctrl(quat, ang_vel, rpyt)
rotor_speeds     = rotor_ctrl(force, torque)
rotor_speeds.shape  # (4,)
```

## Batched evaluation

Add a leading batch dimension to evaluate many states in one call. The controller broadcasts identically to the scalar case.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

N    = 64
pos  = np.random.randn(N, 3) * 0.5
quat = np.tile(np.array([0., 0., 0., 1.]), (N, 1))
vel  = np.zeros((N, 3))
cmd  = np.zeros((N, 13))

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
rpyt.shape  # (64, 4)
```

## Integral error loop

Controllers are pure functions; integral state is an explicit return value. Pass it back as `ctrl_errors` on every subsequent call to accumulate the integral term.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import attitude2force_torque, state2attitude

state_ctrl = parametrize(state2attitude, "cf2x_L250")
att_ctrl   = parametrize(attitude2force_torque, "cf2x_L250")

pos     = np.zeros(3)
quat    = np.array([0., 0., 0., 1.])
vel     = np.zeros(3)
ang_vel = np.zeros(3)
cmd     = np.zeros(13)
cmd[:3] = np.array([1., 0., 1.])  # 1 m offset in x and z

state_errs = None
att_errs   = None

for step in range(20):
    rpyt, int_pos_err = state_ctrl(pos, quat, vel, cmd, ctrl_errors=state_errs)
    state_errs = (int_pos_err,)

    force, torque, r_int_err = att_ctrl(quat, ang_vel, rpyt, ctrl_errors=att_errs)
    att_errs = (r_int_err,)

# Integral errors have been accumulating for 20 steps.
```

## Switching backends

`parametrize` accepts an `xp` keyword to convert parameters upfront into any Array API-compatible framework, avoiding per-call conversion overhead.

```python
import torch
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

# Parameters stored as torch tensors.
ctrl = parametrize(state2attitude, "cf2x_L250", xp=torch)

pos  = torch.zeros(3)
quat = torch.tensor([0., 0., 0., 1.])
vel  = torch.zeros(3)
cmd  = torch.zeros(13)

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
type(rpyt)  # <class 'torch.Tensor'>
```
