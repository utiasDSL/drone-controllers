# Integral Errors

The Mellinger controller uses integral terms to reject steady-state errors. Because every controller in this package is a **pure function**, integral state cannot be stored inside the function; it is an explicit return value that you pass back on the next call.

## Initialisation

Pass `ctrl_errors=None` on the first call. The controller initialises the integral error to zero internally.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
pos  = np.zeros(3)
quat = np.array([0., 0., 0., 1.])
vel  = np.zeros(3)
cmd  = np.zeros(13)

# First call — no integral history.
rpyt, int_pos_err = ctrl(pos, quat, vel, cmd, ctrl_errors=None)
```

## Carrying errors across timesteps

Wrap the returned error in a tuple and pass it as `ctrl_errors` on the next call:

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
pos  = np.zeros(3)
quat = np.array([0., 0., 0., 1.])
vel  = np.zeros(3)
cmd  = np.zeros(13)
cmd[0] = 1.0  # 1 m setpoint error in x

ctrl_errors = None
for _ in range(10):
    rpyt, int_pos_err = ctrl(pos, quat, vel, cmd, ctrl_errors=ctrl_errors)
    ctrl_errors = (int_pos_err,)  # carry forward

# After 10 steps at 100 Hz, integral error ≈ 10 × 0.01 = 0.1 m
```

## Both stages have integral errors

`state2attitude` tracks position error; `attitude2force_torque` tracks angular velocity error. Manage them independently:

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

state_errs = None
att_errs   = None

for _ in range(5):
    rpyt, int_pos_err = state_ctrl(pos, quat, vel, cmd, ctrl_errors=state_errs)
    state_errs = (int_pos_err,)

    force, torque, r_int_err = att_ctrl(quat, ang_vel, rpyt, ctrl_errors=att_errs)
    att_errs = (r_int_err,)
```
