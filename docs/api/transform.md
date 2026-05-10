# Transform

::: drone_controllers.transform

The transform module provides utility functions for converting between different physical representations of quadrotor motor state.

## Motor force / rotor velocity

### `motor_force2rotor_vel`

Invert the quadratic thrust curve $f = a + b\,\omega + c\,\omega^2$ to recover rotor speed from motor force:

```python
import numpy as np
from drone_controllers.core import load_params
from drone_controllers.mellinger import force_torque2rotor_vel
from drone_controllers.transform import motor_force2rotor_vel

params = load_params(force_torque2rotor_vel, "cf2x_L250")
rpm2thrust = params["rpm2thrust"]

forces = np.array([0.05, 0.08, 0.10, 0.05])  # N per motor
rpms = motor_force2rotor_vel(forces, rpm2thrust)
rpms.shape  # (4,)
```

### `rotor_vel2body_force`

Sum rotor forces into a body-frame force vector. Only the z-component is nonzero for an X-frame quad:

```python
import numpy as np
from drone_controllers.core import load_params
from drone_controllers.mellinger import force_torque2rotor_vel
from drone_controllers.transform import rotor_vel2body_force

params = load_params(force_torque2rotor_vel, "cf2x_L250")
rpm2thrust = params["rpm2thrust"]

rotor_speeds = np.full(4, 12_000.)  # RPM
body_force = rotor_vel2body_force(rotor_speeds, rpm2thrust)
body_force.shape  # (3,) -- x, y are zero; z is total thrust
```

### `rotor_vel2body_torque`

Compute body-frame torques from individual rotor speeds using the mixing matrix.

## PWM conversions

### `force2pwm` / `pwm2force`

Linear conversion between thrust in Newtons and the PWM signal sent to the motors:

```python
import numpy as np
from drone_controllers import mellinger
from drone_controllers.core import load_core_params
from drone_controllers.transform import force2pwm, pwm2force

core = load_core_params(mellinger, "cf2x_L250")
thrust_max = float(core["thrust_max"])
pwm_max    = float(core["pwm_max"])

thrust = np.array([0.05, 0.10])
pwms   = force2pwm(thrust, thrust_max, pwm_max)
thrust_recovered = pwm2force(pwms, thrust_max, pwm_max)
np.allclose(thrust, thrust_recovered)  # True
```
