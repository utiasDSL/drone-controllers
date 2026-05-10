```math
\huge \displaystyle u = K_p e + K_i \int e \, dt + K_d \frac{de}{dt}
```

--------------------------------------------------------------------------------
<div align="center">

  **Faithful reimplementations of onboard drone controllers for simulation and research.**

  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
  [![Tests](https://github.com/learnsyslab/drone-controllers/actions/workflows/testing.yml/badge.svg)](https://github.com/learnsyslab/drone-controllers/actions/workflows/testing.yml)
  [![Ruff](https://github.com/learnsyslab/drone-controllers/actions/workflows/ruff.yml/badge.svg)](https://github.com/learnsyslab/drone-controllers/actions/workflows/ruff.yml)
  [![Docs](https://github.com/learnsyslab/drone-controllers/actions/workflows/docs.yml/badge.svg)](https://learnsyslab.github.io/drone-controllers/)

</div>

Drone Controllers provides array-API implementations of onboard drone controllers for use in simulation and research. It powers [Crazyflow](https://github.com/learnsyslab/crazyflow)'s onboard controller simulation and can be used standalone with NumPy, JAX, PyTorch, or any other compliant library.

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

pos  = np.zeros(3)
quat = np.array([0., 0., 0., 1.])  # xyzw, identity (no rotation)
vel  = np.zeros(3)
cmd  = np.zeros(13)  # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rr, pr, yr]

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
```

## Documentation

[learnsyslab.github.io/drone-controllers](https://learnsyslab.github.io/drone-controllers) — installation, user guide, examples, and API reference.

## Features

- **Array API standard**: works identically with NumPy, JAX, PyTorch, and any compliant library
- **Pure functions**: no hidden state; integral errors are explicit return values you pass back each step
- **JIT compatible**: compile any controller with `jax.jit` or `torch.compile` without modification
- **Batching**: add any number of leading batch dimensions; no loops, no special API

## Installation

```
pip install drone-controllers
```

Developer install with editable mode ([pixi](https://pixi.sh/) required):

```
git clone https://github.com/learnsyslab/drone-controllers.git
cd drone-controllers
pixi shell
```

## Controllers

### Mellinger

A geometric tracking controller based on the Crazyflie firmware, split into three composable stages:

| Stage | Function | Input | Output |
|---|---|---|---|
| 1 | `state2attitude` | State + 13-element setpoint | RPYT command + position integral error |
| 2 | `attitude2force_torque` | Attitude + RPYT command | Collective force, body torques + angular velocity integral error |
| 3 | `force_torque2rotor_vel` | Force + torques | 4 motor speeds [RPM] |

Each stage can be used independently or chained into a full pipeline from state setpoint to motor speeds.

### Drone models

Pre-fitted parameters ship for the following platforms:

| `drone_model` | Platform |
|---|---|
| `"cf2x_L250"` | Crazyflie 2.x, L250 props |
| `"cf2x_P250"` | Crazyflie 2.x, P250 props |
| `"cf2x_T350"` | Crazyflie 2.x, T350 props |
| `"cf21B_500"` | Crazyflie 2.1 Brushless, 500 props |

## Related packages

| Package | Description |
|---|---|
| [crazyflow](https://github.com/learnsyslab/crazyflow) | GPU-accelerated differentiable drone simulator in JAX. Uses drone-controllers to provide its onboard controller simulation. |
| [drone-models](https://github.com/learnsyslab/drone-models) | Drone dynamics models (first-principles and fitted) compatible with NumPy, JAX, and PyTorch. |

## Testing

```
pixi run -e tests tests       # unit tests
pixi run -e tests test-docs   # doc snippet tests
```
