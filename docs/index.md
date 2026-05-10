# Drone Controllers

**Drone Controllers** is a Python library providing faithful reimplementations of onboard drone controllers for simulation and research.

- **Array API standard**: works with NumPy, JAX, PyTorch, and any other compliant library
- **Pure functions**: every controller is stateless and JIT-compilable
- **Batching**: arbitrary leading batch dimensions via broadcasting
- **Research-focused**: designed for quadrotor UAV control and sim-to-real transfer

## Quick example

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

pos  = np.zeros(3)
quat = np.array([0., 0., 0., 1.])
vel  = np.zeros(3)
cmd  = np.zeros(13)  # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rr, pr, yr]

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
```

## Supported controllers

| Controller | Module | Description |
|---|---|---|
| Mellinger | `drone_controllers.mellinger` | Geometric tracking controller based on the Crazyflie firmware [[1]](#references) |

Controllers are implemented as plain Python functions, so adding a new one requires no registration or subclassing. Contributions are welcome; see the [GitHub repository](https://github.com/learnsyslab/drone-controllers) to get started.

## Getting help

- [Get Started](getting-started/installation.md): install the package
- [User Guide](user-guide/controllers.md): how everything fits together
- [Examples](examples/index.md): progressive runnable examples
- [API Reference](api/core.md): complete function signatures
- [GitHub Issues](https://github.com/learnsyslab/drone-controllers/issues): bug reports and feature requests

## References

[1] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," ICRA 2011, doi: 10.1109/ICRA.2011.5980409.
