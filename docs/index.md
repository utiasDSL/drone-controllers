# Drone Controllers

$$
u = K_p e + K_i \int e \, dt + K_d \frac{de}{dt}
$$


[![CI](https://github.com/utiasDSL/drone-controllers/actions/workflows/testing.yml/badge.svg)](https://github.com/utiasDSL/drone-controllers/actions)
[![License](https://img.shields.io/github/license/utiasDSL/drone-controllers)](https://github.com/utiasDSL/drone-controllers/blob/main/LICENSE)

**Drone Controllers** is a Python library providing faithful reimplementations of onboard drone controllers that can be used for simulation and modelling.

## Why use Drone Controllers?

- **Array API Standard** — Controllers work with NumPy, JAX, PyTorch, and other array libraries
- **Pure Functions** — All controllers are implemented as pure functions for easy JIT compilation
- **Batching Support** — Built-in support for arbitrary batch dimensions via broadcasting
- **Research-focused** — Designed specifically for robotics research with quadrotor UAVs
- **Type-safe** — Full type hints for better development experience

## Quick Start

[Installation](getting-started/installation.md) is simple with pip:

```bash
pip install drone-controllers
```

Here's a basic example using the Mellinger controller:

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

# Get controller parameters for a specific drone model
controller = parametrize(state2attitude, "cf2x_L250")

# Define state and command
pos = np.array([0.0, 0.0, 1.0])  # position [x, y, z]
quat = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion [x, y, z, w]
vel = np.array([0.0, 0.0, 0.0])  # velocity [vx, vy, vz]
ang_vel = np.array([0.0, 0.0, 0.0])  # angular velocity [wx, wy, wz]

# Command: [x, y, z, vx, vy, vz, ax, ay, az, yaw, r_rate, p_rate, y_rate]
cmd = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Compute control output
rpyt, pos_err_i = controller(pos, quat, vel, ang_vel, cmd)
print(f"Roll-Pitch-Yaw-Thrust command: {rpyt}")
```

## Key Features

### Implemented Controllers

- **[Mellinger Controller](api/drone_controllers/mellinger/control.md)** — Geometric tracking controller based on the original Crazyflie implementation

### Supported Drone Models

- **cf2x_L250** — Crazyflie 2.x with 250mm frame
- More models coming soon!

### Core Functionality

- **[Parameter System](api/core.md)** — Automatic controller parametrization for different drone models
- **[Transform utilities](api/transform.md)** — Conversions between motor forces, rotor speeds, and PWM
- **[Drone registry](api/drones.md)** — Centralized catalog of supported drone platforms

## Controller Architecture

The library implements controllers as a pipeline of pure functions:

1. **State → Attitude**: Convert desired state to attitude commands
2. **Attitude → Force/Torque**: Convert attitude commands to desired forces and torques  
3. **Force/Torque → Rotor Speeds**: Convert desired forces/torques to individual motor commands

This modular design allows mixing and matching different components while maintaining compatibility.

## Array API Compatibility

All controllers support the Python Array API standard, meaning you can use them with:

- **NumPy** — Standard numerical computing
- **JAX** — JIT compilation and automatic differentiation
- **PyTorch** — Deep learning integration
- **CuPy** — GPU acceleration

## Getting Help

- Read the [Getting Started](getting-started/installation.md) guide
- Browse the [API Reference](api/core.md)  
- Check out [Concepts](concepts/overview.md) for theory
- Report issues on [GitHub](https://github.com/utiasDSL/drone-controllers/issues)
