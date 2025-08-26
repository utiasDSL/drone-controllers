# Quick Start

This guide will get you up and running with Drone Controllers in just a few minutes.

## Basic Controller Usage

The library implements controllers as pure functions that can be parametrized for specific drone models. Here's how to use the Mellinger controller:

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude, attitude2force_torque, force_torque2rotor_vel

# Parametrize controllers for a specific drone model
state_ctrl = parametrize(state2attitude, "cf2x_L250")
attitude_ctrl = parametrize(attitude2force_torque, "cf2x_L250") 
rotor_ctrl = parametrize(force_torque2rotor_vel, "cf2x_L250")

# Define current state
pos = np.array([0.0, 0.0, 0.5])         # Current position [x, y, z]
quat = np.array([0.0, 0.0, 0.0, 1.0])   # Current quaternion [x, y, z, w]  
vel = np.array([0.0, 0.0, 0.0])         # Current velocity [vx, vy, vz]
ang_vel = np.array([0.0, 0.0, 0.0])     # Current angular velocity [wx, wy, wz]

# Define command (13 elements)
# [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
cmd = np.array([1.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Step 1: State to attitude control
rpyt_cmd, pos_error_integral = state_ctrl(pos, quat, vel, ang_vel, cmd)
print(f"Attitude command (R,P,Y,T): {rpyt_cmd}")

# Step 2: Attitude to force/torque
force, torque, att_error_integral = attitude_ctrl(pos, quat, vel, ang_vel, rpyt_cmd)
print(f"Desired force: {force[0]:.3f} N")
print(f"Desired torque: {torque}")

# Step 3: Force/torque to rotor velocities
rotor_speeds = rotor_ctrl(force, torque)
print(f"Rotor speeds: {rotor_speeds} rad/s")
```

## Working with Batches

All controllers support batching through broadcasting. You can process multiple drones or time steps simultaneously:

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

# Parametrize controller
controller = parametrize(state2attitude, "cf2x_L250")

# Batch processing: 3 drones, 5 time steps each
batch_shape = (3, 5)

# Create batch states (3 drones × 5 timesteps)
pos_batch = np.random.randn(*batch_shape, 3)
quat_batch = np.tile([0, 0, 0, 1], (*batch_shape, 1))  # Level attitude
vel_batch = np.random.randn(*batch_shape, 3) * 0.1
ang_vel_batch = np.random.randn(*batch_shape, 3) * 0.1

# Create batch commands
cmd_batch = np.zeros((*batch_shape, 13))
cmd_batch[..., :3] = pos_batch + np.random.randn(*batch_shape, 3) * 0.5  # Target positions

# Process entire batch at once
rpyt_batch, pos_err_batch = controller(pos_batch, quat_batch, vel_batch, ang_vel_batch, cmd_batch)

print(f"Batch output shape: {rpyt_batch.shape}")  # Should be (3, 5, 4)
print(f"Per-drone commands: {rpyt_batch[0, 0, :]}")  # First drone, first timestep
```

## Manual Parameter Loading

You can also load parameters manually without using the `parametrize` decorator:

```python
import numpy as np
from functools import partial
from drone_controllers.mellinger import state2attitude
from drone_controllers.mellinger.params import StateParams

# Load parameters manually
params = StateParams.load("cf2x_L250")
print(f"Position gains: {params.kp}")
print(f"Velocity gains: {params.kd}")
print(f"Drone mass: {params.mass} kg")

# Create controller with custom parameters
controller = partial(state2attitude, **params._asdict())

# Use as before
pos = np.array([0.0, 0.0, 1.0])
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.array([0.0, 0.0, 0.0])
ang_vel = np.array([0.0, 0.0, 0.0])
cmd = np.ones(13)

rpyt, pos_err = controller(pos, quat, vel, ang_vel, cmd, ctrl_freq=100)
```

## Array API Compatibility

The controllers work with different array libraries. Here's an example with JAX:

```python
import jax.numpy as jnp
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

# Create JAX arrays
pos = jnp.array([0.0, 0.0, 1.0])
quat = jnp.array([0.0, 0.0, 0.0, 1.0])
vel = jnp.array([0.0, 0.0, 0.0])
ang_vel = jnp.array([0.0, 0.0, 0.0])
cmd = jnp.ones(13)

# Parametrize controller (works with any array API)
controller = parametrize(state2attitude, "cf2x_L250")

# JIT compile for performance
from jax import jit
jit_controller = jit(controller)

rpyt, pos_err = jit_controller(pos, quat, vel, ang_vel, cmd)
print(f"Output type: {type(rpyt)}")  # JAX array
```

## Error Handling with Integral Terms

Controllers maintain integral errors for robustness. Here's how to handle them properly:

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

controller = parametrize(state2attitude, "cf2x_L250")

# Initialize state
pos = np.array([0.0, 0.0, 0.5])
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.array([0.0, 0.0, 0.0])
ang_vel = np.array([0.0, 0.0, 0.0])

# Target hover at 1m altitude
cmd = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# First call - no integral error history
rpyt1, pos_err_i1 = controller(pos, quat, vel, ang_vel, cmd, ctrl_errors=None)

# Subsequent calls - pass integral error from previous step
rpyt2, pos_err_i2 = controller(pos, quat, vel, ang_vel, cmd, ctrl_errors=(pos_err_i1,))

print(f"Integral error evolution: {np.linalg.norm(pos_err_i1)} -> {np.linalg.norm(pos_err_i2)}")
```

## Available Drone Models

Currently supported drone models:

```python
from drone_controllers.drones import Drones

# See all available models
for drone in Drones:
    print(f"Model: {drone.value}")

# Currently available:
# - cf2x_L250: Crazyflie 2.x with 250mm frame
```

## Next Steps

Now that you've seen the basics, explore:

- **[Concepts](../concepts/overview.md)** - Understand the theory behind the controllers
- **[API Reference](../api/core.md)** - Complete API documentation

## Common Issues

### Import Errors

If you get import errors, make sure you've installed the package correctly:

```bash
pip install -e .  # For development installation
```

### Array API Compatibility

If you encounter issues with array operations, ensure you're using compatible versions:

```bash
pip install numpy>=2.0.0 array-api-compat array-api-extra
```

### Parameter Loading Errors

If parameter loading fails, check that the drone model is supported:

```python
from drone_controllers.drones import Drones
print(list(Drones))  # Shows available models
```
