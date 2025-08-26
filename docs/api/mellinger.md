# Mellinger Controller

The Mellinger controller is a geometric tracking controller originally developed for aggressive quadrotor maneuvers [[1]](#references). This implementation is based on the Crazyflie firmware version.

## Controller Functions

### state2attitude

::: drone_controllers.mellinger.control.state2attitude

The position control component of the Mellinger controller. Converts desired position, velocity, and acceleration into attitude commands.

**Example:**
```python
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude

controller = parametrize(state2attitude, "cf2x_L250")

rpyt, pos_err_i = controller(pos, quat, vel, ang_vel, cmd)
```

### attitude2force_torque

::: drone_controllers.mellinger.control.attitude2force_torque

The attitude control component that converts attitude commands into desired forces and torques.

**Example:**
```python
from drone_controllers import parametrize
from drone_controllers.mellinger import attitude2force_torque

controller = parametrize(attitude2force_torque, "cf2x_L250")

force, torque, att_err_i = controller(pos, quat, vel, ang_vel, rpyt_cmd)
```

### force_torque2rotor_vel

::: drone_controllers.mellinger.control.force_torque2rotor_vel

Converts desired forces and torques into individual rotor velocities using the quadrotor mixing matrix.

**Example:**
```python
from drone_controllers import parametrize
from drone_controllers.mellinger import force_torque2rotor_vel

controller = parametrize(force_torque2rotor_vel, "cf2x_L250")

rotor_speeds = controller(force, torque)
```

## Parameter Classes

### StateParams

::: drone_controllers.mellinger.params.StateParams

Parameters for the position control loop.

### AttitudeParams

::: drone_controllers.mellinger.params.AttitudeParams

Parameters for the attitude control loop.

### ForceTorqueParams

::: drone_controllers.mellinger.params.ForceTorqueParams

Parameters for the force/torque to rotor speed conversion.

## Complete Controller Pipeline

Here's how to use all three components together:

```python
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import (
    state2attitude, 
    attitude2force_torque, 
    force_torque2rotor_vel
)

# Parametrize all three controller components
state_ctrl = parametrize(state2attitude, "cf2x_L250")
attitude_ctrl = parametrize(attitude2force_torque, "cf2x_L250")
rotor_ctrl = parametrize(force_torque2rotor_vel, "cf2x_L250")

# Define state
pos = np.array([0.0, 0.0, 1.0])
quat = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
vel = np.array([0.0, 0.0, 0.0])
ang_vel = np.array([0.0, 0.0, 0.0])

# Define command: [x, y, z, vx, vy, vz, ax, ay, az, yaw, r_rate, p_rate, y_rate]
cmd = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Run the complete pipeline
rpyt, pos_err_i = state_ctrl(pos, quat, vel, ang_vel, cmd)
force, torque, att_err_i = attitude_ctrl(pos, quat, vel, ang_vel, rpyt)
rotor_speeds = rotor_ctrl(force, torque)

print(f"Final rotor speeds: {rotor_speeds} rad/s")
```

## Integral Error Handling

The Mellinger controller uses integral terms for robustness. You must pass integral errors between calls:

```python
# Initialize
pos_err_i = None
att_err_i = None

for step in range(100):
    # Update state and command...
    
    # Pass previous integral errors
    ctrl_errors = (pos_err_i,) if pos_err_i is not None else None
    rpyt, pos_err_i = state_ctrl(pos, quat, vel, ang_vel, cmd, ctrl_errors=ctrl_errors)
    
    ctrl_errors = (att_err_i,) if att_err_i is not None else None  
    force, torque, att_err_i = attitude_ctrl(pos, quat, vel, ang_vel, rpyt, ctrl_errors=ctrl_errors)
    
    rotor_speeds = rotor_ctrl(force, torque)
```

## Array API Compatibility

All Mellinger functions support the Python Array API and can be used with JAX, PyTorch, etc.:

```python
import jax.numpy as jnp
from jax import jit

# Convert to JAX arrays
pos_jax = jnp.array([0.0, 0.0, 1.0])
quat_jax = jnp.array([0.0, 0.0, 0.0, 1.0])
# ... other arrays

# JIT compile the controller
jit_controller = jit(parametrize(state2attitude, "cf2x_L250"))

rpyt, pos_err_i = jit_controller(pos_jax, quat_jax, vel_jax, ang_vel_jax, cmd_jax)
```

# References

[1] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," 2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011, pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409.
