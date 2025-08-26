# Transform

::: drone_controllers.transform

The transform module provides utility functions for converting between different physical representations of quadrotor parameters.

## Motor Force Conversions

### motor_force2rotor_vel

Convert motor forces to rotor velocities using the thrust coefficient:

```python
import numpy as np
from drone_controllers.transform import motor_force2rotor_vel

motor_forces = np.array([0.1, 0.1, 0.1, 0.1])  # N
kf = 3.16e-10  # Thrust coefficient
rotor_speeds = motor_force2rotor_vel(motor_forces, kf)
print(f"Rotor speeds: {rotor_speeds} rad/s")
```

### rotor_vel2body_force

Convert rotor velocities to body forces:

```python
from drone_controllers.transform import rotor_vel2body_force

rotor_speeds = np.array([1000, 1000, 1000, 1000])  # rad/s
kf = 3.16e-10
body_force = rotor_vel2body_force(rotor_speeds, kf)  # Returns [0, 0, total_thrust]
```

### rotor_vel2body_torque

Convert rotor velocities to body torques using mixing matrix:

```python
from drone_controllers.transform import rotor_vel2body_torque

rotor_speeds = np.array([1000, 1000, 1000, 1000])  # rad/s
kf = 3.16e-10
km = 7.94e-12
L = 0.046  # Arm length in meters
mixing_matrix = np.array([[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]])

body_torque = rotor_vel2body_torque(rotor_speeds, kf, km, L, mixing_matrix)
```

## PWM Conversions

### force2pwm

Convert thrust forces to PWM values:

```python
from drone_controllers.transform import force2pwm

thrust = 0.4  # N
thrust_max = 0.6  # N
pwm_max = 65535

pwm_value = force2pwm(thrust, thrust_max, pwm_max)
print(f"PWM value: {pwm_value}")
```

### pwm2force

Convert PWM values back to thrust forces:

```python
from drone_controllers.transform import pwm2force

pwm_value = 43690
thrust_max = 0.6  # N  
pwm_max = 65535

thrust = pwm2force(pwm_value, thrust_max, pwm_max)
print(f"Thrust: {thrust} N")
```

## Array API Compatibility

All transform functions work with the array API standard and support broadcasting:

```python
import jax.numpy as jnp
from drone_controllers.transform import motor_force2rotor_vel

# Works with JAX arrays
motor_forces_jax = jnp.array([[0.1, 0.1, 0.1, 0.1], 
                              [0.2, 0.2, 0.2, 0.2]])  # Batch of 2
kf = 3.16e-10

rotor_speeds = motor_force2rotor_vel(motor_forces_jax, kf)
print(f"Batch output shape: {rotor_speeds.shape}")  # (2, 4)
```

## Usage in Controller Pipeline

These transforms are typically used at the end of the control pipeline:

```python
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude, attitude2force_torque, force_torque2rotor_vel

# Set up controller pipeline
state_ctrl = parametrize(state2attitude, "cf2x_L250")
attitude_ctrl = parametrize(attitude2force_torque, "cf2x_L250")
rotor_ctrl = parametrize(force_torque2rotor_vel, "cf2x_L250")

# Run full pipeline
rpyt, _ = state_ctrl(pos, quat, vel, ang_vel, cmd)
force, torque, _ = attitude_ctrl(pos, quat, vel, ang_vel, rpyt)
rotor_speeds = rotor_ctrl(force, torque)

# Convert to actual motor forces if needed
from drone_controllers.transform import motor_force2rotor_vel
# The rotor_ctrl already returns rotor speeds, but if you had forces:
# rotor_speeds = motor_force2rotor_vel(motor_forces, kf=3.16e-10)
```

The transform module provides utilities for coordinate transformations and rotation representations commonly used in drone control.

## Rotation Conversions

### Rotation Matrices

Functions for working with 3x3 rotation matrices:

```python
from drone_controllers.transform import (
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    matrix_to_euler, euler_to_matrix
)
import numpy as np

# Elementary rotations
R_x = rotation_matrix_x(np.pi/4)  # 45° about x-axis
R_y = rotation_matrix_y(np.pi/6)  # 30° about y-axis  
R_z = rotation_matrix_z(np.pi/3)  # 60° about z-axis

# Combined rotation
R_combined = R_z @ R_y @ R_x
```

### Euler Angles

Convert between rotation matrices and Euler angles:

```python
# Convert rotation matrix to Euler angles (ZYX convention)
roll, pitch, yaw = matrix_to_euler(rotation_matrix)

# Convert Euler angles to rotation matrix
R = euler_to_matrix(roll, pitch, yaw)
```

### Quaternions

Quaternion operations for attitude representation:

```python
from drone_controllers.transform import (
    quaternion_to_matrix, matrix_to_quaternion,
    quaternion_multiply, quaternion_conjugate
)

# Convert between quaternions and rotation matrices
quat = matrix_to_quaternion(rotation_matrix)
R = quaternion_to_matrix(quat)

# Quaternion operations
q1 = np.array([0.7071, 0, 0, 0.7071])  # 90° about z-axis
q2 = np.array([0.7071, 0.7071, 0, 0])  # 90° about x-axis
q_combined = quaternion_multiply(q1, q2)
```

## Vector Operations

### Cross Product Matrix

Generate skew-symmetric matrices for cross products:

```python
from drone_controllers.transform import skew_symmetric

vector = np.array([1, 2, 3])
skew_matrix = skew_symmetric(vector)

# Now: skew_matrix @ other_vector == np.cross(vector, other_vector)
```

### Vector Projections

Project vectors onto planes or other vectors:

```python
from drone_controllers.transform import project_vector_onto_plane

vector = np.array([1, 1, 1])
plane_normal = np.array([0, 0, 1])  # XY plane
projected = project_vector_onto_plane(vector, plane_normal)
```

## Coordinate Frame Transformations

### Frame Conversions

Transform vectors between different coordinate frames:

```python
from drone_controllers.transform import transform_vector

# Transform vector from body to world frame
vector_body = np.array([1, 0, 0])  # Forward in body frame
rotation_body_to_world = euler_to_matrix(0, np.pi/4, 0)  # 45° pitch
vector_world = transform_vector(vector_body, rotation_body_to_world)
```

### Common Frames

Utilities for standard aerospace coordinate frames:

```python
from drone_controllers.transform import (
    ned_to_enu, enu_to_ned,
    body_to_world, world_to_body
)

# Convert between NED and ENU conventions
position_ned = np.array([10, 5, -2])  # North, East, Down
position_enu = ned_to_enu(position_ned)  # East, North, Up
```

## Angular Velocity Operations

### Integration on SO(3)

Integrate angular velocities on the rotation group:

```python
from drone_controllers.transform import integrate_angular_velocity

current_attitude = np.eye(3)
angular_velocity = np.array([0.1, 0.2, 0.05])  # rad/s
dt = 0.01

new_attitude = integrate_angular_velocity(
    current_attitude, angular_velocity, dt
)
```

### Rodrigues Formula

Direct integration using Rodrigues' rotation formula:

```python
from drone_controllers.transform import rodrigues_rotation

axis = np.array([0, 0, 1])  # Rotation axis
angle = np.pi/2  # 90 degrees
rotation_matrix = rodrigues_rotation(axis, angle)
```

## Utility Functions

### Angle Normalization

Normalize angles to standard ranges:

```python
from drone_controllers.transform import normalize_angle, wrap_to_pi

angle = 3 * np.pi  # Large angle
normalized = normalize_angle(angle)  # Wrapped to [0, 2π)
wrapped = wrap_to_pi(angle)  # Wrapped to [-π, π]
```

### Rotation Composition

Compose multiple rotations efficiently:

```python
from drone_controllers.transform import compose_rotations

rotations = [R1, R2, R3]  # List of rotation matrices
composed = compose_rotations(rotations)  # Equivalent to R3 @ R2 @ R1
```

## Constants and Conventions

The module defines standard constants and conventions:

```python
from drone_controllers.transform import (
    GRAVITY_EARTH,  # Standard gravity (9.80665 m/s²)
    IDENTITY_ROTATION,  # 3x3 identity matrix
    ZERO_VECTOR,  # Zero 3D vector
)
```

## Performance Notes

- Rotation matrices are preferred for computational efficiency
- Quaternions are used when interpolation is needed
- Euler angles are primarily for human-readable output
- All functions support vectorized operations where possible
