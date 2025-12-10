"""..."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_extra as xpx
from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

from drone_controllers.core import register_controller_parameters
from drone_controllers.mellinger.params import AttitudeParams, ForceTorqueParams, StateParams
from drone_controllers.transform import force2pwm, motor_force2rotor_vel, pwm2force

if TYPE_CHECKING:
    from drone_controllers._typing import Array  # To be changed to array_api_typing later


@register_controller_parameters(StateParams)
def state2attitude(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    ctrl_errors: tuple[Array, ...] | None = None,
    ctrl_info: tuple[Array, ...] | None = None,
    ctrl_freq: float = 100,
    *,
    mass: float,
    kp: Array,
    kd: Array,
    ki: Array,
    gravity_vec: Array,
    mass_thrust: float,
    int_err_max: Array,
    thrust_max: float,
    pwm_max: float,
) -> tuple[Array, Array]:
    """Compute the positional part of the mellinger controller.

    All controllers are implemented as pure functions. Therefore, integral errors have to be passed
    as an argument and returned as well.

    Args:
        pos: Drone position with shape (..., 3).
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        vel: Drone velocity with shape (..., 3).
        ang_vel: Drone angular drone velocity in rad/s with shape (..., 3).
        cmd: Full state command in SI units and rad with shape (..., 13). The entries are
            [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
        ctrl_errors: Tuple of integral errors. For state2attitude, the tuple contains a single array
            (..., 3) for the position integral error or is None.
        ctrl_info: Tuple of arrays with additional data. Not used in state2attitude.
        ctrl_freq: Control frequency in Hz
        mass: Drone mass used for calculations in the controller in kg.
        kp: Proportional gain for the position controller with shape (3,).
        kd: Derivative gain for the position controller with shape (3,).
        ki: Integral gain for the position controller with shape (3,).
        gravity_vec: Gravity vector with shape (3,). We assume gravity to be in the negative z
            direction. E.g., [0, 0, -9.81].
        mass_thrust: Conversion factor from thrust to PWM.
        int_err_max: Range of the integral error with shape (3,). i_range in the firmware.
        thrust_max: Maximum thrust in N.
        pwm_max: Maximum PWM value.

    Returns:
        The RPY collective thrust command [rad, rad, rad, N], and the integral error of the position
        controller.
    """
    xp = array_namespace(pos)

    setpoint_pos = cmd[..., 0:3]
    setpoint_vel = cmd[..., 3:6]
    setpoint_acc = cmd[..., 6:9]
    setpoint_yaw = cmd[..., 9]
    dt = 1 / ctrl_freq
    # setpointRPY_rates = cmd[..., 10:13]
    # From firmware controller_mellinger
    pos_err = setpoint_pos - pos  # l. 145 Position Error (ep)
    vel_err = setpoint_vel - vel  # l. 148 Velocity Error (ev)
    # l.151 ff Integral Error
    int_pos_err = xp.zeros_like(pos) if ctrl_errors is None else ctrl_errors[0]
    int_pos_err = xp.clip(int_pos_err + pos_err * dt, -int_err_max, int_err_max)
    # l. 161 Desired thrust [F_des]
    # => only one case here, since setpoint is always in absolute mode
    # Note: since we've defined the gravity in z direction, a "-" needs to be added
    target_thrust = (
        mass * (setpoint_acc - gravity_vec) + kp * pos_err + kd * vel_err + ki * int_pos_err
    )
    # l. 178 Rate-controlled YAW is moving YAW angle setpoint
    # => only one case here, since the setpoint is always in absolute mode
    desired_yaw = setpoint_yaw
    # l. 189 Z-Axis [zB]
    rot = R.from_quat(quat).as_matrix()
    z_axis = rot[..., -1]  # 3rd column or roation matrix is z axis
    # l. 194 yaw correction (only if position control is not used)
    # => skipped since we always use position control here

    # l. 204 Current thrust [F]
    # Taking the dot product of the last axis:
    current_thrust = xp.vecdot(target_thrust, z_axis, axis=-1)
    # l. 207 Calculate axis [zB_des]
    z_axis_desired = target_thrust / xp.linalg.vector_norm(target_thrust)
    # l. 210 [xC_des]
    # x_axis_desired = z_axis_desired x [sin(yaw), cos(yaw), 0]^T
    x_c_des_x = xp.cos(desired_yaw)
    x_c_des_y = xp.sin(desired_yaw)
    x_c_des_z = xp.zeros_like(x_c_des_x)
    x_c_des = xp.stack((x_c_des_x, x_c_des_y, x_c_des_z), axis=-1)
    # [yB_des]
    y_axis_desired = xp.linalg.cross(z_axis_desired, x_c_des)
    y_axis_desired = y_axis_desired / xp.linalg.vector_norm(y_axis_desired)
    # [xB_des]
    x_axis_desired = xp.linalg.cross(y_axis_desired, z_axis_desired)
    # converting desired axis to rotation matrix and then to RPY.
    matrix = xp.stack((x_axis_desired, y_axis_desired, z_axis_desired), axis=-1)
    # l. 220 [eR] The mellinger controller now continues with the attitude controller. However, we
    # decouple the attitude controller from the state controller. We therefore stop here and
    # continue the computation in the attitude2force_torque controller. The conversion to RPY is
    # necessary to pass the command to the attitude2force_torque controller in the correct format.
    command_RPY = R.from_matrix(matrix).as_euler("xyz", degrees=False)
    # l. 283 [control_thrust]
    # The firmware returns thrust in PWM, but we want to stay in SI units. The conversion from
    # thrust to PWM uses a mass_thrust parameter, which is a constant converting thrust values to
    # PWMs. This transformation changes the thrust value, because it is fixed to a specific value
    # instead of dynamically scaling with the mass parameter of the controller! Hence, we include
    # this conversion here and thus effectively rescale the thrust slightly. The conversion below
    # maps thrust -> PWM -> rescaled thrust.
    thrust = pwm2force(mass_thrust * current_thrust, thrust_max * 4, pwm_max)
    command_rpyt = xp.concat((command_RPY, thrust[..., None]), axis=-1)
    return command_rpyt, int_pos_err


@register_controller_parameters(AttitudeParams)
def attitude2force_torque(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    prev_ang_vel: Array | None = None,
    ctrl_errors: tuple[Array, ...] | None = None,
    ctrl_info: tuple[Array, ...] | None = None,
    ctrl_freq: int = 500,
    *,
    kR: Array,
    kw: Array,
    ki_m: Array,
    kd_omega: Array,
    int_err_max: Array,
    torque_pwm_max: Array,
    thrust_max: float,
    pwm_min: float,
    pwm_max: float,
    L: float,
    thrust2torque: float,
    mixing_matrix: Array,
) -> tuple[Array, Array, Array]:
    """Compute the attitude to desired force-torque part of the Mellinger controller.

    Note:
        We omit the axis flip in the firmware as it has only been introduced to make the controller
        compatible with the new frame of the Crazyflie 2.1.

    Args:
        pos: Drone position with shape (..., 3).
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        vel: Drone velocity with shape (..., 3).
        ang_vel: Drone angular drone velocity in rad/s with shape (..., 3).
        cmd: Commanded attitude (roll, pitch, yaw) and total thrust [rad, rad, rad, N].
        ctrl_errors: Tuple of integral errors. For attitude2force_torque, the tuple contains a
            single array (..., 3) for the angular velocity integral error or is None.
        ctrl_info: Tuple of arrays with additional data. Not used in attitude2force_torque.
        ctrl_freq: Control frequency in Hz
        kR: Proportional gain for the rotation error with shape (3,).
        kw: Proportional gain for the angular velocity error with shape (3,).
        ki_m: Integral gain for the rotation error with shape (3,).
        kd_omega: Derivative gain for the angular velocity error with shape (3,).
        int_err_max: Range of the integral error with shape (3,). i_range in the firmware.
        torque_pwm_max: Maximum torque in PWM.
        thrust_max: Maximum thrust in N.
        pwm_min: Minimum PWM value.
        pwm_max: Maximum PWM value.
        ang_vel_des: Desired angular velocity in rad/s.
        prev_ang_vel: Previous angular velocity in rad/s.
        prev_ang_vel_des: Previous angular velocity command in rad/s.
        L: Distance from the center of the quadrotor to the center of the rotor in m.
        thrust2torque: Conversion factor (m).
        mixing_matrix: Mixing matrix for the motor forces with shape (4, 3).

    Returns:
        4 Motor forces [N], i_error_m
    """
    xp = array_namespace(quat)
    force_des = cmd[..., 3]  # Total thrust in N
    rpy_des = cmd[..., :3]
    dt = 1 / ctrl_freq
    # l. 220 ff [eR]. We're using the "inefficient" code path from the firmware
    rot = R.from_quat(quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=False)
    # Equivalent to eRM = R_des.T @ R_act - R_act.T @ R_des
    # Firmware does not multiply by 0.5 here, but the original paper does. We replicate the firmware
    # exactly to avoid sim2real issues with the original controller parameters.
    R_delta = (rot_des.inv() * rot).as_matrix()
    eRM = R_delta - R_delta.mT
    # Vee operator (SO3 to R3)
    eR = xp.stack((eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1)
    # l.248 ff [ew]
    # Warning: We assume zero desired angular velocity
    ang_vel_des = xp.zeros_like(ang_vel)
    prev_ang_vel_des = xp.zeros_like(ang_vel)
    ew = ang_vel_des - ang_vel
    # WARNING: if the setpoint is ever != 0 => change sign of ew.y!

    # l.259 ff [err_d_rpy]
    prev_ang_vel = xp.zeros_like(ang_vel) if prev_ang_vel is None else prev_ang_vel
    ang_vel_d_err = ((ang_vel_des - prev_ang_vel_des) - (ang_vel - prev_ang_vel)) / dt
    # l.281: No err_d_yaw
    ang_vel_d_err = xpx.at(ang_vel_d_err)[..., 2].set(0)

    # l. 268 ff Integral Error
    r_int_error = xp.zeros_like(ang_vel) if ctrl_errors is None else ctrl_errors[0]
    r_int_error = r_int_error - eR * dt
    r_int_error = xp.clip(r_int_error, -int_err_max, int_err_max)
    # l. 278 ff Moment:
    torque_pwm = -kR * eR + kw * ew + ki_m * r_int_error + kd_omega * ang_vel_d_err
    # l. 297 ff
    torque_pwm = xp.clip(torque_pwm, -torque_pwm_max, torque_pwm_max)
    torque_pwm = xp.where((force_des > 0)[..., None], torque_pwm, 0.0)
    force_des_pwm = force2pwm(force_des / 4, thrust_max, pwm_max)
    pwms = force_torque_pwms2pwms(force_des_pwm, torque_pwm, mixing_matrix)
    pwms = xp.where(xp.all(pwms == 0), 0.0, xp.clip(pwms, pwm_min, pwm_max))

    # Info: The Mellinger controller in the firmware ends here. However, we enforce a standardized
    # interface in the simulation from states -> attitude -> force_torque. We therefore need this
    # function to convert from PWMs to forces and torques.
    # In the firmware, this is done implicitly with the motor mixing. We therefore do the motor
    # mixing here, calculate the resulting force and torque, and return them.
    # This process is then reversed in the next step, where we recover the desired motor forces from
    # the force and torque.
    motor_forces = pwm2force(pwms, thrust_max, pwm_max)
    # TODO: Long-term, the Mellinger controller should use the new power distribution which
    # calculates motor forces in Newtons. However, for now the firmware uses the legacy power
    # distribution, so we keep it here for compatibility. To have a single consistent interface for
    # controllers within drone_models, we still want to return SI forces and torques. We thus need
    # to convert the legacy output to SI units.
    # l. 310 ff
    torque_des = (mixing_matrix @ motor_forces[..., None])[..., 0] * xp.stack([L, L, thrust2torque])
    force_des = xp.sum(motor_forces, axis=-1)[..., None]
    return force_des, torque_des, r_int_error


def force_torque_pwms2pwms(force_pwm: Array, torque_pwm: Array, mixing_matrix: Array) -> Array:
    """Convert desired collective thrust and torques to rotor speeds using legacy behavior."""
    xp = array_namespace(force_pwm)
    torque_pwm = xp.concatenate((torque_pwm[..., :2] / 2, torque_pwm[..., 2:]), axis=-1)
    return force_pwm[..., None] + (torque_pwm @ mixing_matrix)


@register_controller_parameters(ForceTorqueParams)
def force_torque2rotor_vel(
    force: Array,
    torque: Array,
    *,
    thrust_min: float,
    thrust_max: float,
    L: float,
    rpm2thrust: Array,
    thrust2torque: float,
    mixing_matrix: Array,
) -> Array:
    """Convert desired collective thrust and torques to rotor speeds.

    The firmware calculates PWMs for each motor, compensates for the battery voltage, and then
    applies the modified PWMs to the motors. We assume perfect battery compensation here, skip the
    PWM interface except for clipping, and instead return desired motor forces.

    Note:
        The equivalent function in the crazyflie firmware is power_distribution from
        power_distribution_quadrotor.c.

    Warning:
        This function assumes an X rotor configuration.

    Args:
        force: Desired thrust in SI units with shape (...,).
        torque: Desired torque in SI units with shape (..., 3).
        thrust_min: Minimum thrust in N.
        thrust_max: Maximum thrust in N.
        L: Distance from the center of the quadrotor to the center of the rotor in m.
        rpm2thrust: Force constants (N/RPM, N/RPM**2).
        thrust2torque: Conversion factor (m).
        mixing_matrix: Mixing matrix for the motor forces with shape (4, 3).

    Returns:
        The desired motor forces in SI units with shape (..., 4).
    """
    xp = array_namespace(torque)
    assert torque.shape[-1] == 3, f"Torque must have shape (..., 3), but has {torque.shape}"
    assert force.shape[-1] == 1, f"Force must have shape (..., 1), but has {force.shape}"
    torque_forces = (torque * xp.asarray([1 / L, 1 / L, 1 / thrust2torque])) @ mixing_matrix
    motor_forces = (torque_forces + force) / 4
    # Clip motor forces on the thrust instead of PWM level.
    motor_forces = xp.where(xp.all(force == 0), 0.0, xp.clip(motor_forces, thrust_min, thrust_max))
    # Assume perfect battery compensation and calculate the desired motor speeds directly
    return motor_force2rotor_vel(motor_forces, rpm2thrust)
