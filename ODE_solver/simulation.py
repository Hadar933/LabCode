from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

MASS = 2.6e-4  # from the hummingbird paper
WING_LENGTH = 0.07  # meters
AERODYNAMIC_CENTER = 0.7 * WING_LENGTH
GYRATION_RADIUS = 0.6 * WING_LENGTH  # we use this for moment of inertia
MoI = MASS * GYRATION_RADIUS ** 2
AIR_DENSITY = 1.2  # From Arion's simulatio
WING_AREA = 0.5 * WING_LENGTH * (0.5 * WING_LENGTH) * np.pi  # 1/2 ellipse with minor radios ~ 1/2 major = length/2
# drag coefficients from whitney & wood (JFM 2010):
C_D_MAX = 3.4
C_D_0 = 0.4
C_L_MAX = 1.8
ZERO_CROSSING = 1
RADIAN45 = np.pi / 4
RADIAN135 = 3 * RADIAN45


def phi_dot_zero_crossing_event(t, y):
    """
    this event is given to solve_ivp to track if phi_dot == 0
    :param t: unused time variable
    :param y: a vector of [phi,phi_dot]
    :return:
    """
    return y[1]


class RobotSimulation:
    def __init__(self,
                 phi0: float, phi_dot0: float,
                 start_t: float, end_t: float,
                 motor_torque: Callable = lambda x: 0,
                 alpha: float = RADIAN45,
                 solve_ivp_time_increments: float = 0.001) -> None:
        """
        :param motor_torque: A function that returns float that represents the current torque provided by the motor
        :param phi0: initial phi position of the current time window
        :param phi_dot0: initial ang. velocity of the current time window
        :param start_t: start time of the current window
        :param end_t: end time of the current window
        :param solve_ivp_time_increments: the time increments
        """
        self.solution = 0
        self.motor_torque = motor_torque
        self.alpha = alpha  # angle of attack
        self.phi0 = phi0
        self.phi_dot0 = phi_dot0
        self.start_t = start_t
        self.end_t = end_t
        self.solve_ivp_time_increments = solve_ivp_time_increments

    def set_time(self, new_start: float, new_end: float) -> None:
        """
        we update the start time to be the end of the prev time and the end time
        as the last end time + the window size we wish
        """
        self.start_t = new_start
        self.end_t = new_end

    def set_init_cond(self, new_phi0: float, new_phi_dot0: float) -> None:
        """
        we update the initial conditions to be the last value of previous iterations
        """
        self.phi0 = new_phi0
        self.phi_dot0 = new_phi_dot0

    def set_motor_torque(self, new_motor_torque: Callable) -> None:
        """
        sets the moment to new value, based on the action the learning algorithm provided
        """
        self.motor_torque = new_motor_torque

    def flip_alpha(self) -> None:
        """
        changes the angle of attack's sign
        :return:
        """
        self.alpha = RADIAN135 if self.alpha == RADIAN45 else RADIAN45

    def c_drag(self) -> float:
        """
        calculates the drag coefficient based on the angle of attack
        """
        return (C_D_MAX + C_D_0) / 2 - (C_D_MAX - C_D_0) / 2 * np.cos(2 * self.alpha)

    def c_lift(self) -> float:
        """
        calculates the lift coefficient based on the angle of attack
        """
        return C_L_MAX * np.sin(2 * self.alpha)

    def drag_torque(self, phi_dot: np.ndarray) -> np.ndarray:
        """
        the drag moment
        """
        return 0.5 * AIR_DENSITY * WING_AREA * self.c_drag() * (GYRATION_RADIUS ** 2) * phi_dot * np.abs(phi_dot)

    def phi_derivatives(self, t, y):
        """
        A function that defines the ODE that is to be solved: I * phi_ddot = tau_z - tau_drag.
        We think of y as a vector y = [phi,phi_dot]. the ode solves dy/dt = f(y,t)
        :return:
        """
        phi, phi_dot = y[0], y[1]
        dy_dt = [phi_dot, (self.motor_torque(t) - self.drag_torque(phi_dot)) / MoI]
        return dy_dt

    def lift_force(self, phi_dot: np.ndarray) -> np.ndarray:
        """
        calculated the drag force on the wing, which will be used as reward
        TODO: theoretically phi_dot here will have all + or all - sign, as we separate zero crosses, no? YES
        :param phi_dot:
        :return:
        """
        f_lift = 0.5 * AIR_DENSITY * WING_AREA * self.c_lift() * phi_dot * np.abs(phi_dot)
        return f_lift

    def solve_dynamics(self, *args):
        """
        solves the ODE
        :param args: if given, it must be of the format [phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, time_arr]

        :return:
        """
        phi_0, phi_dot_0 = self.phi0, self.phi_dot0
        start_t, end_t, delta_t = self.start_t, self.end_t, self.solve_ivp_time_increments
        phi_dot_zero_crossing_event.terminal = True
        phi_dot_zero_crossing_event.direction = -np.sign(phi_dot_0)
        ang = []
        times_between_zero_cross = []
        sol_between_zero_cross = []
        lift_force = []
        torque = []
        while start_t < end_t:
            sol = solve_ivp(self.phi_derivatives, t_span=(start_t, end_t), y0=[phi_0, phi_dot_0],
                            events=phi_dot_zero_crossing_event)
            if np.abs(sol.y[1][0]) < 1e-10: sol.y[1][0] = 0
            if np.abs(sol.y[1][-1]) < 1e-10: sol.y[1][-1] = 0

            ang.append(self.alpha * np.ones(len(sol.t)))  # set alpha for every t based on solution's size
            times_between_zero_cross.append(sol.t)
            sol_between_zero_cross.append(sol.y)
            lift_force.append(self.lift_force(sol.y[1]))
            torque.append(self.motor_torque(0) * np.ones(len(sol.t)))
            if sol.status == ZERO_CROSSING:
                # TODO: make sure we set the sign of flip alpha not w.r.t after the zero crossing, set alpha according to what phi was as that time
                start_t = sol.t[-1] + delta_t
                phi_0, phi_dot_0 = sol.y[0][-1], sol.y[1][-1]  # last step is now initial value
                phi_dot_zero_crossing_event.direction *= -1
                self.flip_alpha()
            else:  # no zero crossing = the solution is for [start_t,end_t] and we are essentially done
                break
        time = np.concatenate(times_between_zero_cross)
        phi, phi_dot = np.concatenate(sol_between_zero_cross, axis=1)
        ang = np.concatenate(ang)
        lift_force = np.concatenate(lift_force)
        torque = np.concatenate(torque)
        _, phi_ddot = self.phi_derivatives(time, [phi, phi_dot])
        if args:
            for i, arr in enumerate([phi, phi_dot, phi_ddot, ang, time, lift_force, torque]):
                args[i].append(arr)
        return phi, phi_dot, phi_ddot, ang, time, lift_force, torque
