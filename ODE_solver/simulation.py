from typing import Callable, List
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from constants import *


def print_parameters():
    """
    a utility function that prints out all the calculated parameters in the simulation
    """
    print(
        f"Gyration radius = {GYRATION_RADIUS} |"
        f"Moment of Inertia = {MoI} |"
        f"Wing Area = {WING_AREA}"
    )


def _phi_dot_zero_crossing_event(t, y):
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
                 psi0: float, psi_dot0: float,
                 start_t: float, end_t: float,
                 motor_torque: Callable = lambda x: 0,
                 alpha: float = RADIAN45,
                 zero_crossing_time_increments: float = ZERO_CROSSING_TIME_INCREMENTS) -> None:
        """
        :param motor_torque: A function that returns float that represents the current torque provided by the motor
        :param phi0: initial phi position of the current time window
        :param phi_dot0: initial ang. velocity of the current time window
        :param start_t: start time of the current window
        :param end_t: end time of the current window
        :param zero_crossing_time_increments: the time increments
        """
        self.solution = 0
        self.motor_torque = motor_torque
        self.atk_angle = alpha  # angle of attack
        self.phi0 = phi0
        self.phi_dot0 = phi_dot0
        self.psi0 = psi0
        self.psi_dot0 = psi_dot0
        self.start_t = start_t
        self.end_t = end_t
        self.zero_crossing_time_increment = zero_crossing_time_increments

    def set_time(self, new_start: float, new_end: float) -> None:
        """
        we update the start time to be the end of the prev time and the end time
        as the last end time + the window size we wish
        """
        self.start_t = new_start
        self.end_t = new_end

    def set_init_cond(self, new_phi0: float, new_phi_dot0: float,
                      new_psi0: float, new_psi_dot0: float) -> None:
        """
        we update the initial conditions to be the last value of previous iterations
        """
        self.phi0 = new_phi0
        self.phi_dot0 = new_phi_dot0
        self.psi0 = new_psi0
        self.psi_dot0 = new_psi_dot0

    def set_motor_torque(self, new_motor_torque: Callable) -> None:
        """
        sets the moment to new value, based on the action the learning algorithm provided
        we use a function as this provides more versatility, if needed
        """
        self.motor_torque = new_motor_torque

    def _flip_atk_angle(self) -> None:
        """
        changes the angle of attack's sign
        :return:
        """
        self.atk_angle = RADIAN135 if self.atk_angle == RADIAN45 else RADIAN45

    def _c_drag(self) -> float:
        """
        calculates the drag coefficient based on the angle of attack
        this function is used to calculate the drag torque
        """
        return (C_D_MAX + C_D_0) / 2 - (C_D_MAX - C_D_0) / 2 * np.cos(2 * self.atk_angle)

    def _c_lift(self, atk_angle: np.ndarray) -> float:
        """
        takes a vector of angles and calculates the lift coefficient.
        this function is used to calculate the lift force
        """
        return C_L_MAX * np.sin(2 * atk_angle)

    def _lift_force(self, phi_dot: np.ndarray, atk_angle: np.ndarray) -> np.ndarray:
        """
        calculated the drag force on the wing, which will be used as reward
        TODO: theoretically phi_dot here will have all + or all - sign, as we separate zero crosses, no?
        :param atk_angle: a np array of angle values
        :param phi_dot: a np array of ang. velocities
        :return: a lift force np array
        """
        c_lift = self._c_lift(atk_angle)
        f_lift = 0.5 * AIR_DENSITY * WING_AREA * c_lift * phi_dot * np.abs(phi_dot)
        return f_lift

    def _drag_force(self, phi_dot: np.ndarray) -> np.ndarray:
        """
        takes a vector of velocities and calculates the drag force vector.
        """
        return 0.5 * AIR_DENSITY * WING_AREA * self._c_drag() * phi_dot * np.abs(phi_dot)

    def _drag_torque(self, phi_dot: np.ndarray) -> np.ndarray:
        """
        takes a vector of velocities and calculates the drag moment vector.
        """
        return 0.5 * AIR_DENSITY * WING_AREA * self._c_drag() * (GYRATION_RADIUS ** 2) * phi_dot * np.abs(phi_dot)

    def _total_force(self, phi_dot: np.ndarray, atk_angle: np.ndarray):
        f_lift = self._lift_force(phi_dot, atk_angle)
        f_drag = self._drag_force(phi_dot)

        return np.sqrt(f_drag ** 2 + f_lift ** 2)

    def _phi_psi_derivatives(self, t, y):
        """
        A function that defines the diff-equationS that are to be solved:
         1. I * phi_ddot = tau_z - tau_drag.
         2. I * psi_ddot = F_tot x R - kappa * psi - gamma * psi_dot
        We think of y as a vector y = [phi,phi_dot,psi,psi_dot]. we solve dy/dt = f(t,y)
        :return: dy/dt
        """
        phi, phi_dot, psi, psi_dot = y
        phi_ddot = (self.motor_torque(t) - self._drag_torque(phi_dot)) / MoI
        psi_ddot = (self._total_force(phi_dot, psi) - PSI_KAPPA * psi - PSI_GAMMA * psi_dot) / MoI
        dy_dt = [phi_dot, phi_ddot, psi_dot, psi_ddot]
        return dy_dt

    def solve_dynamics(self):
        """
        a public function that solves the Diff-eq.
        :return:
        """
        sol = solve_ivp(self._phi_psi_derivatives,
                        t_span=(self.start_t, self.end_t),
                        y0=[self.phi0, self.phi_dot0, self.psi0, self.psi_dot0])
        phi, phi_dot, psi, psi_dot = sol.y
        _, phi_ddot, _, psi_ddot = self._phi_psi_derivatives(sol.t, [phi, phi_dot, psi, psi_dot])

        torque = self.motor_torque(0) * np.ones(len(sol.t))  # torque is constant in every solve_dynamics call
        lift_force = self._lift_force(phi_dot, psi)
        time = sol.t
        return phi, phi_dot, phi_ddot, psi, psi_dot, psi_ddot, time, lift_force, torque


if __name__ == '__main__':

    start_t, end_t, n_steps = 0, 0.2, 1
    t = np.linspace(start_t, end_t, n_steps)
    f = 4
    torque = 0.02 * np.sin(2 * np.pi * f * t)

    phi0, phi_dot0 = 0, 2e-4
    psi0, psi_dot0 = 0, 2e-4
    sim = RobotSimulation(phi0, phi_dot0, psi0, psi_dot0, start_t, end_t)
    vals = [np.array([]) for _ in range(9)]  # phi, phi_dot, phi_ddot, psi, psi_dot, psi_ddot, time, lift_force, torque
    for action in tqdm(torque):
        sim.set_motor_torque(lambda x: action)
        new_vals = sim.solve_dynamics()
        for i in range(9): vals[i] = np.concatenate((vals[i], new_vals[i]))
        phi0, phi_dot0 = vals[0][-1], vals[1][-1]
        psi0, psi_dot0 = vals[3][-1], vals[4][-1]
        start_t = end_t
        end_t += 0.2
        sim.set_init_cond(phi0, phi_dot0, psi0, psi_dot0)
        sim.set_time(start_t, end_t)
    time = vals[6]
    # for item, title in zip(vals, ['phi', 'phi_dot', 'phi_ddot',
    #                               'psi', 'psi_dot', 'psi_ddot',
    #                               'time', 'lift_force', 'torque']):
    #     if title == 'time': continue
    #     plt.plot(time, item)
    #     plt.title(title)
    #     plt.show()
