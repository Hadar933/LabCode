import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# TODO:
"""
1. we might want to regularize the torque's gradient (changes in torque cannot be to large, as to account for the motor's
capabilities)
2. the reward of minimal reward should be of the absolute value (Arion's paper - elastic bound condition)
3. add event of sign(phi_dot)
4. make sure atol and rtol are calibrated
"""

"""
###################################################################################################################
                                                Constants: 
###################################################################################################################
"""

NUM_SAMPLES = 100  # every time window has this much samples
MASS = 1  # TODO: get wing mass
WING_LENGTH = 0.07  # meter
AERODYNAMIC_CENTER = 0.7 * WING_LENGTH
GYRATION_RADIUS = 0.6 * WING_LENGTH
MoI = MASS * GYRATION_RADIUS ** 2
AIR_DENSITY = 1.2  # From Arion's simulatio
WING_AREA = 0.5 * WING_LENGTH * (0.5 * WING_LENGTH) * np.pi  # 1/2 ellipse with minor radios ~ 1/2 major = length/2
# drag coefficients from whitney & wood (JFM 2010):
C_D_MAX = 3.4
C_D_0 = 0.4
C_L_MAX = 1.8
ZERO_CROSSING = 1
"""
###################################################################################################################
                                                Utilities:
###################################################################################################################
"""


def plot(time, phi, phi_dot, phi_ddot):
    """
    plots the given data in three separate graphs
    """
    fig, axs = plt.subplots(3, 1)
    fig.suptitle(
        r"$\ddot \phi = 1*a-b\dot \phi^2$, $\phi_0=0.002$, $\dot \phi_0=0$, $a=\frac{1}{mL_{aero}^2}$,"
        r"$b=\frac{1}{2mL_{aero}^2}\rho_{air} A_{wing} C_{drag}L^2$")
    axs[0].plot(time, phi, 'red', linewidth=2)
    axs[0].set(ylabel=r'$\phi$')
    axs[1].plot(time, phi_dot, 'orange', linewidth=2)
    axs[1].set(ylabel=r'$\dot \phi$')
    axs[2].plot(time, phi_ddot, 'green', linewidth=2)
    axs[2].set(ylabel=r'$\ddot \phi$')
    axs[2].set(xlabel='time')
    for ax in axs.flat:
        ax.grid()
        # ax.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.show()


"""
###################################################################################################################
                                                Solver:
###################################################################################################################
"""


def phi_dot_zero_crossing_event(t, y):
    """
    this event is given to solve_ivp to track if phi_dot == 0
    :param t: unused time variable
    :param y: a vector of [phi,phi_dot]
    :return:
    """
    return y[1]


class RobotSimulation:
    def __init__(self, tau_z) -> None:
        """
        :param tau_z: A function that returns float of the current torque
        :param phi0: initial position of the current time window
        :param phi_dot0: initial velocity of the current time window
        :param start_t: start time of the current window
        :param end_t: end time of the current window
        :param delta_t: the time increments
        """
        self._alpha = 45  # angle of attack
        self.tau_z = tau_z  # motor torque function
        self.solution = 0
        self.phi0 = 0
        self.phi_dot0 = 0.002
        self.start_t = 0
        self.end_t = 2
        self.delta_t = 0.01

    def update_time(self, new_start, new_end) -> None:
        """
        we update the start time to be the end of the prev time and the end time
        as the last end time + the window size we wish
        :param new_start:
        :param new_end:
        :return:
        """
        self.start_t = new_start
        self.end_t = new_end

    def update_initial_cond(self, new_phi0, new_phi_dot0) -> None:
        """
        we update the initial conditions to be the last value of previous iterations

        :param new_phi0:
        :param new_phi_dot0:
        :return:
        """
        self.phi0 = new_phi0
        self.phi_dot0 = new_phi_dot0

    def set_tau_z(self, new_tau_z) -> None:
        """
        sets the moment to new value, based on the action the learning algorithm provided
        :param new_tau_z:
        """
        self.tau_z = new_tau_z

    def flip_alpha(self) -> None:
        """
        changes the angle of attack's sign
        :return:
        """
        self._alpha = -self._alpha

    def get_c_drag(self) -> float:
        """
        calculates the drag coefficient based on the angle of attack
        """
        return (C_D_MAX + C_D_0) / 2 - (C_D_MAX - C_D_0) / 2 * np.cos(2 * self._alpha)

    def get_c_lift(self) -> float:
        """
        calculates the lift coefficient based on the angle of attack
        """
        return C_L_MAX * np.sin(2 * self._alpha)

    def tau_drag(self, phi_dot):
        """
        TODO: try a rect with period time = time window
        the drag moment
        """
        return 0.5 * AIR_DENSITY * WING_AREA * self.get_c_drag() * (GYRATION_RADIUS ** 2) * (phi_dot ** 2)

    def phi_derivatives(self, t, y):
        """
        A function that defines the ODE that is to be solved: I * phi_ddot = tau_z - tau_drag.
        We think of y as a vector y = [phi,phi_dot]. the ode solves dy/dt = f(y,t)
        :return:
        """
        phi, phi_dot = y[0], y[1]
        dy_dt = [phi_dot, (self.tau_z(t) - self.tau_drag(phi_dot)) / MoI]
        return dy_dt

    def drag_force(self, phi_dot):
        """
        calculated the drag force on the wing, which will be used as reward
        :param phi_dot:
        :return:
        """
        return 0.5 * AIR_DENSITY * WING_AREA * self.get_c_lift() * (phi_dot ** 2)

    def solve_dynamics(self):
        """
        solves the ODE
        :return:
        """
        phi_0, phi_dot_0 = self.phi0, self.phi_dot0
        start_t, end_t, delta_t = self.start_t, self.end_t, self.delta_t
        phi_dot_zero_crossing_event.terminal = True

        ang = []
        times_between_zero_cross = []
        sol_between_zero_cross = []
        while start_t < end_t:
            ang.append(self._alpha)
            time = np.arange(start_t, end_t, delta_t)
            sol = solve_ivp(self.phi_derivatives, t_span=(start_t, end_t), y0=[phi_0, phi_dot_0], t_eval=time,
                            events=phi_dot_zero_crossing_event)
            self.solution = sol.y
            times_between_zero_cross.append(sol.t)
            sol_between_zero_cross.append(sol.y)
            if sol.status == ZERO_CROSSING:  # phi_dot == 0 so we solve again from where we have finished
                start_t = sol.t[-1] + delta_t
                phi_0, phi_dot_0 = sol.y[0][-1], sol.y[1][-1]  # last step is now initial value
                self.flip_alpha()
            else:
                break
        time = np.concatenate(times_between_zero_cross)
        phi, phi_dot = np.concatenate(sol_between_zero_cross, axis=1)

        _, phi_ddot = self.phi_derivatives(time, [phi, phi_dot])
        plot(time, phi, phi_dot, phi_ddot)
        print(ang)

        # else:
        #     for t in np.arange(start_t, end_t, delta_t):  # t defines the last time sample in the current iteration
        #         t_window = np.arange(prev_t, t - inner_delta_t, inner_delta_t)  # a time window [previous t, current t)
        #         sol = solve_ivp(self.phi_derivatives, t_span=(prev_t, t), y0=[phi_0, phi_dot_0], t_eval=t_window,
        #                         events=lambda t, phi_dot: phi_dot)
        #         phi, phi_dot = sol.y
        #         _, phi_ddot = self.phi_derivatives(t_window, [phi, phi_dot])
        #         phi_0, phi_dot_0 = phi[-1], phi_dot[-1]
        #         prev_t = t
        #         plot(t_window, phi, phi_dot, phi_ddot)
