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

MASS = 2.6e-4  # TODO: get wing mass
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


def plot(time, phi, phi_dot, phi_ddot, torque, phi0, phidot0):
    """
    plots the given data in three separate graphs
    """
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(7, 9)
    fig.suptitle(
        r"$\ddot \phi = a\tau_z -b\dot \phi^2$ " + "\n" + \
        r"$\phi_0=$" + f"{phi0:.3f}[rad], " + \
        r"$\dot \phi_0=$" + f"{phidot0:.3f}[rad/sec], " + \
        r"$\tau_z=$" + f"{torque:.3f}[N/m]" + "\n" \
                                              r"$a=\frac{1}{2mL_{aero}^2}, b=\frac{1}{2mL_{aero}^2}\rho_{air} A_{wing} C_{drag}L^2$" + "\n"
    )
    axs[0].plot(time, phi, 'red', linewidth=2)
    axs[0].set(ylabel=r'$\phi$ [rad]')
    axs[1].plot(time, phi_dot, 'orange', linewidth=2)
    axs[1].set(ylabel=r'$\dot \phi$ [rad/sec]')
    axs[2].plot(time, phi_ddot, 'green', linewidth=2)
    axs[2].set(ylabel=r'$\ddot \phi$ [rad/$sec^2$]')
    axs[2].set(xlabel='time [sec]')
    for ax in axs.flat:
        ax.grid()
        # ax.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.show()


def phi_dot_zero_crossing_event(t, y):
    """
    this event is given to solve_ivp to track if phi_dot == 0
    :param t: unused time variable
    :param y: a vector of [phi,phi_dot]
    :return:
    """
    return y[1]


class RobotSimulation:
    def __init__(self, motor_torque, alpha=45, phi0=0.0, phi_dot0=0.01, start_t=0, end_t=0.05, delta_t=0.001) -> None:
        """
        :param motor_torque: A function that returns float that represents the current torque provided by the motor
        :param phi0: initial phi position of the current time window
        :param phi_dot0: initial ang. velocity of the current time window
        :param start_t: start time of the current window
        :param end_t: end time of the current window
        :param delta_t: the time increments
        """
        self.solution = 0
        self.motor_torque = motor_torque
        self.alpha = alpha  # angle of attack
        self.phi0 = phi0
        self.phi_dot0 = phi_dot0
        # when defining time values we try to avoid floating points start end, as this may interfere with solve_ivp:
        self.start_t = start_t
        self.end_t = end_t
        self.delta_t = delta_t

    def set_time(self, new_start, new_end) -> None:
        """
        we update the start time to be the end of the prev time and the end time
        as the last end time + the window size we wish
        """
        self.start_t = new_start
        self.end_t = new_end

    def set_init_cond(self, new_phi0, new_phi_dot0) -> None:
        """
        we update the initial conditions to be the last value of previous iterations
        """
        self.phi0 = new_phi0
        self.phi_dot0 = new_phi_dot0

    def set_motor_torque(self, new_motor_torque) -> None:
        """
        sets the moment to new value, based on the action the learning algorithm provided
        """
        self.motor_torque = new_motor_torque

    def flip_alpha(self) -> None:
        """
        changes the angle of attack's sign
        :return:
        """
        self.alpha = -self.alpha

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

    def drag_torque(self, phi_dot):
        """
        the drag moment
        """
        return 0.5 * AIR_DENSITY * WING_AREA * self.c_drag() * GYRATION_RADIUS * phi_dot * np.abs(phi_dot)

    def phi_derivatives(self, t, y):
        """
        A function that defines the ODE that is to be solved: I * phi_ddot = tau_z - tau_drag.
        We think of y as a vector y = [phi,phi_dot]. the ode solves dy/dt = f(y,t)
        :return:
        """
        phi, phi_dot = y[0], y[1]
        dy_dt = [phi_dot, (self.motor_torque(t) - self.drag_torque(phi_dot)) / MoI]
        return dy_dt

    def lift_force(self, phi_dot):
        """
        calculated the drag force on the wing, which will be used as reward
        :param phi_dot:
        :return:
        """
        return np.abs(0.5 * AIR_DENSITY * WING_AREA * self.c_lift() * (phi_dot ** 2))

    def solve_dynamics(self, phiarr, phidotarr, phiddotarr):
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
        already_crossed_zero = False
        while start_t < end_t:

            ang.append(self.alpha)
            if not already_crossed_zero:
                sol = solve_ivp(self.phi_derivatives, t_span=(start_t, end_t), y0=[phi_0, phi_dot_0],
                                events=phi_dot_zero_crossing_event)
            else:
                sol = solve_ivp(self.phi_derivatives, t_span=(start_t, end_t), y0=[phi_0, phi_dot_0])
            self.solution = sol.y
            times_between_zero_cross.append(sol.t)
            sol_between_zero_cross.append(sol.y)
            if sol.status == ZERO_CROSSING and not already_crossed_zero:  #
                already_crossed_zero = True  # we use this to avoid multiple entries when phi dot ~ 0
                # print(f"Zero crossing at time t={start_t}, phi={sol.y[0][-1]}, phi_dot = {sol.y[1][-1]}\n"
                #       f". Jumping to time {sol.t[-1] + delta_t}")

                start_t = sol.t[-1] + delta_t
                phi_0, phi_dot_0 = sol.y[0][-1], sol.y[1][-1]  # last step is now initial value
                self.flip_alpha()
            else:
                break
        time = np.concatenate(times_between_zero_cross)
        phi, phi_dot = np.concatenate(sol_between_zero_cross, axis=1)

        _, phi_ddot = self.phi_derivatives(time, [phi, phi_dot])
        phiarr.append(phi)
        phidotarr.append(phi_dot)
        phiddotarr.append(phi_ddot)
        # plot(time, phi, phi_dot, phi_ddot, self.motor_torque(0), self.phi0, self.phi_dot0)


if __name__ == '__main__':
    # TODO: time windows are too large - one second is many many cycles
    phi_arr = []
    phi_dot_arr = []
    phi_ddot_arr = []
    phi0 = 0
    phidot0 = 0.01
    start_t = 0
    end_t = 0.05
    delta_t = 0.001
    sin = 0.02 * np.sin(2 * np.pi * np.linspace(0, 2, 50))
    for action in sin:
        # for action in np.concatenate([-np.arange(0, 0.03, 0.005), np.arange(-0.03, 0.03, 0.005)]):

        sim = RobotSimulation(lambda x: action, phi0=phi0, phi_dot0=phidot0, start_t=start_t, end_t=end_t)
        sim.solve_dynamics(phi_arr, phi_dot_arr, phi_ddot_arr)
        phi0, phidot0 = sim.solution[0][-1], sim.solution[1][-1]
        start_t = end_t
        end_t += 0.05
    phi_arr = np.concatenate(phi_arr)
    phi_dot_arr = np.concatenate(phi_dot_arr)
    phi_ddot_arr = np.concatenate(phi_ddot_arr)
    plot(np.linspace(0, end_t, len(phi_arr)), phi_arr, phi_dot_arr, phi_ddot_arr, 0, 0, 0)

    # class Solver():
    #     def __init__(self):
    #         self.torque = 0
    #
    #     def set_torque(self, new):
    #         self.torque = new
    #
    #     def phi_derivatives(self, t, y):
    #         """
    #         A function that defines the ODE that is to be solved: I * phi_ddot = tau_z - tau_drag.
    #         We think of y as a vector y = [phi,phi_dot]. the ode solves dy/dt = f(y,t)
    #         """
    #         # for exact values use
    #         a = 1090179.6616082331
    #         b = 271.9406850484702
    #
    #         phi, phi_dot = y[0], y[1]
    #         dy_dt = [phi_dot, a * self.torque - b * phi_dot * np.abs(phi_dot)]
    #         return dy_dt
    #
    # #
    # phi_arr = []
    # phi_dot_arr = []
    # phi_ddot_arr = []
    # phi_0 = 0
    # phi_dot_0 = 0.01
    # start_t = 0
    # end_t = 0.05
    # delta_t = 0.001
    # s = Solver()
    # sin = np.sin(2 * np.pi * np.linspace(0, 1, 20))
    # for torque in sin:
    #     s.set_torque(torque)
    #     sol = solve_ivp(s.phi_derivatives, t_span=(start_t, end_t), y0=[phi_0, phi_dot_0])
    #     phi, phi_dot = sol.y
    #     _, phi_ddot = s.phi_derivatives(0, [phi, phi_dot])
    #     phi_arr.append(phi)
    #     phi_dot_arr.append(phi_dot)
    #     phi_ddot_arr.append(phi_ddot)
    #
    #     phi_0 = phi[-1]
    #     phi_dot_0 = phi_dot[-1]
    #     start_t = end_t
    #     end_t += 0.05
    #
    # phi_arr = np.concatenate(phi_arr)
    # phi_dot_arr = np.concatenate(phi_dot_arr)
    # phi_ddot_arr = np.concatenate(phi_ddot_arr)
    # plot(np.linspace(0, end_t, len(phi_arr)), phi_arr, phi_dot_arr, phi_ddot_arr, 0, 0, 0)
