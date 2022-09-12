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


def plot_all(time, phi, phi_dot, phi_ddot, alpha, torque, phi0, phidot0):
    """
    plots the given data in three separate graphs
    """
    fig, axs = plt.subplots(4, 1)
    fig.set_size_inches(7, 9)
    fig.suptitle(
        r"$\ddot \phi = a\tau_z -b\dot \phi^2$ " + "\n" +
        r"$\phi_0=$" + f"{phi0:.3f}[rad], " +
        r"$\dot \phi_0=$" + f"{phidot0:.3f}[rad/sec], " +
        r"$\tau_z=$" + f"{torque:.3f}[N/m]" + "\n" +
        r"$a=\frac{1}{2mL_{aero}^2}, b=\frac{1}{2mL_{aero}^2}\rho_{air} A_{wing} C_{drag}L^2$" + "\n"
    )
    axs[0].plot(time, phi, 'red', linewidth=2)
    axs[0].set(ylabel=r'$\phi$ [rad]')

    axs[1].plot(time, phi_dot, 'orange', linewidth=2)
    axs[1].set(ylabel=r'$\dot \phi$ [rad/sec]')

    axs[2].plot(time, phi_ddot, 'green', linewidth=2)
    axs[2].set(ylabel=r'$\ddot \phi$ [rad/$sec^2$]')

    axs[3].plot(time, alpha, 'blue', linewidth=2)
    axs[3].set(ylabel=r'$\alpha$')
    axs[3].set(xlabel='time [sec]')

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
    def __init__(self, motor_torque=lambda x: 0, alpha=RADIAN45, phi0=0.0, phi_dot0=0.01, start_t=0, end_t=0.05,
                 delta_t=0.001) -> None:
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
        # dy_dt = [phi_dot, self.motor_torque(t)-phi_dot * np.abs(phi_dot)]
        # dy_dt = [phi_dot, -phi]
        return dy_dt

    def lift_force(self, phi_dot):
        """
        calculated the drag force on the wing, which will be used as reward
        :param phi_dot:
        :return:
        """
        return np.abs(0.5 * AIR_DENSITY * WING_AREA * self.c_lift() * (phi_dot ** 2))

    def solve_dynamics(self, phiarr, phidotarr, phiddotarr, angarr, timearr):
        """
        solves the ODE
        :return:
        """
        phi_0, phi_dot_0 = self.phi0, self.phi_dot0
        start_t, end_t, delta_t = self.start_t, self.end_t, self.delta_t
        phi_dot_zero_crossing_event.terminal = True
        phi_dot_zero_crossing_event.direction = -np.sign(phi_dot_0)
        ang = []
        times_between_zero_cross = []
        sol_between_zero_cross = []
        while start_t < end_t:
            sol = solve_ivp(self.phi_derivatives, t_span=(start_t, end_t), y0=[phi_0, phi_dot_0],
                            events=phi_dot_zero_crossing_event, max_step=0.01)
            self.solution = sol.y
            ang.append(self.alpha * np.ones(len(sol.t)))  # set alpha for every t based on solution's size
            times_between_zero_cross.append(sol.t)
            sol_between_zero_cross.append(sol.y)
            if sol.status == ZERO_CROSSING:
                start_t = sol.t[-1] + delta_t
                phi_0, phi_dot_0 = sol.y[0][-1], sol.y[1][-1]  # last step is now initial value
                phi_dot_zero_crossing_event.direction *= -1
                self.flip_alpha()
            else:  # no zero crossing = the solution is for [start_t,end_t] and we are essentially done
                break
        time = np.concatenate(times_between_zero_cross)
        phi, phi_dot = np.concatenate(sol_between_zero_cross, axis=1)
        ang = np.concatenate(ang)

        _, phi_ddot = self.phi_derivatives(time, [phi, phi_dot])
        phiarr.append(phi)
        phidotarr.append(phi_dot)
        phiddotarr.append(phi_ddot)
        angarr.append(ang)
        timearr.append(time)
        # plot(time, phi, phi_dot, phi_ddot, self.motor_torque(0), self.phi0, self.phi_dot0)


if __name__ == '__main__':
    phi_arr, phi_dot_arr, phi_ddot_arr, angarr, timearr = [], [], [], [], []
    phi0, phidot0 = 0, 0.01
    start_t, end_t, delta_t = 0, 0.05, 0.001
    sim = RobotSimulation()
    for action in 0.02 * (np.sin(2 * np.pi * np.linspace(0, 3, 60))):
        # print(f"a={action:.3f}, start_t={start_t:.3f}, end_t={end_t:.3f},alpha={sim.alpha:.3f}")
        sim.set_motor_torque(lambda x: action)
        sim.solve_dynamics(phi_arr, phi_dot_arr, phi_ddot_arr, angarr, timearr)
        phi0, phidot0 = sim.solution[0][-1], sim.solution[1][-1]
        start_t = end_t
        end_t += 0.05
        sim.set_init_cond(phi0, phidot0)
        sim.set_time(start_t, end_t)
    phi_arr = np.concatenate(phi_arr)
    phi_dot_arr = np.concatenate(phi_dot_arr)
    phi_ddot_arr = np.concatenate(phi_ddot_arr)
    angle_arr = np.concatenate(angarr)
    timearr = np.concatenate(timearr)
    angarr = np.concatenate(angarr)
    plot_all(timearr, phi_arr, phi_dot_arr, phi_ddot_arr, angarr, 0, 0, 0)

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
    #         # dy_dt = [phi_dot, a * self.torque - b * phi_dot * np.abs(phi_dot)]
    #         dy_dt = [phi_dot, -phi]
    #         return dy_dt
    #
    # phi_arr = []
    # phi_dot_arr = []
    # phi_ddot_arr = []
    # phi_0 = 0
    # phi_dot_0 = 0.01
    # start_t = 0
    # end_t = 50
    # delta_t = 0.01
    # s = Solver()
    #
    # for torque in [0.02]:
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
