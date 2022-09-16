import numpy as np
from simulation import RobotSimulation
import matplotlib.pyplot as plt


# TODO: add grid plot of sum of rewards to evaluate the model w.r.t Acos(wt) - "Energy landscape"
# TODO: check how to access the loss to add regularization - ask Aviv maybe
# TODO: add minimal work to the reward - see image on iPhone
# TODO: is it ok that the state is only the LAST phi?
# TODO: check that the lift force makes sense!

def energy_landscape():
    # might want to vectorize later
    N = 20
    end_time = 3  # seconds
    max_amplitude = 0.02  # Nm
    max_freq = 40  # Hz
    A = np.linspace(0.001, max_amplitude, N)  # (N,1)
    F = np.linspace(0.001, max_freq, N)  # (N,1)
    all_rewards = []
    for a in A:
        for f in F:
            print(f"A=[{a:.4f}/{max_amplitude}],"
                  f" f=[{f:.4f}/{max_freq}]",
                  end='\r')
            t = np.linspace(0, end_time, N)  # (N,1)
            torque = a * np.cos(2 * np.pi * f * t)  # (N,1)
            start_t, end_t, delta_t = 0, 0.05, 0.001
            sim = RobotSimulation()
            cumulative_lift_reward = 0
            for action in torque:
                sim.set_motor_torque(lambda x: action)
                sim.solve_dynamics()
                phi, phi_dot = sim.solution
                phi0, phidot0 = phi[-1], phi_dot[-1]
                start_t = end_t
                end_t += 0.05
                sim.set_init_cond(phi0, phidot0)
                sim.set_time(start_t, end_t)
                cumulative_lift_reward += (sim.drag_torque(phi_dot)).mean()
            all_rewards.append(cumulative_lift_reward)
    grid = np.array(all_rewards).reshape((N, N))
    h = plt.contourf(F, A, grid)
    # plt.axis('scaled')
    plt.title(r"Accumulated Lift reward $\langle F_{lift} \rangle =\frac{1}{2}\rho_{air} A_{wing} C_{drag}\dot\phi^2$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [m]")
    plt.colorbar()
    plt.show()


def plot_all(time, phi, phi_dot, phi_ddot, alpha, force, torque, phi0, phidot0):
    """
    plots the given data in three separate graphs
    """
    fig, axs = plt.subplots(5, 1)
    fig.set_size_inches(7, 9)
    fig.suptitle(
        r"$\ddot \phi = a\tau_z -b\dot \phi^2$ " + "\n" +
        r"$\phi_0=$" + f"{phi0:.3f}[rad], " +
        r"$\dot \phi_0=$" + f"{phidot0:.3f}[rad/sec], " +
        r"$\tau_z=$" + f"{torque}[N/m]" + "\n" +
        r"$a=\frac{1}{2mL_{aero}^2}, b=\frac{1}{2mL_{aero}^2}\rho_{air} A_{wing} C_{drag}L^2$" + "\n"
    )
    axs[0].plot(time, phi, 'red', linewidth=2)
    axs[0].set(ylabel=r'$\phi$ [rad]')

    axs[1].plot(time, phi_dot, 'orange', linewidth=2)
    axs[1].set(ylabel=r'$\dot \phi$ [rad/sec]')

    axs[2].plot(time, phi_ddot, 'green', linewidth=2)
    axs[2].set(ylabel=r'$\ddot \phi$ [rad/$sec^2$]')

    axs[3].plot(time, alpha, 'blue', linewidth=2)
    axs[3].set(ylabel=r'$\alpha$ [rad]')

    axs[4].plot(time, force, 'purple', linewidth=2)
    axs[4].set(ylabel="Force [N]")
    axs[4].set(xlabel='time [sec]')

    for ax in axs.flat:
        ax.grid()
        # ax.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.show()


def check_simulation_given_torque(torque: np.ndarray, torque_name: str, do_plot: bool):
    """
    solves the ODE in sequential order given the torque values and plots all relevant
    angles / forces / etc..
    :param torque: a np array that contains motor torque values to follow
    :return:
    """
    phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, time_arr, force_arr = [], [], [], [], [], []
    start_t, end_t = 0, 0.05
    sim = RobotSimulation()

    phi0_name = sim.phi0
    phi_dot0_name = sim.phi_dot0

    for action in torque:
        sim.set_motor_torque(lambda x: action)
        sim.solve_dynamics(phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, time_arr, force_arr)
        phi, phi_dot = sim.solution
        phi0, phidot0 = phi[-1], phi_dot[-1]
        start_t = end_t
        end_t += 0.05
        sim.set_init_cond(phi0, phidot0)
        sim.set_time(start_t, end_t)

    phi_arr = np.concatenate(phi_arr)
    phi_dot_arr = np.concatenate(phi_dot_arr)
    phi_ddot_arr = np.concatenate(phi_ddot_arr)
    time_arr = np.concatenate(time_arr)
    ang_arr = np.concatenate(ang_arr)
    force_arr = np.concatenate(force_arr)
    if do_plot: plot_all(time_arr, phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, force_arr,
                         torque_name, phi0_name, phi_dot0_name)
    return phi_arr, phi_dot_arr, phi_ddot_arr, time_arr, ang_arr, force_arr


def energy_landscape2():
    # might want to vectorize later
    N = 3
    end_time = 3  # seconds
    max_amplitude = 0.02  # Nm
    max_freq = 40  # Hz
    A = np.linspace(0.001, max_amplitude, N)  # (N,1)
    F = np.linspace(0.001, max_freq, N)  # (N,1)
    all_rewards = []
    for a in A:
        for f in F:
            print(f"A=[{a:.4f}/{max_amplitude}],"
                  f" f=[{f:.4f}/{max_freq}]",
                  end='\r')
            t = np.linspace(0, end_time, N)  # (N,1)
            torque = a * np.cos(2 * np.pi * f * t)  # (N,1)
            torque_name = f"{a:.3f}cos(2" + r"$\pi$" + f"{f:.3f}t)"
            _, _, _, _, _, force_arr = check_simulation_given_torque(torque, torque_name, True)
            all_rewards.append(np.mean(force_arr))
    grid = np.array(all_rewards).reshape((N, N))
    h = plt.contourf(F, A, grid)
    # plt.axis('scaled')
    plt.title(r"Accumulated Lift reward $\langle F_{lift} \rangle =\frac{1}{2}\rho_{air} A_{wing} C_{drag}\dot\phi^2$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [m]")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    energy_landscape2()
