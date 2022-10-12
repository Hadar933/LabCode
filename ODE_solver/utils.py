import numpy as np
from simulation import RobotSimulation
import matplotlib.pyplot as plt


# TODO: whats up with the time values? inconsistent                                                (V)
# TODO: check that the lift force makes sense                                                      (V)
# TODO: check how to access the loss to add regularization - ask Aviv maybe                        (V) - avoid
# TODO: add grid plot of sum of rewards to evaluate the model w.r.t Acos(wt) - "Energy landscape"  (V) - freq does not affect


# TODO: make sense of all the Tensorboard loss values                                              (X)
# TODO: add minimal work to the reward - see image on iPhone                                       (X)
# TODO: is it ok that the state is only the LAST phi?                                              (X)
# TODO: make sure atol and rtol are calibrated in solve_ivp                                        (X)


def plot_all(time, torque, phi, phi_dot, phi_ddot, alpha, force, torque_name, phi0, phidot0):
    """
    plots the given data
    """
    fig, axs = plt.subplots(6, 1)
    fig.set_size_inches(18, 12)
    fig.suptitle(
        r"$\ddot \phi = a\tau_z -b\dot \phi^2$ " + "\n" +
        r"$\phi_0=$" + f"{phi0:.3f}[rad], " +
        r"$\dot \phi_0=$" + f"{phidot0:.3f}[rad/sec], " +
        r"$\tau_z=$" + f"{torque_name}[N/m]" + "\n" +
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
    axs[4].set(ylabel=r"$F_{Lift}$ [N]")

    axs[5].plot(time, torque, 'black', linewidth=2)
    axs[5].set(ylabel=r"$\tau_{z}$ [Nm]")
    axs[5].set(xlabel='time [sec]')

    for ax in axs.flat:
        ax.grid()
        # ax.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.show()


def check_simulation_given_torque(delta_t, torque: np.ndarray, torque_name: str, do_plot: bool):
    """
    solves the ODE in sequential order given the torque values and plots all relevant
    angles / forces / etc..
    :param torque: a np array that contains motor torque values to follow
    :param torque_name: a string that represents the torque function (for the plot)
    :param do_plot: True iff we wish to plot all the saved values
    :return: arrays that represents phi,phi_dot,phi_ddot,time,alpha,lift force
    """
    phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, time_arr, force_arr, torque_arr = [], [], [], [], [], [], []
    start_t, end_t = 0, 10 * delta_t  # TODO: playing with the timing here - I think that setting += delta_t causes bugs. we dont need to sync the action timing with the cosine timing
    sim = RobotSimulation(end_t=end_t, phi_dot0=2e-4)

    phi0_name = sim.phi0
    phi_dot0_name = sim.phi_dot0

    for action in torque:
        sim.set_motor_torque(lambda x: action)
        sim.solve_dynamics(phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, time_arr, force_arr, torque_arr)
        phi, phi_dot = sim.solution
        phi0, phidot0 = phi[-1], phi_dot[-1]
        start_t = end_t
        end_t += 10 * delta_t  # TODO: playing with the timing here - I think that setting += delta_t causes bugs. we dont need to sync the action timing with the cosine timing
        sim.set_init_cond(phi0, phidot0)
        sim.set_time(start_t, end_t)

    phi_arr = np.concatenate(phi_arr)
    phi_dot_arr = np.concatenate(phi_dot_arr)
    phi_ddot_arr = np.concatenate(phi_ddot_arr)
    time_arr = np.concatenate(time_arr)
    ang_arr = np.concatenate(ang_arr)
    force_arr = np.concatenate(force_arr)  # TODO: this array behaves weird.
    torque_arr = np.concatenate(torque_arr)
    if do_plot: plot_all(time_arr, torque_arr, phi_arr, phi_dot_arr, phi_ddot_arr, ang_arr, force_arr,
                         torque_name, phi0_name, phi_dot0_name)
    return phi_arr, phi_dot_arr, phi_ddot_arr, time_arr, ang_arr, force_arr


def energy_landscape(n_timesteps, end_time, max_amplitude, max_freq, n_samples):
    """
    scans a range of amplitudes A={a_1,a_2,...,a_N} and frequencies F={f_1,f_2,...,f_N} and calculates the
    accumulated lift force that was generated using torque = a*cos(2*pi*f*t) for end_time seconds
    """
    delta_t = end_time / n_timesteps
    t = np.linspace(0, end_time, n_timesteps)
    amplitude_arr = np.linspace(0.005, max_amplitude, n_samples)  # (N,1)
    freq_array = np.linspace(1, max_freq, n_samples)  # (N,1)
    all_rewards = []
    for i, a in enumerate(amplitude_arr):
        for j, f in enumerate(freq_array):
            print(f"[{j + i * n_samples + 1}/{n_samples * n_samples}] "
                  f"A=[{a:.4f}/{max_amplitude}],"
                  f" f=[{f:.4f}/{max_freq}]",
                  end='\r')
            torque = a * np.cos(2 * np.pi * f * t)
            torque_name = f"{a:.3f}cos(2" + r"$\pi$" + f"{f:.3f}t)"
            _, _, _, _, _, force_arr = check_simulation_given_torque(delta_t, torque, torque_name, False)
            all_rewards.append(np.mean(force_arr))
    grid = np.array(all_rewards).reshape((n_samples, n_samples))
    h = plt.contourf(freq_array, amplitude_arr, grid)
    # plt.axis('scaled')
    plt.title(r"Accumulated Lift reward $\langle F_{lift} \rangle = "
              r"\langle \frac{1}{2}\rho_{air} A_{wing} C_{drag}\dot\phi^2 \rangle$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [Nm]")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    end_t = 1
    n_steps = 500
    t = np.linspace(0, end_t, n_steps)
    for f in [5]:
        # tau = 0.02 * np.sin(2 * np.pi * f * t)
        tau = ([0.02] * 10 + [-0.02] * 10) * 25
        tau_name = "cos(2" + r"$\pi$" + f"{f}t)"
        _, _, _, _, _, force_arr = check_simulation_given_torque(end_t / n_steps, tau, tau_name, True)
        print(f"{tau_name}, F = {np.mean(force_arr)}")

    # energy_landscape(n_steps, end_t, 0.025, 32, 10)
