import numpy as np
from simulation import RobotSimulation
import matplotlib.pyplot as plt


def energy_landscape():
    # might want to vectorize later
    N = 20
    A = np.linspace(0.01, 2, N)  # (N,1)
    F = np.linspace(0.01, 40, N)  # (N,1)
    N_cycles = 2
    end_time = 3 # seconds
    all_rewards = []
    for a in A:
        for f in F:
            print(f"A=[{a:.4f}/1], f=[{f:.4f}/30]", end='\r')
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
    plt.title(r"Accumulated Lift reward $F_{lift}=\frac{1}{2}\rho_{air} A_{wing} C_{drag}\dot\phi^2$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [m]")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    energy_landscape()
