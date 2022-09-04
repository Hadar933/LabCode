from gym import Env
from gym.spaces import Box
import numpy as np
from simulation import RobotSimulation


class WingEnv(Env):
    def __init__(self, min_torque=-np.inf, max_torque=np.inf, min_phi=0, max_phi=180):
        """
        the action space is a continuous torque value
        the observation space is currently a continuous value for phi (later I will add psi, theta)
        :param min_torque: the minimal torque we can apply (+-1 is considered +-90 deg)
        :param max_torque: the maximal torque we can apply
        :param min_phi: the minimal rotation angle
        :param max_phi: the maximal rotation angle
        """
        self.min_torque = np.array([min_torque], dtype=np.float32)
        self.max_torque = np.array([max_torque], dtype=np.float32)
        self.min_phi = np.float32(min_phi)
        self.max_phi = np.float32(max_phi)

        self.action_space = Box(self.min_torque, self.max_torque, dtype=np.float32)
        self.observation_space = Box(np.array([0], dtype=np.float32), np.array([np.inf], dtype=np.float32),
                                     dtype=np.float32)
        self.state = 0  # some initial phi state
        self.rounds = 20  # number of rounds
        self.collected_reward = []
        self.simulation = RobotSimulation(tau_z=lambda x: self.action)
        self.time_window = 1

    def step(self, action):
        done = False if self.rounds > 0 else True
        self.rounds -= 1
        info = {}
        # calculate the new phi:
        self.simulation.set_tau_z(lambda x: action)
        self.simulation.solve_dynamics()
        sol = self.simulation.solution
        phi = sol[0][-1]
        self.state = np.float32(phi)

        # calculate the reward:
        phi_dot = sol[1][-1]  # we use the velocity at the last time step
        reward = self.simulation.drag_force(phi_dot)
        if phi > self.max_phi:
            reward /= 2
        elif phi < self.min_phi:  # we punish angles not in [0,180]
            reward /= 2
        self.collected_reward.append(reward)

        # update time window and init cond for next iterations
        self.simulation.update_time(self.simulation.end_t, self.simulation.end_t + self.time_window)
        self.simulation.update_initial_cond(phi, phi_dot)

        return np.array([self.state]), reward, done, info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.state = 0.0
        self.action = 0.0
        self.rounds = 20
        return np.array([self.state]).astype(np.float32)
