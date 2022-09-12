from gym import Env
from gym.spaces import Box
import numpy as np
from simulation import RobotSimulation


class WingEnv(Env):
    def __init__(self, min_torque=-1, max_torque=1, min_phi=0, max_phi=2 * np.pi):
        """
        the action space is a continuous torque value
        the observation space is currently a continuous value for phi (later I will add psi, theta)
        :param min_torque: the minimal torque we can apply (+-1 is considered +-90 deg)
        :param max_torque: the maximal torque we can apply
        :param min_phi: the minimal rotation angle
        :param max_phi: the maximal rotation angle
        """
        self.max_torque = max_torque
        self.min_torque = min_torque
        self.max_phi = max_phi
        self.min_phi = min_phi
        self.low = np.array([-1.0])
        self.high = np.array([1.0])

        self.action_space = Box(self.low, self.high)
        self.observation_space = Box(self.low, self.high)
        self.state = 0  # initial phi state
        self.action = 0.02  # initial torque value
        self.rounds = 20  # arbitrary number of rounds
        self.collected_reward = []
        self.simulation = RobotSimulation(motor_torque=lambda x: self.action)
        self.time_window = 0.05

    def trim_phi(self, phi):
        """
        given unbounded phi np array, casts it [-1,1]
        :param phi:
        :return:
        """
        half = np.max(phi) / 2
        phi = (np.clip(phi, self.min_phi, self.max_phi) - half) / half
        return phi

    def step(self, action):
        done = False if self.rounds > 0 else True
        self.rounds -= 1
        info = {}

        # calculate the new phi:
        self.simulation.set_motor_torque(lambda x: action)
        self.simulation.solve_dynamics([], [], [], [], [])
        sol = self.simulation.solution
        # phi = self.trim_phi(sol[0]) # TODO: if you normalize - all other params must be normalized

        last_phi = sol[0][-1]
        self.state = np.float32(last_phi)

        # calculate the reward:
        last_phi_dot = sol[1][-1]
        # TODO: use avg reward on all phi dot vector
        reward = self.simulation.lift_force(last_phi_dot)
        if last_phi > self.max_phi or last_phi < self.min_phi:  # we punish angles not in [0,180]
            reward -= 0.1 * reward
        self.collected_reward.append(reward)

        # update time window and init cond for next iterations
        self.simulation.set_time(self.simulation.end_t, self.simulation.end_t + self.time_window)
        self.simulation.set_init_cond(last_phi, last_phi_dot)
        print(f"s={self.state}, a={action}, r={reward}")
        return np.array([self.state]), reward, done, info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.state = 0.0
        self.action = 0.0
        self.rounds = 20
        return np.array([self.state]).astype(np.float32)

