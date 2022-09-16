from gym import Env
from gym.spaces import Box
import numpy as np
from simulation import RobotSimulation


class WingEnv(Env):
    def __init__(self, min_torque=-1, max_torque=1, min_phi=0, max_phi=np.pi):
        """
        the action space is a continuous torque value
        the observation space is currently a continuous value for phi (later I will add psi, theta)
        :param min_torque: the minimal torque we can apply (+-1 is considered +-90 deg)
        :param max_torque: the maximal torque we can apply
        :param min_phi: the minimal rotation angle
        :param max_phi: the maximal rotation angle
        """
        self.state = 0  # initial phi state
        self.action = 0.0  # initial torque value
        self.rounds = 20  # arbitrary number of rounds
        self.iters = 0
        self.time_window = 0.05

        self.info = {}

        self.max_approx_torque = 0.02
        self.max_action_diff = 0.1 * self.max_approx_torque

        self.max_torque = max_torque
        self.min_torque = min_torque
        self.max_phi = max_phi
        self.min_phi = min_phi
        self.low = np.array([-1.0])
        self.high = np.array([1.0])
        self.action_space = Box(self.low, self.high)
        self.observation_space = Box(np.array([-np.inf]), np.array([np.inf]))

        self.collected_reward = []

        self.simulation = RobotSimulation()

    def trim_phi(self, phi):
        """
        given unbounded phi np array, casts it [-1,1]
        :param phi:
        :return:
        """
        half = np.max(phi) / 2
        phi = (np.clip(phi, self.min_phi, self.max_phi) - half) / half
        return phi

    def step(self, action: np.ndarray):
        action *= self.max_approx_torque
        self.iters += 1
        done = False if self.rounds > 0 else True
        self.rounds -= 1

        # calculate the new phi:
        self.simulation.set_motor_torque(lambda x: action)
        self.simulation.solve_dynamics()
        phi, phi_dot = self.simulation.solution
        last_phi, last_phi_dot = phi[-1], phi_dot[-1]
        self.state = np.float32(last_phi)

        # calculate the reward:
        reward = self.simulation.lift_force(phi_dot).mean()

        # punish w.r.t bad phi values
        surpass_max_reward = np.where(phi > self.max_phi, np.abs(phi - self.max_phi), 0)
        surpass_min_reward = np.where(phi < self.min_phi, np.abs(phi - self.min_phi), 0)
        surpass_reward = surpass_min_reward.sum() + surpass_max_reward.sum()
        reward -= surpass_reward

        # punish w.r.t to large changes to the torque
        # action_norm = np.linalg.norm(self.action - action)
        # if action_norm > self.max_action_diff:
        #     reward -= action_norm * reward
        self.collected_reward.append(reward)

        # update time window and init cond for next iterations
        self.simulation.set_time(self.simulation.end_t, self.simulation.end_t + self.time_window)
        self.simulation.set_init_cond(last_phi, last_phi_dot)
        self.info = {
            'iter': self.iters,
            'state': self.state,
            'action': self.action,
            'reward': reward
        }
        if self.iters % 100 == 0: print(f"[{self.iters}] |"
                                        f" s={self.state:.4f} |"
                                        f" a={action.item():.4f} |"
                                        f" r={reward.item():.4f} |")
        self.action = action  # updating new action
        return np.array([self.state]), reward.item(), done, self.info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.state = 0.0
        self.action = 0.0
        self.rounds = 20
        self.collected_reward = []
        return np.array([self.state]).astype(np.float32)
