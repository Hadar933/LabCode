from gym import Env
from gym.spaces import Box, Dict
import numpy as np
from simulation import RobotSimulation
import random


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
        self.all_phi_history = np.array([])
        self.state = 0  # initial phi state
        self.action = 0.0  # initial torque value
        self.rounds = 20  # arbitrary number of rounds
        self.iters = 0
        self.time_window = 0.02
        self.history_size = 10

        self.info = {}

        self.max_approx_torque = 0.02
        self.max_action_diff = 0.05 * self.max_approx_torque

        # self.max_torque = max_torque
        # self.min_torque = min_torque
        self.max_phi = max_phi
        self.min_phi = min_phi
        self.low = np.array([-1.0])
        self.high = np.array([1.0])
        self.action_space = Box(self.low, self.high)
        # self.observation_space = Box(np.array([-np.inf]), np.array([np.inf]))
        self.observation_space = Dict({
            "phi": Box(low=np.array([-np.inf] * self.history_size),
                       high=np.array([np.inf] * self.history_size))
        })
        self.collected_reward = []

        self.simulation = RobotSimulation()

    # def trim_phi(self, phi):
    #     """
    #     given unbounded phi np array, casts it [-1,1]
    #     :param phi:
    #     :return:
    #     """
    #     half = np.max(phi) / 2
    #     phi = (np.clip(phi, self.min_phi, self.max_phi) - half) / half
    #     return phi
    def delete_history(self):
        self.all_phi_history = np.array([])

    def step(self, action: np.ndarray):
        action *= self.max_approx_torque
        self.iters += 1
        done = False if self.rounds > 0 else True
        self.rounds -= 1

        # calculate the new phi:
        self.simulation.set_motor_torque(lambda x: action)
        self.simulation.solve_dynamics()
        phi, phi_dot = self.simulation.solution

        self.all_phi_history = np.concatenate([self.all_phi_history, phi])
        last_phi, last_phi_dot = phi[-1], phi_dot[-1]
        phi_history = self.all_phi_history[-self.history_size:]
        # self.state = np.float32(last_phi)
        self.state = phi_history.astype(np.float32)
        # calculate the reward:
        lift_reward = self.simulation.lift_force(phi_dot).mean()  # TODO: does this even affects the outcome?

        # punish w.r.t bad phi values
        surpass_max_reward = np.where(phi > self.max_phi, np.abs(phi - self.max_phi), 0)
        surpass_min_reward = np.where(phi < self.min_phi, np.abs(phi - self.min_phi), 0)
        relative_size = len(surpass_min_reward) + len(surpass_max_reward)
        angle_reward = surpass_min_reward.sum() + surpass_max_reward.sum()

        # punish w.r.t to large changes to the torque
        action_norm = np.linalg.norm(self.action - action)
        action_reward = 0
        if action_norm > self.max_action_diff:
            action_reward = relative_size * action_norm * 100

        reward = lift_reward - angle_reward - action_reward
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
        if self.iters % 100 == 0:
            print(f"[{self.iters}] |"
                  f" s={self.state[-1]:.4f} |"
                  f" a={action.item():.4f} |"
                  f" r_lift={lift_reward:.4f} |"
                  f" r_phi={angle_reward:.4f} |"
                  f" r_tau={action_reward :.4f} |"
                  f" r_final={reward.item():.4f} |")

        self.action = action  # updating new action

        return {'phi': self.state}, reward.item(), done, self.info

        # return np.array([self.state]), reward.item(), done, self.info

    def render(self, mode="human"):
        pass

    def reset(self):
        zero_history = [0.0] * (self.history_size - 1)
        random_state = [round(random.uniform(self.min_phi, self.max_phi), ndigits=2)]
        self.state = np.array(zero_history + random_state, dtype=np.float32)
        self.action = 0.0
        self.rounds = 20
        self.collected_reward = []
        return {
            'phi': self.state
        }
        # return np.array([self.state]).astype(np.float32)
