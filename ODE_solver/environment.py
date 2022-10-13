from collections import deque
from typing import List

from gym import Env
from gym.spaces import Box, Dict
import numpy as np
from simulation import RobotSimulation
import random


class WingEnv(Env):
    def __init__(self, min_torque: float = -1, max_torque: float = 1,
                 min_phi: float = 0, max_phi: float = np.pi,
                 history_size: int = 10,
                 step_time: int = 0.02,
                 steps_per_episode: int = 20):
        """
        the action space is a continuous torque value
        the observation space is currently a continuous value for phi (later I will add psi, theta)
        :param min_torque: the minimal torque we can apply (+-1 is considered +-90 deg)
        :param max_torque: the maximal torque we can apply
        :param min_phi: the minimal rotation angle
        :param max_phi: the maximal rotation angle
        """
        # initialize history as deques (FILO) of fixed size
        self.history_size: int = history_size
        self.phi_history_deque: deque = deque([0.0] * self.history_size, maxlen=self.history_size)
        self.torque_history_deque: deque = deque([0.0] * self.history_size, maxlen=self.history_size)

        self.rounds: int = steps_per_episode  # time of each episode is rounds * step_time
        self.iters: int = 0
        self.step_time: float = step_time  # seconds
        self.max_approx_torque: float = 0.02
        self.max_action_diff: float = 0.05 * self.max_approx_torque

        self.max_torque: float = max_torque
        self.min_torque: float = min_torque
        self.max_phi: float = max_phi
        self.min_phi: float = min_phi

        self.action_space: Box = Box(np.array([min_torque], dtype=np.float32), np.array([max_torque], dtype=np.float32))
        self.observation_space: Dict = Dict({
            "phi": Box(low=np.array([-np.inf] * self.history_size, dtype=np.float32),
                       high=np.array([np.inf] * self.history_size), dtype=np.float32),
            'torque': Box(low=np.array([min_torque] * self.history_size, dtype=np.float32),
                          high=np.array([max_torque] * self.history_size, dtype=np.float32))
        })
        self.simulation: RobotSimulation = RobotSimulation(phi0=0, phi_dot0=2e-4, start_t=0, end_t=self.step_time)
        self.info: dict = {}

    def step(self, action: np.ndarray):
        action *= self.max_approx_torque
        self.iters += 1
        done = False if self.rounds > 0 else True
        self.rounds -= 1

        # INVOKE SIMULATION:
        self.simulation.set_motor_torque(lambda x: action)
        self.simulation.solve_dynamics()
        phi, phi_dot = self.simulation.solution
        last_phi, last_phi_dot = phi[-1], phi_dot[-1]

        # UPDATE STATE & ACTION HISTORY:
        for item in phi[-self.history_size:]: self.phi_history_deque.append(item)  # appending (FILO) the last elements
        self.torque_history_deque.append(action.item())

        # working with stacks converted to np arrays
        np_phi = np.array(self.phi_history_deque).astype(np.float32)
        np_torque = np.array(self.torque_history_deque).astype(np.float32)

        # CALCULATING THE REWARD:
        lift_reward = self.simulation.lift_force(phi_dot).mean()
        lift_rel_size = len(phi_dot)

        # punish w.r.t bad phi values
        surpass_max_phi = np.where(np_phi > self.max_phi, np.abs(np_phi - self.max_phi), 0)
        surpass_min_phi = np.where(np_phi < self.min_phi, np.abs(np_phi - self.min_phi), 0)
        phi_rel_size = len(np.nonzero(surpass_min_phi)[0]) + len(
            np.nonzero(surpass_max_phi)[0])  # TODO: this can be zero
        phi_reward = surpass_min_phi.sum() + surpass_max_phi.sum()

        # punish w.r.t to large changes to the torque
        action_norm = np.abs(np_torque[:-1] - action)
        surpass_torque_diff = np.where(action_norm > self.max_action_diff, np.abs(action), 0)
        torque_rel_size = len(surpass_torque_diff.nonzero()[0])
        torque_reward = surpass_torque_diff.sum()

        tot = lift_rel_size + phi_rel_size + torque_rel_size
        reward = (lift_rel_size * lift_reward - (phi_rel_size * phi_reward) - (torque_rel_size * torque_reward)) / tot

        # UPDATING THE TIME WINDOW AND INITIAL CONDITION
        self.simulation.set_time(self.simulation.end_t, self.simulation.end_t + self.step_time)
        self.simulation.set_init_cond(last_phi, last_phi_dot)

        self.info = {
            'iter': self.iters,
            'state': np_phi[-1],
            'action': np_torque[-1],
            'lift_reward': lift_reward,
            'lift_rel_size': lift_rel_size,
            'angle_reward': phi_reward,
            'phi_rel_size': phi_rel_size,
            'torque_reward': torque_reward,
            'torque_rel_size': torque_rel_size,
            'total_reward': reward.item()
        }
        if self.iters % 100 == 0:
            self.pretty_print_info()

        obs = {'phi': np_phi, 'torque': np_torque}
        return obs, reward.item(), done, self.info

    def pretty_print_info(self) -> None:
        """
        a friendly function that prints the information of the environment
        """
        lift_rel = self.info['lift_rel_size']
        phi_rel = self.info['phi_rel_size']
        torque_rel = self.info['torque_rel_size']
        tot = lift_rel + phi_rel + torque_rel

        state = self.info['state']
        state_in_range = "OK" if self.min_phi <= state <= self.max_phi else "BAD"
        print(f"[{self.info['iter']}] |"
              f" s={state:.2f} ({state_in_range}) |"
              f" a={self.info['action']:.4f} |"
              f" r_LIFT= {self.info['lift_reward']:.2f} ({lift_rel / tot:.2f}) |"
              f" r_STATE={self.info['angle_reward']:.2f} ({phi_rel / tot:.2f}) |"
              f" r_ACTION={self.info['torque_reward'] :.2f} ({torque_rel / tot:.2f}) |"
              f" r_TOTAL={self.info['total_reward']:.2f} |")

    def render(self, mode="human"):
        self.pretty_print_info()

    def reset(self):
        """
        resets the environment with new state and action histories
        :return:
        """
        zero_history = [0.0] * (self.history_size - 1)
        self.phi_history_deque = deque(zero_history, maxlen=self.history_size)
        self.torque_history_deque = deque(zero_history, maxlen=self.history_size)

        random_phi = round(random.uniform(self.min_phi, self.max_phi), ndigits=2)
        random_torque = round(random.uniform(self.min_torque, self.max_torque), ndigits=2)

        self.phi_history_deque.append(random_phi)
        self.torque_history_deque.append(random_torque)

        obs = {'phi': np.array(self.phi_history_deque, dtype=np.float32),
               'torque': np.array(self.torque_history_deque, dtype=np.float32)}

        self.rounds = 20

        return obs
