from collections import deque

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
        # initialize history as deques (FILO) of fixed size
        self.history_size = 10
        self.phi_history = deque([0.0] * self.history_size, maxlen=self.history_size)
        self.torque_history = deque([0.0] * self.history_size, maxlen=self.history_size)

        self.rounds = 20  # arbitrary number of rounds
        self.iters = 0
        self.time_window = 0.02
        self.max_approx_torque = 0.02
        self.max_action_diff = 0.05 * self.max_approx_torque

        self.max_torque = max_torque
        self.min_torque = min_torque
        self.max_phi = max_phi
        self.min_phi = min_phi

        self.action_space = Box(np.array([min_torque], dtype=np.float32), np.array([max_torque], dtype=np.float32))
        self.observation_space = Dict({
            "phi": Box(low=np.array([-np.inf] * self.history_size, dtype=np.float32),
                       high=np.array([np.inf] * self.history_size), dtype=np.float32),
            'torque': Box(low=np.array([min_torque] * self.history_size, dtype=np.float32),
                          high=np.array([max_torque] * self.history_size, dtype=np.float32))
        })
        self.collected_reward = []
        self.simulation = RobotSimulation()
        self.info = {}

    def delete_history(self):
        self.phi_history = np.array([])
        self.torque_history = np.array([])

    def step(self, action: np.ndarray):
        action *= self.max_approx_torque
        self.iters += 1
        done = False if self.rounds > 0 else True
        self.rounds -= 1

        # invoke simulation:
        self.simulation.set_motor_torque(lambda x: action)
        self.simulation.solve_dynamics()
        phi, phi_dot = self.simulation.solution
        last_phi, last_phi_dot = phi[-1], phi_dot[-1]

        # updating history stacks:
        for item in phi[-self.history_size:]: self.phi_history.append(item)  # appending (FILO) the last elements
        self.torque_history.append(action.item())

        # working with stacks converted to np arrays
        np_phi = np.array(self.phi_history).astype(np.float32)
        np_torque = np.array(self.torque_history).astype(np.float32)

        # calculate the reward:
        lift_reward = self.simulation.lift_force(phi_dot).mean()  # TODO: why does this gets negative values?
        lift_rel_size = len(phi_dot)

        # punish w.r.t bad phi values
        surpass_max_phi = np.where(np_phi > self.max_phi, np.abs(np_phi - self.max_phi), 0)
        surpass_min_phi = np.where(np_phi < self.min_phi, np.abs(np_phi - self.min_phi), 0)
        phi_rel_size = len(np.nonzero(surpass_min_phi)[0]) + len(
            np.nonzero(surpass_max_phi)[0])  # TODO: this can be zero
        phi_reward = surpass_min_phi.sum() + surpass_max_phi.sum()

        # punish w.r.t to large changes to the torque
        action_norm = np.abs(np_torque[:-1] - action)
        surpass_torque_diff = np.where(action_norm > self.max_action_diff, action_norm, 0)
        torque_rel_size = len(surpass_torque_diff.nonzero()[0])
        torque_reward = surpass_torque_diff.sum()

        reward = (lift_rel_size * lift_reward - phi_rel_size * phi_reward - torque_rel_size * torque_reward) / (
                lift_rel_size + phi_rel_size + torque_rel_size)
        self.collected_reward.append(reward)

        # update time window and init cond for next iterations
        self.simulation.set_time(self.simulation.end_t, self.simulation.end_t + self.time_window)
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
              f" r_LIFT= {self.info['lift_reward']:.2f} ({lift_rel/tot:.2f}) |"
              f" r_STATE={self.info['angle_reward']:.2f} ({phi_rel/tot:.2f}) |"
              f" r_ACTION={self.info['torque_reward'] :.2f} ({torque_rel/tot:.2f}) |"
              f" r_TOTAL={self.info['total_reward']:.2f} |")

    def render(self, mode="human"):
        self.pretty_print_info()

    def reset(self):

        self.phi_history = deque([0] * self.history_size, maxlen=self.history_size)
        self.torque_history = deque([0] * self.history_size, maxlen=self.history_size)

        zero_history = [0.0] * (self.history_size - 1)
        random_phi = round(random.uniform(self.min_phi, self.max_phi), ndigits=2)
        random_torque = round(random.uniform(self.min_torque, self.max_torque), ndigits=2)

        self.phi_history.append(random_phi)
        self.torque_history.append(random_torque)

        phi = np.array(zero_history + [random_phi], dtype=np.float32)
        torque = np.array(zero_history + [random_torque], dtype=np.float32)
        obs = {'phi': phi, 'torque': torque}

        self.rounds = 20
        self.collected_reward = []

        return obs
