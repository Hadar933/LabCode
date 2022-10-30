from collections import deque
from typing import Tuple

from gym import Env
from gym.spaces import Box, Dict
import numpy as np
from simulation import RobotSimulation
import random
from constants import *


class WingEnv(Env):
    def __init__(self,
                 min_torque: float = MIN_TORQUE, max_torque: float = MAX_TORQUE,
                 min_phi: float = MIN_PHI, max_phi: float = MAX_PHI,
                 history_size: int = HISTORY_SIZE,
                 step_time: int = STEP_TIME,
                 steps_per_episode: int = STEPS_PER_EPISODE,
                 max_approx_torque: int = MAX_APPROX_TORQUE):
        """
        the action space is a continuous torque value
        the observation space is currently a continuous value for phi (later I will add psi, theta)
        :param min_torque: the minimal torque we can apply
        :param max_torque: the maximal torque we can apply
        :param min_phi: the minimal rotation angle
        :param max_phi: the maximal rotation angle
        """
        # initialize history as deque (FILO) of fixed size
        self.history_size: int = history_size
        self.phi_history_deque: deque = deque([0.0] * self.history_size, maxlen=self.history_size)
        self.torque_history_deque: deque = deque([0.0] * self.history_size, maxlen=self.history_size)

        self.steps_per_episode: int = steps_per_episode  # time of each episode is rounds * step_time
        self.n_steps: int = 0
        self.step_time: float = step_time  # seconds
        self.max_approx_torque: float = max_approx_torque
        self.max_action_diff: float = ACTION_ERROR_PERCENTAGE * self.max_approx_torque

        self.max_torque: float = max_torque
        self.min_torque: float = min_torque
        self.max_phi: float = max_phi
        self.min_phi: float = min_phi

        self.action_space: Box = Box(np.array([min_torque], dtype=np.float32), np.array([max_torque], dtype=np.float32))
        self.observation_space: Dict = Dict({
            PHI_KEY: Box(low=np.array([-np.inf] * self.history_size, dtype=np.float32),
                         high=np.array([np.inf] * self.history_size), dtype=np.float32),
            TORQUE_KEY: Box(low=np.array([min_torque] * self.history_size, dtype=np.float32),
                            high=np.array([max_torque] * self.history_size, dtype=np.float32))
        })
        self.simulation: RobotSimulation = RobotSimulation(phi0=INITIAL_PHI0, phi_dot0=INITIAL_PHI_DOT0,
                                                           start_t=0, end_t=self.step_time)

        self.info: dict = {}

    def step(self, action: np.ndarray):
        """
        takes a step by solving the ode for a given time window and calculates the resulted reward
        :param action: a torque value the motor applies
        :return: observation,reward value, done boolean and info dictionary
        """
        action *= self.max_approx_torque
        self.n_steps += 1
        done = False if self.steps_per_episode > 0 else True
        self.steps_per_episode -= 1

        # (1) INVOKE SIMULATION:
        self.simulation.set_motor_torque(lambda x: action)
        phi, phi_dot, _, _, time, lift_force, _ = self.simulation.solve_dynamics()

        # (2) UPDATE STATE & ACTION HISTORY:
        for item in phi[-self.history_size:]: self.phi_history_deque.append(item)  # appending (FILO) the last elements
        self.torque_history_deque.append(action.item())
        # (2.1) working with stacks converted to np arrays
        np_phi = np.array(self.phi_history_deque).astype(np.float32)
        np_torque = np.array(self.torque_history_deque).astype(np.float32)

        # (3) CALCULATING THE REWARD:
        lift_reward = lift_force.mean()
        # (3.1) punish w.r.t bad phi values:
        surpass_max_phi = np.where(np_phi > self.max_phi, np.abs(np_phi - self.max_phi), 0)
        surpass_min_phi = np.where(np_phi < self.min_phi, np.abs(np_phi - self.min_phi), 0)
        phi_reward = surpass_min_phi.sum() + surpass_max_phi.sum()
        # (3.2) punish w.r.t to large changes to the torque:
        torque_reward = np.sum(np.diff(np_torque) ** 2)
        # (3.3) adding power reward
        power_reward = np.max([0, np.mean(action * phi_dot)])
        # (3.4) weighted sum
        reward = LIFT_WEIGHT * lift_reward - (PHI_WEIGHT * phi_reward) - (TORQUE_WEIGHT * torque_reward) - (
                POWER_WEIGHT * power_reward)

        # (4) UPDATING THE TIME WINDOW AND INITIAL CONDITION
        self.simulation.set_init_cond(phi[-1], phi_dot[-1])
        self.simulation.set_time(self.simulation.end_t, self.simulation.end_t + self.step_time)

        self._update_env_info(action, lift_force, lift_reward, np_phi, np_torque, phi, phi_dot, phi_reward, reward,
                              time, torque_reward, power_reward)

        if self.n_steps % 100 == 0: self._pretty_print_info()

        obs = {PHI_KEY: np_phi, TORQUE_KEY: np_torque}
        return obs, reward.item(), done, self.info

    def _update_env_info(self, action, lift_force, lift_reward, np_phi, np_torque, phi, phi_dot, phi_reward, reward,
                         time, torque_reward, power_reward):
        """
        updates relevant model information into the self.info variable
        TODO: ideally this is only relevant AFTER the training phase
        """
        self.info[ITERATION_KEY] = self.n_steps
        self.info[STATE_KEY] = np_phi[-1]
        self.info[ACTION_KEY] = np_torque[-1]
        self.info[STEP_SIMULATION_OUT_KEY] = {
            PHI_KEY: phi,
            PHI_DOT_KEY: phi_dot,
            LIFT_FORCE_KEY: lift_force,
            TIME_KEY: time,
            ACTION_KEY: action * np.ones(time.shape[0])}
        self.info[REWARD_KEY] = {
            LIFT_REWARD_KEY: LIFT_WEIGHT * lift_reward,
            ANGLE_REWARD_KEY: PHI_WEIGHT * phi_reward,
            TORQUE_REWARD_KEY: TORQUE_WEIGHT * torque_reward,
            POWER_REWARD_KEY: POWER_WEIGHT * power_reward,
            TOTAL_REWARD_KEY: reward.item()}

    def _pretty_print_info(self) -> None:
        """
        a friendly function that prints the information of the environment
        """
        print(f"[{self.info[ITERATION_KEY]}] |"
              f" s={self.info[STATE_KEY]:.2f} |"
              f" a={self.info[ACTION_KEY]:.4f} |"
              f" r_LIFT= {self.info[REWARD_KEY][LIFT_REWARD_KEY]:.2f} |"
              f" r_STATE={self.info[REWARD_KEY][ANGLE_REWARD_KEY]:.2f} |"
              f" r_ACTION={self.info[REWARD_KEY][TORQUE_REWARD_KEY] :.2f} |"
              f" r_POWER={self.info[REWARD_KEY][POWER_REWARD_KEY] :.2f} |"
              f" r_TOTAL={self.info[REWARD_KEY][TOTAL_REWARD_KEY]:.2f} |"
              )

    def render(self, mode="human") -> None:
        self._pretty_print_info()

    def reset(self) -> dict[str, np.ndarray]:
        """
        resets the environment with new state and action histories
        :return: an observation dictionary
        """
        zero_history = [0.0] * (self.history_size - 1)
        self.phi_history_deque = deque(zero_history, maxlen=self.history_size)
        self.torque_history_deque = deque(zero_history, maxlen=self.history_size)

        random_phi = round(random.uniform(self.min_phi, self.max_phi), ndigits=2)
        random_torque = round(random.uniform(self.min_torque, self.max_torque), ndigits=2)

        self.phi_history_deque.append(random_phi)
        self.torque_history_deque.append(random_torque)

        obs = {PHI_KEY: np.array(self.phi_history_deque, dtype=np.float32),
               TORQUE_KEY: np.array(self.torque_history_deque, dtype=np.float32)}

        self.steps_per_episode = STEPS_PER_EPISODE

        return obs
