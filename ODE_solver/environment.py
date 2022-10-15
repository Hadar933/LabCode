from collections import deque
from gym import Env
from gym.spaces import Box, Dict
import numpy as np
from simulation import RobotSimulation
import random

MIN_TORQUE = -1
MAX_TORQUE = -1
MIN_PHI = 0
MAX_PHI = np.pi
HISTORY_SIZE = 10
STEP_TIME = 0.01
STEPS_PER_EPISODE = 20
MAX_APPROX_TORQUE = 0.02
ACTION_ERROR_PERCENTAGE = 0.05
INITIAL_PHI0 = 0
INITIAL_PHI_DOT0 = 2e-4
LIFT_WEIGHT = 1
PHI_WEIGHT = 1
TORQUE_WEIGHT = 1


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
        :param min_torque: the minimal torque we can apply (+-1 is considered +-90 deg)
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
            "phi": Box(low=np.array([-np.inf] * self.history_size, dtype=np.float32),
                       high=np.array([np.inf] * self.history_size), dtype=np.float32),
            'torque': Box(low=np.array([min_torque] * self.history_size, dtype=np.float32),
                          high=np.array([max_torque] * self.history_size, dtype=np.float32))
        })
        self.simulation: RobotSimulation = RobotSimulation(phi0=INITIAL_PHI0, phi_dot0=INITIAL_PHI_DOT0, start_t=0,
                                                           end_t=self.step_time)
        self.info: dict = {}

    def step(self, action: np.ndarray):
        action *= self.max_approx_torque
        self.n_steps += 1
        done = False if self.steps_per_episode > 0 else True
        self.steps_per_episode -= 1

        # (1) INVOKE SIMULATION:
        self.simulation.set_motor_torque(lambda x: action)
        phi, phi_dot, _, _, _, lift_force, _ = self.simulation.solve_dynamics()

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
        # (3.3) weighted sum
        reward = LIFT_WEIGHT * lift_reward - (PHI_WEIGHT * phi_reward) - (TORQUE_WEIGHT * torque_reward)

        # (4) UPDATING THE TIME WINDOW AND INITIAL CONDITION
        self.simulation.set_init_cond(phi[-1], phi_dot[-1])
        self.simulation.set_time(self.simulation.end_t, self.simulation.end_t + self.step_time)

        self.info = {'iter': self.n_steps, 'state': np_phi[-1], 'action': np_torque[-1],
                     'lift_reward': LIFT_WEIGHT * lift_reward, 'angle_reward': PHI_WEIGHT * phi_reward,
                     'torque_reward': TORQUE_WEIGHT * torque_reward, 'total_reward': reward.item()}

        if self.n_steps % 100 == 0: self._pretty_print_info()

        obs = {'phi': np_phi, 'torque': np_torque}
        return obs, reward.item(), done, self.info

    def _pretty_print_info(self) -> None:
        """
        a friendly function that prints the information of the environment
        """
        state = self.info['state']
        state_in_range = "OK" if self.min_phi <= state <= self.max_phi else "BAD"
        print(f"[{self.info['iter']}] |"
              f" s={state:.2f} ({state_in_range}) |"
              f" a={self.info['action']:.4f} |"
              f" r_LIFT= {self.info['lift_reward']:.2f} |"
              f" r_STATE={self.info['angle_reward']:.2f} |"
              f" r_ACTION={self.info['torque_reward'] :.2f} |"
              f" r_TOTAL={self.info['total_reward']:.2f} |")

    def render(self, mode="human"):
        self._pretty_print_info()

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

        self.steps_per_episode = STEPS_PER_EPISODE

        return obs
