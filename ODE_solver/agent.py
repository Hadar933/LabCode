import sys
from typing import List

from matplotlib import pyplot as plt
import numpy as np
from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3 import PPO
import os
from constants import *


class SummaryWriterCallback(BaseCallback):
    """
    a class that tracks events in tensorboard and adds them as plots.
    """

    def __init__(self, env: WingEnv):
        super(SummaryWriterCallback, self).__init__()
        self.env = env

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(frmtr for frmtr in output_formats if isinstance(frmtr, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        """
        every step, we track the reward values
        :return:
        """
        if self.n_calls % self._log_freq == 0:
            r_lift = self.env.info[REWARD_KEY][LIFT_REWARD_KEY]
            r_action = self.env.info[REWARD_KEY][TORQUE_REWARD_KEY]
            r_phi = self.env.info[REWARD_KEY][ANGLE_REWARD_KEY]
            self.tb_formatter.writer.add_scalar(f'check_info/{LIFT_REWARD_KEY}', r_lift, self.num_timesteps)
            self.tb_formatter.writer.add_scalar(f'check_info/{TORQUE_REWARD_KEY}', r_action, self.num_timesteps)
            self.tb_formatter.writer.add_scalar(f'check_info/{ANGLE_REWARD_KEY}', r_phi, self.num_timesteps)
            self.tb_formatter.writer.flush()


class Agent:
    def __init__(self, model_name: str):
        self.env: WingEnv = WingEnv()
        check_env(self.env)
        print("Checking environment: OK")
        self.model_name = model_name
        self.in_colab = 'google.colab' in sys.modules
        print(f"Working in colab: {self.in_colab}")

    def train(self, n_train_steps: int) -> None:
        """
        performs model training with (in colab) or without (on pc) tensorboard support
        """
        if f"{self.model_name}.zip" in os.listdir():
            print(f"Already trained model {self.model_name}. use Agent.load_model instead")
            return

        print(f"Training Model {self.model_name}")
        self.env.reset()
        if self.in_colab:
            model = PPO("MultiInputPolicy", self.env, device='cuda', tensorboard_log='/content/tensorboard')
            model.learn(total_timesteps=n_train_steps, callback=SummaryWriterCallback(self.env))
        else:  # local
            model = PPO("MultiInputPolicy", self.env)
            model.learn(total_timesteps=n_train_steps)
        model.save(self.model_name)
        del model

    def load(self, n_test_steps: int):
        """
        loads the trained model and tracks relevant data we wish to plot
        """
        print(f"Loading model {self.model_name}")
        obs = self.env.reset()
        model = PPO.load(f"{self.model_name}.zip")
        phi_arr, phidot_arr, lift_force_arr, time_arr, action_arr = [np.array([], dtype=np.float32) for _ in range(5)]
        r_lift_arr, r_phi_arr, r_torque_arr, r_power_arr, r_tot_arr = [np.array([], dtype=np.float32) for _ in range(5)]
        print(f"Running {n_test_steps} model test steps")
        for i in range(n_test_steps):
            action, _states = model.predict(obs)
            obs, reward, done, info = self.env.step(action)

            phi, phi_dot, lift_force, time, action = info[STEP_SIMULATION_OUT_KEY].values()
            phi_arr = np.append(phi_arr, phi)
            phidot_arr = np.append(phidot_arr, phi_dot)
            lift_force_arr = np.append(lift_force_arr, lift_force)
            time_arr = np.append(time_arr, time)
            action_arr = np.append(action_arr, action)

            lift_reward, phi_reward, torque_reward, power_reward, total_reward = info[REWARD_KEY].values()
            r_lift_arr = np.append(r_lift_arr, lift_reward)
            r_phi_arr = np.append(r_phi_arr, phi_reward)
            r_torque_arr = np.append(r_torque_arr, torque_reward)
            r_power_arr = np.append(r_power_arr, power_reward)
            r_tot_arr = np.append(r_tot_arr, total_reward)

            # if done: obs = env.reset()
        self.env.close()
        sim_arr = [phi_arr, phidot_arr, lift_force_arr, action_arr, time_arr]
        reward_arr = [r_lift_arr, r_phi_arr, r_torque_arr, r_power_arr, r_tot_arr]
        return sim_arr, reward_arr

    def plot_steps(self, sim_data: List[np.ndarray],
                   reward_data: List[np.ndarray],
                   time_arr: np.ndarray,
                   ) -> None:
        """
        plots the simulation returned values and the value of the loss function
        :param sim_data: a list of relevant data arrays like phi,phi dot, etc...
        :param reward_data: a list of all the components in the reward
        :param time_arr: will be used as x-axis
        """
        # plots simulation data:
        num_plots = len(sim_data) + 1
        fig, axs = plt.subplots(num_plots, 1)
        fig.set_size_inches(12, 8)
        fig.suptitle(self.model_name)
        cmap = plt.cm.get_cmap('twilight_shifted', num_plots - 1)
        units_arr = [r'$\phi$ [rad]', r'$\dot\phi$ [rad/sec]', r'$F_{lift}$ [N]', r'$\tau$ [Nm]']
        for i, (sim_arr, unit) in enumerate(zip(sim_data, units_arr)):
            axs[i].plot(time_arr, sim_arr, linewidth=1.2, c=cmap(i), marker='o', markersize=1.2)
            axs[i].set(ylabel=unit)
            axs[i].grid()
        axs[i].set(xlabel='time [sec]')

        # plots reward:
        reward_idx = i + 1
        for reward_arr in reward_data:
            axs[reward_idx].plot(range(len(reward_arr)), reward_arr, linewidth=1.2, marker='o', markersize=1.2)
        axs[reward_idx].set(ylabel='reward [Arb.U]', xlabel='Step [#]')
        axs[reward_idx].grid()
        axs[reward_idx].legend([r'$r_{lift}$', r'$r_{\phi}$', r'$r_{\tau}$', r'$r_P$', r'$r_{tot}$'])
        if self.in_colab: plt.savefig(f"{self.model_name}_reward_fig")

        plt.show()


def main():
    n_train_steps = 180_000
    n_test_steps = 100
    model_name = f"PPO_{str(n_train_steps)[:-3]}k"
    plot_after_invocation = True

    A = Agent(model_name)
    A.train(n_train_steps)
    sim_data, reward_data = A.load(n_test_steps)
    time = sim_data.pop()
    if plot_after_invocation: A.plot_steps(sim_data, reward_data, time)


if __name__ == '__main__':
    main()
