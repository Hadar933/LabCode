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
        if self.n_calls % self._log_freq == 0:
            r_lift = self.env.info[REWARD_KEY][LIFT_REWARD_KEY]
            r_action = self.env.info[REWARD_KEY][TORQUE_REWARD_KEY]
            r_phi = self.env.info[REWARD_KEY][ANGLE_REWARD_KEY]
            self.tb_formatter.writer.add_scalar(f'check_info/{LIFT_REWARD_KEY}', r_lift, self.num_timesteps)
            self.tb_formatter.writer.add_scalar(f'check_info/{TORQUE_REWARD_KEY}', r_action, self.num_timesteps)
            self.tb_formatter.writer.add_scalar(f'check_info/{ANGLE_REWARD_KEY}', r_phi, self.num_timesteps)
            self.tb_formatter.writer.flush()


# class _TensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """
#
#     def __init__(self, environment: WingEnv, verbose=0):
#         super(_TensorboardCallback, self).__init__(verbose)
#         self.env = environment
#
#     def _on_step(self) -> bool:
#         """
#         every model step this function is being called and saves some relevant model values
#         :return:
#         """
#         phi = self.env.info[STATE_KEY]
#         action = self.env.info[ACTION_KEY]
#         self.logger.record("phi", phi)
#         self.logger.record("action", action)
#         return True


def train_model_and_save(env: WingEnv, steps_to_train: int, name: str, use_tensorboard_in_colab: bool) -> None:
    """
    performs model training with (in colab) or without (on pc) tensorboard support
    """
    print("Training Model")
    env.reset()
    if use_tensorboard_in_colab:
        model = PPO("MultiInputPolicy", env, device='cuda', tensorboard_log='/content/tensorboard')
        model.learn(total_timesteps=steps_to_train, callback=SummaryWriterCallback(env))
    else:
        model = PPO("MultiInputPolicy", env)
        model.learn(total_timesteps=steps_to_train)
    model.save(name)
    del model


def load_model_and_invoke(env: WingEnv, name: str, n_steps: int):
    """
    loads the trained model and tracks relevant data we wish to plot
    """
    obs = env.reset()
    model = PPO.load(f"{name}.zip")
    phi_arr, phidot_arr, lift_force_arr, time_arr, action_arr = [np.array([], dtype=np.float32) for _ in range(5)]
    r_lift_arr, r_phi_arr, r_torque_arr, r_power_arr, r_tot_arr = [np.array([], dtype=np.float32) for _ in range(5)]

    for i in range(n_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

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
    env.close()
    sim_arr = [phi_arr, phidot_arr, lift_force_arr, action_arr, time_arr]
    reward_arr = [r_lift_arr, r_phi_arr, r_torque_arr, r_power_arr, r_tot_arr]
    return sim_arr, reward_arr


def plot_steps(sim_data, reward_data, time_arr, name, save):
    num_plots = len(sim_data) + 1
    fig, axs = plt.subplots(num_plots, 1)
    fig.set_size_inches(12, 8)
    fig.suptitle(name)
    cmap = plt.cm.get_cmap('twilight_shifted', num_plots - 1)
    for i, (sim_arr, unit) in enumerate(
            zip(sim_data, [r'$\phi$ [rad]', r'$\dot\phi$ [rad/sec]', r'$F_{lift}$ [N]', r'$\tau$ [Nm]'])):
        axs[i].plot(time_arr, sim_arr, linewidth=1.2, c=cmap(i), marker='o', markersize=2.5)
        axs[i].set(ylabel=unit)
        axs[i].grid()
    axs[i].set(xlabel='time [sec]')
    if save: plt.savefig(f"{name}_sim_fig")

    reward_idx = i + 1
    for reward_arr in reward_data:
        axs[reward_idx].plot(range(len(reward_arr)), reward_arr, linewidth=1.2, marker='o', markersize=2.2)
    axs[reward_idx].set(ylabel='reward [Arb.U]', xlabel='Step [#]')
    axs[reward_idx].grid()
    axs[reward_idx].legend([r'$r_{lift}$', r'$r_{\phi}$', r'$r_{\tau}$', r'$r_P$', r'$r_{tot}$'])
    if save: plt.savefig(f"{name}_reward_fig")

    plt.show()


if __name__ == '__main__':
    in_colab = 'COLAB_GPU' in os.environ
    print(f"Working in colab: {in_colab}")
    n_train_steps = 1_000_000
    invoke_steps = 200
    model_name = f"PPO_{str(n_train_steps)[:-3]}k"
    plot_after_invocation = True

    wing_env = WingEnv()
    print("Checking environment")
    check_env(wing_env)

    if f"{model_name}.zip" not in os.listdir():  # need to train
        print(f"Training model {model_name}...")
        train_model_and_save(wing_env, n_train_steps, model_name, in_colab)
    print("Invoking model")
    sim_data, reward_data = load_model_and_invoke(wing_env, model_name, invoke_steps)
    time = sim_data.pop()
    if plot_after_invocation: plot_steps(sim_data, reward_data, time, model_name, in_colab)
