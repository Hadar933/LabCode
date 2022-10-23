from matplotlib import pyplot as plt
import numpy as np
from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import os


class _TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, environment: WingEnv, verbose=0):
        super(_TensorboardCallback, self).__init__(verbose)
        self.env = environment

    def _on_step(self) -> bool:
        """
        every model step this function is being called and saves some relevant model values
        :return:
        """
        phi = self.env.info['state']
        action = self.env.info['action']
        self.logger.record("phi", phi)
        self.logger.record("action", action)
        return True


def train_model_and_save(env: WingEnv, steps_to_train: int, name: str, use_tensorboard_in_colab: bool) -> None:
    print("Training Model")
    env.reset()
    if use_tensorboard_in_colab:
        model = PPO("MultiInputPolicy", env, device='cuda', tensorboard_log='/content/tensorboard')
        model.learn(total_timesteps=steps_to_train, callback=_TensorboardCallback(env))
    else:
        model = PPO("MultiInputPolicy", env)
        model.learn(total_timesteps=steps_to_train)
    model.save(name)
    del model


def load_model_and_invoke(env: WingEnv, name: str, n_steps: int):
    obs = env.reset()
    model = PPO.load(f"{name}.zip")
    phi_arr = np.array([], dtype=np.float32)
    phi_dot_arr = np.array([], dtype=np.float32)
    lift_force_arr = np.array([], dtype=np.float32)
    time_arr = np.array([], dtype=np.float32)
    action_arr = np.array([], dtype=np.float32)
    for i in range(n_steps):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        phi, phi_dot, lift_force, time, action = info['curr_simulation_output'].values()
        phi_arr = np.append(phi_arr, phi)
        phi_dot_arr = np.append(phi_dot_arr, phi_dot)
        lift_force_arr = np.append(lift_force_arr, lift_force)
        time_arr = np.append(time_arr, time)
        action_arr = np.append(action_arr, action)
        # if done: obs = env.reset()
    env.close()
    return [phi_arr, phi_dot_arr, lift_force_arr, action_arr, time_arr]


def plot_steps(all_arrays, time_arr, name, save):
    num_plots = len(all_arrays)
    fig, axs = plt.subplots(num_plots, 1)
    fig.set_size_inches(12, 6)
    fig.suptitle(name)
    cmap = plt.cm.get_cmap('twilight_shifted', num_plots)
    for i, (arr, name) in enumerate(
            zip(all_arrays, [r"$\phi$ [rad]", r"$\dot\phi$ [rad/sec]", r"$F_{LIFT}$ [N]", r"$\tau$ [Nm]"])):
        axs[i].plot(time_arr, arr, linewidth=1.2, c=cmap(i), marker='o', markersize=2.5)
        axs[i].set(ylabel=name)
        axs[i].grid()
    axs[num_plots - 1].set(xlabel='time [sec]')

    if save: plt.savefig(f"{name}_fig")
    plt.show()


if __name__ == '__main__':
    in_colab = 'COLAB_GPU' in os.environ
    print(f"Working in colab: {in_colab}")
    n_train_steps = 500_000
    invoke_steps = 50
    model_name = f"PPO_{str(n_train_steps)[:-3]}k"
    plot_after_invocation = True

    wing_env = WingEnv()
    print("Checking environment")
    check_env(wing_env)

    if f"{model_name}.zip" not in os.listdir():  # need to train
        print("Training model...")
        train_model_and_save(wing_env, n_train_steps, model_name, in_colab)
    print("Invoking model")
    all_arrays = load_model_and_invoke(wing_env, model_name, invoke_steps)
    time = all_arrays.pop()
    if plot_after_invocation: plot_steps(all_arrays, time, model_name, in_colab)

    phi_arr, phi_dot_arr, lift_force_arr, action_arr = all_arrays
