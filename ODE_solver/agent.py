from matplotlib import pyplot as plt
import numpy as np
from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import os


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, environment: WingEnv, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = environment

    def _on_step(self) -> bool:
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
        model.learn(total_timesteps=steps_to_train, callback=TensorboardCallback(env))
    else:
        model = PPO("MultiInputPolicy", env)
        model.learn(total_timesteps=steps_to_train)
    model.save(name)
    del model


def load_model_and_invoke(env: WingEnv, name: str, n_steps: int):
    obs = env.reset()
    model = PPO.load(f"{name}.zip")
    rewards, states, actions, time = [0.0], [0.0], [0.0], [0.0]

    for i in range(n_steps - 1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(obs['phi'][-1])
        actions.append(action.item())
        time.append((i + 1) * env.step_time)
        # if done: obs = env.reset()
    env.close()
    return rewards, states, actions, time


def plot_steps(rewards, states, actions, time, name, save):
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(18, 8)
    fig.suptitle(name)

    axs[0].plot(time, rewards, linewidth=0.8)
    axs[0].set(ylabel=r"Reward [Arb.U]")

    axs[1].plot(time, actions, linewidth=0.8)
    axs[1].set(ylabel=r"Action $\tau [Nm]$")

    axs[2].plot(time, states, linewidth=0.8)
    axs[2].set(ylabel=r"State $\phi [rad]$")
    axs[2].set(xlabel='time [sec]')
    axs[2].axhline(y=np.pi, color='r', linestyle='--')
    axs[2].text(x=900, y=np.pi + 0.3, s=r"$\phi=\pi$")
    axs[2].axhline(y=0, color='r', linestyle='--')
    axs[2].text(x=900, y=0.3, s=r"$\phi=0$")

    if save: plt.savefig(f"{name}_fig")
    plt.show()


if __name__ == '__main__':
    in_colab = 'COLAB_GPU' in os.environ
    print(f"Working in colab: {in_colab}")
    n_train_steps = 180_000
    invoke_for = 1000
    model_name = f"PPO_{str(n_train_steps)[:-3]}k"
    plot_after_invocation = True

    wing_env = WingEnv()
    print("Checking environment")
    check_env(wing_env)

    if f"{model_name}.zip" not in os.listdir():  # need to train
        print("Training model...")
        train_model_and_save(wing_env, n_train_steps, model_name, in_colab)

    print("Invoking model")
    R, S, A, T = load_model_and_invoke(wing_env, model_name, invoke_for)
    if plot_after_invocation: plot_steps(R, S, A, T, model_name, in_colab)
