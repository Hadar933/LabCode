from matplotlib import pyplot as plt
import numpy as np
from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import os


def automated_env_checker(env):
    print("Testing Environment")
    check_env(env)
    print("Finished Testing Environment")


def train_model(env, timesteps, steps_str, use_tensorboarad_in_colab):
    print("Training Model")
    env.reset()
    if use_tensorboarad_in_colab:
        model = PPO("MultiInputPolicy", env, device='cuda', tensorboard_log='/content/tensorboard')
    else:
        model = PPO("MultiInputPolicy", env)
    model.learn(total_timesteps=timesteps)
    model.save(f"PPO_{steps_str}")
    del model
    print("Finished Training Model")


def simulate_steps(env, steps_str, n_steps):
    obs = env.reset()
    model = PPO.load(f"PPO_{steps_str}.zip")
    R, S, A, time = [0.0], [0.0], [0.0], [0.0]

    for i in range(n_steps - 1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        R.append(reward)
        S.append(obs['phi'][-1])
        A.append(action.item())
        time.append((i + 1) * env.time_window)
        if done:
            obs = env.reset()
    env.close()
    return R, S, A, time


def plot_steps(R, S, A, time, steps_str, save):
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(18, 8)
    fig.suptitle(f"PP0 {steps_str} steps (SB3)")
    axs[0].plot(time, R, linewidth=0.8)
    axs[0].set(ylabel=r"Reward [Arb.U]")
    axs[1].plot(time, A, linewidth=0.8)
    axs[1].set(ylabel=r"Action $\tau [Nm]$")
    axs[2].plot(time, S, linewidth=0.8)
    axs[2].set(ylabel=r"State $\phi [rad]$")
    axs[2].set(xlabel='time [sec]')
    axs[2].axhline(y=np.pi, color='r', linestyle='--')
    axs[2].text(x=900, y=np.pi + 0.3, s=r"$\phi=\pi$")
    axs[2].axhline(y=0, color='r', linestyle='--')
    axs[2].text(x=900, y=0.3, s=r"$\phi=0$")
    if save: plt.savefig(f"PP0 {steps_str} steps (SB3)")
    plt.show()


if __name__ == '__main__':
    in_colab = 'COLAB_GPU' in os.environ
    n_steps = 100_000
    steps_str = f"{str(n_steps)[:-3]}k_regularized_action"
    env = WingEnv()
    automated_env_checker(env)
    train_model(env, n_steps, steps_str, in_colab)
    R, S, A, time = simulate_steps(env, steps_str, 1000)
    plot_steps(R, S, A, time, steps_str, in_colab)
