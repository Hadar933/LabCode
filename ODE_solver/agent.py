from matplotlib import pyplot as plt
import numpy as np
from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

timesteps = 15_000
steps_str = f"{str(timesteps)[:-3]}k_regularized_action"
# %%
env = WingEnv()
check_env(env)
# %%
env.reset()
model = PPO("MlpPolicy", env)

model.learn(total_timesteps=timesteps)
model.save(f"PPO_{steps_str}")
del model
# %%
obs = env.reset()
model = PPO.load(f"PPO_{steps_str}.zip")
R, S, A = [], [], []
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    R.append(reward)
    S.append(obs)
    A.append(action)
    if done:
        obs = env.reset()
env.close()
# %%
fig, axs = plt.subplots(3, 1)
fig.set_size_inches(18, 8)

fig.suptitle(
    f"Reward, Action and State (Top to Bottom) - PP0 {steps_str} steps (SB3)"
)
axs[0].plot(range(1000), R, linewidth=0.8)
axs[0].set(ylabel=r"Reward [Arb.U]")

axs[1].plot(range(1000), A, linewidth=0.8)
axs[1].set(ylabel=r"Action $\tau [Nm]$")

axs[2].plot(range(1000), S, linewidth=0.8)
axs[2].set(ylabel=r"State $\phi [rad]$")
axs[2].axhline(y=np.pi, color='r', linestyle='--')
axs[2].text(x=900, y=np.pi + 0.3, s=r"$\phi=\pi$")
axs[2].axhline(y=0, color='r', linestyle='--')
axs[2].text(x=900, y=0.3, s=r"$\phi=0$")
plt.show()
