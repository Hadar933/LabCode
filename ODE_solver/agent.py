from matplotlib import pyplot as plt
import numpy as np
from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

timesteps = 20_000
steps_str = f"{str(timesteps)[:-3]}k"
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
fig.set_size_inches(7, 9)

fig.suptitle(
    f"Reward, Action and State (Top to Bottom) - PP0 {steps_str}"
)
axs[0].scatter(range(1000), R, s=0.3)
axs[0].set(ylabel=r"$Arb.U$")

axs[1].scatter(range(1000), A, s=0.3)
axs[1].set(ylabel=r"$\tau [Nm]$")

axs[2].scatter(range(1000), S, s=0.3)
axs[2].set(ylabel=r"$\phi [rad]$")
axs[2].axhline(y=np.pi, color='r', linestyle='--')
axs[2].text(x=900, y=np.pi + 0.3, s=r"$\phi=\pi$")
axs[2].axhline(y=0, color='r', linestyle='--')
axs[2].text(x=900, y=0.3, s=r"$\phi=0$")
plt.show()
