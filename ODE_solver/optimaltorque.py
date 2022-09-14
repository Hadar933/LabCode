from matplotlib import pyplot as plt

from environment import WingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

# %%
env = WingEnv()
check_env(env)
# %%
env.reset()
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10_000)
model.save("PPO")
del model
# %%
obs = env.reset()
model = PPO.load("PPO.zip")
R, S, A = [], [], []
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    R.append(reward)
    S.append(obs)
    A.append(action)
    print(f"a={action},s={obs},r={reward}")
    # env.render()
    if done:
        obs = env.reset()
env.close()
# %%
plt.scatter(range(1000), R, s=0.3)
plt.title("Reward")
plt.show()

plt.scatter(range(1000), A, s=0.3)
plt.title("Action")
plt.show()

plt.scatter(range(1000), S, s=0.3)
plt.title("State")
plt.show()
