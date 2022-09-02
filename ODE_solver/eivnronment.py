from gym import Env
from gym.spaces import Box
import numpy as np
from robotsimulation import RobotSimulation


class WingEnv(Env):
    def __init__(self, min_torque=0, max_torque=1, min_phi=0, max_phi=180):
        """
        the action space is a continuous torque value
        the observation space is currently a continuous value for phi (later I will add psi, theta)
        :param min_torque: the minimal torque we can apply
        :param max_torque: the maximal torque we can apply
        :param min_phi: the minimal rotation angle
        :param max_phi: the maximal rotation angle
        """
        self.min_torque = min_torque
        self.max_torque = max_torque
        self.min_phi = min_phi
        self.max_phi = max_phi

        self.action_space = Box(np.array([min_torque]), np.array([max_torque]))
        self.observation_space = Box(np.array([min_phi]), np.array([max_phi]))
        self.state = 0  # some initial phi state
        self.rounds = 20  # number of rounds
        self.collected_reward = []
        self.simulation = RobotSimulation(tau_z=lambda x: self.action)
        self.time_window = 2

    def step(self, action):
        done = False if self.rounds > 0 else True
        self.rounds -= 1
        info = {}

        # calculate the new phi:
        self.simulation.set_tau_z(lambda x: action)
        self.simulation.solve_dynamics()
        sol = self.simulation.solution
        phi = sol[0][-1]
        self.state = phi

        # calculate the reward:
        phi_dot = sol[1][-1]  # we use the velocity at the last time step
        reward = self.simulation.drag_force(phi_dot)
        if phi > self.max_phi or phi < self.min_phi: # we punish angles not in [0,180]
            reward -= 1
        self.collected_reward.append(reward)

        # update time window and init cond for next iterations
        self.simulation.update_time(self.simulation.end_t, self.simulation.end_t + self.time_window)
        self.simulation.update_initial_cond(phi, phi_dot)

        return self.state, reward, done, info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.state = 0
        self.action = 0
        self.rounds = 20
        return self.state


if __name__ == '__main__':
    env = WingEnv()
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print(f"Episode: {episode}. Score: {score}.")
