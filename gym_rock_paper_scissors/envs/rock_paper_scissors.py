import gym
import numpy as np
from gym import spaces

ROCK = 0
PAPER = 1
SCISSORS = 2
RENDER_MAP = {0: "@", 1: "#", 2: "%"}

class RockPaperScissorsBaseEnv(gym.Env):
    optimal_winning_rate = None
    def __init__(self) -> None:
        self.action_space = spaces.Discrete(3)

        # previous user action, previous env action
        self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))

    def step(self, action):
        info = {}
        env_action = self.env_policy(self.prev_obs)  # FIXME: concat prev_obs and prev_action
        if action == ROCK and env_action == SCISSORS:
            reward = 1
        elif action == SCISSORS and env_action == ROCK:
            reward = -1
        elif env_action < action:
            reward = 1
        elif action == env_action:
            reward = 0
        else:
            reward = -1
        self.prev_obs = env_action
        self.prev_action = action
        done = False if reward == 0 else True
        state = np.array([action, self.prev_obs])
        return state, reward, done, info

    def reset(self):
        self.prev_obs = None

    def render(self, mode="human"):
        user = RENDER_MAP[self.prev_action]
        computer = RENDER_MAP[self.prev_obs]
        print(
            "USER vs COMP\n",
            f" {user}  ||  {computer} ",
        )

    def env_policy(self, obs):
        """return env_action
        """
        raise NotImplementedError

class RockPaperScissorsSequencePolicyEnv(RockPaperScissorsBaseEnv):
    optimal_winning_rate = 1
    def env_policy(self, obs):
        if obs == None:
            env_action = PAPER
        elif obs == ROCK:
            env_action = PAPER
        elif obs == PAPER:
            env_action = SCISSORS
        elif obs == SCISSORS:
            env_action = ROCK
        return env_action

class RockPaperScissorsRandomPolicyEnv(RockPaperScissorsBaseEnv):
    optimal_winning_rate = 1/3
    def env_policy(self, obs):
        return np.random.choice([ROCK, PAPER, SCISSORS])

class RockPaperScissorsBiasedPolicyEnv(RockPaperScissorsBaseEnv):
    optimal_winning_rate = 1/2
    def env_policy(self, obs):
        return np.random.choice([ROCK, PAPER, SCISSORS], p=[0.5, 0.25, 0.25])

class RockPaperScissorsRandomEnv(RockPaperScissorsBaseEnv):
    """TODO: randomly change the computer policy in each episode
    """
    pass

if __name__ == "__main__":
    env = RockPaperScissorsSequencePolicyEnv()
    env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(np.random.choice([ROCK, PAPER, SCISSORS]))
        env.render()
        print(obs, reward, done, info)
