import gym
import numpy as np
from gym import spaces

ROCK = 0
PAPER = 1
SCISSORS = 2
NULL_ACTION = 3
RENDER_MAP = {0: "@", 1: "#", 2: "%"}


class RockPaperScissorsBaseEnv(gym.Env):
    optimal_winning_rate = None

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(3)

        # previous user action, previous env action
        self.observation_space = spaces.Discrete(4)

    def step(self, action):
        info = {}
        # FIXME: concat prev_obs and prev_action
        env_action = self.env_policy(self.prev_obs)
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
        # state = np.array([action, self.prev_obs])
        obs = np.array([env_action])
        return obs, reward, done, info

    def reset(self):
        self.prev_obs = None
        return np.array([NULL_ACTION])

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

    def __init__(self, start_with=PAPER, other_sequence=False) -> None:
        super().__init__()
        self.start_with = start_with
        self.other_sequence = other_sequence

    def env_policy(self, obs):
        if obs == None:
            env_action = self.start_with
        elif obs == ROCK:
            if self.other_sequence:
                env_action = SCISSORS
            else:
                env_action = PAPER
        elif obs == PAPER:
            if self.other_sequence:
                env_action = ROCK
            else:
                env_action = SCISSORS
        elif obs == SCISSORS:
            if self.other_sequence:
                env_action = PAPER
            else:
                env_action = ROCK
        return env_action


class RockPaperScissorsRandomPolicyEnv(RockPaperScissorsBaseEnv):
    optimal_winning_rate = 1/2  # win + win after draw + ...

    def env_policy(self, obs):
        return np.random.choice([ROCK, PAPER, SCISSORS])


class RockPaperScissorsBiasedPolicyEnv(RockPaperScissorsBaseEnv):
    optimal_winning_rate = 2/3  # win + win after draw + ...

    def __init__(self, biased_by=ROCK) -> None:
        super().__init__()
        self.biased_by = biased_by

    def env_policy(self, obs):
        if self.biased_by == ROCK:
            return np.random.choice([ROCK, PAPER, SCISSORS], p=[0.5, 0.25, 0.25])
        if self.biased_by == PAPER:
            return np.random.choice([ROCK, PAPER, SCISSORS], p=[0.25, 0.5, 0.25])
        if self.biased_by == SCISSORS:
            return np.random.choice([ROCK, PAPER, SCISSORS], p=[0.25, 0.25, 0.5])


class RockPaperScissorsSequencePolicy2Env(RockPaperScissorsBaseEnv):
    optimal_winning_rate = 1

    def __init__(self, start_with=PAPER, other_sequence=False, double_with=ROCK) -> None:
        super().__init__()
        self.start_with = start_with
        self.other_sequence = other_sequence
        self.double_with = double_with
        self.double_flag = False

    def reset(self):
        self.double_flag = False
        return super().reset()

    def env_policy(self, obs):
        if obs == None:
            env_action = self.start_with
        elif obs == ROCK:
            if self.double_with == ROCK and not self.double_flag:
                env_action = ROCK
                self.double_flag = True
            else:
                self.double_flag = False
                if self.other_sequence:
                    env_action = SCISSORS
                else:
                    env_action = PAPER
        elif obs == PAPER:
            if self.double_with == PAPER and not self.double_flag:
                env_action = PAPER
                self.double_flag = True
            else:
                self.double_flag = False
                if self.other_sequence:
                    env_action = ROCK
                else:
                    env_action = SCISSORS
        elif obs == SCISSORS:
            if self.double_with == SCISSORS and not self.double_flag:
                env_action = SCISSORS
                self.double_flag = True
            else:
                self.double_flag = False
                if self.other_sequence:
                    env_action = PAPER
                else:
                    env_action = ROCK
        return env_action


if __name__ == "__main__":
    env = RockPaperScissorsSequencePolicy2Env(other_sequence=True, double_with=SCISSORS)
    env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(
            np.random.choice([ROCK, PAPER, SCISSORS]))
        env.render()
        print(obs, reward, done, info)
