import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import gym_rock_paper_scissors
from gym_rock_paper_scissors.utils.eval import eval_rock_paper_scissors_agent
import os

# install custom env using pip first

sequence_env = gym.make("RockPaperScissorsSequencePolicyEnv-v0", other_sequence=True)
random_env = gym.make("RockPaperScissorsRandomPolicyEnv-v0")
biased_env = gym.make("RockPaperScissorsBiasedPolicyEnv-v0")

agent = DQN(MlpPolicy, sequence_env, verbose=1)
agent.learn(total_timesteps=80000, log_interval=4)
agent.save("dqn_rps")

del agent

agent = DQN.load("dqn_rps")
score = eval_rock_paper_scissors_agent(agent, sequence_env)
print(score)

os.remove("dqn_rps.zip")