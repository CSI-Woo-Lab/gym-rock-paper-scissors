from gym.envs.registration import register
from gym_rock_paper_scissors.envs.rock_paper_scissors import ROCK, PAPER, SCISSORS, NULL_ACTION

register(
    id="RockPaperScissorsSequencePolicyEnv-v0",
    entry_point="gym_rock_paper_scissors.envs:RockPaperScissorsSequencePolicyEnv"
)

register(
    id="RockPaperScissorsRandomPolicyEnv-v0",
    entry_point="gym_rock_paper_scissors.envs:RockPaperScissorsRandomPolicyEnv"
)

register(
    id="RockPaperScissorsBiasedPolicyEnv-v0",
    entry_point="gym_rock_paper_scissors.envs:RockPaperScissorsBiasedPolicyEnv"
)

register(
    id="RockPaperScissorsSequencePolicy2Env-v0",
    entry_point="gym_rock_paper_scissors.envs:RockPaperScissorsSequencePolicy2Env"
)
