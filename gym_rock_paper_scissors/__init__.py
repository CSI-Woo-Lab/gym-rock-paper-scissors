from gym.envs.registration import register

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
