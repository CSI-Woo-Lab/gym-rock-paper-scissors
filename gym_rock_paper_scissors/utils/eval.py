from gym_rock_paper_scissors.envs.rock_paper_scissors import RockPaperScissorsBaseEnv, RockPaperScissorsRandomPolicyEnv, RockPaperScissorsSequencePolicyEnv, RockPaperScissorsRandomEnv, RockPaperScissorsBiasedPolicyEnv


# TODO: test
def eval_rock_paper_scissors_agent(agent, env: RockPaperScissorsBaseEnv, n_eval_episode=1000):
    if type(env).__name__ == "RockPaperScissorsSequenceEnv":
        optimal_winning_rate = RockPaperScissorsSequencePolicyEnv.optimal_winning_rate
    elif type(env).__name__ == "RockPaperScissorsRandomPolicyEnv":
        optimal_winning_rate = RockPaperScissorsRandomPolicyEnv.optimal_winning_rate
    elif type(env).__name__ == "RockPaperScissorsBiasedEnv":
        optimal_winning_rate = RockPaperScissorsBiasedPolicyEnv.optimal_winning_rate
    else:
        raise TypeError("env type is not matched")

    n_win = 0
    for _ in range(n_eval_episode):
        done = False
        obs = env.reset()
        while done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            if reward > 0:
                n_win += 1
    winning_rate = n_win / n_eval_episode
    normalized_score = winning_rate / optimal_winning_rate
    return normalized_score