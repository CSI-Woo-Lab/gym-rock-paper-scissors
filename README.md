# gym-rock-paper-scissors

OpenAI Gym-style RL environment of Rock Paper Scissors game.

## Usage

To set up the custom module, do

```sh
git clone https://github.com/CSI-Woo-Lab/gym-rock-paper-scissors.git
pip install -e gym-rock-paper-scissors/
```

(using `virtualenv` or `miniconda` is recommended)

Then make `Env` objects.

```python
import gym
import gym_rock_paper_scissors
from gym_rock_paper_scissors import ROCK, PAPER, SCISSORS

sequence_env = gym.make("RockPaperScissorsSequencePolicyEnv-v0", start_with=SCISSORS, other_sequence=True)

sequence_env_2 = gym.make("RockPaperScissorsSequencePolicy2Env-v0", start_with=PAPER, other_sequence=False, double_with=ROCK)

biased_env = gym.make("RockPaperScissorsBiasedPolicyEnv-v0", biased_by=PAPER)

random_env = gym.make("RockPaperScissorsRandomPolicyEnv-v0")
```
