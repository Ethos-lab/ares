# Ares

Ares is a system-oriented wargame framework for adversarial ML used to simulate interactions between an attacker and defender in a real-time battle.

## Requirements

Ares uses the following dependencies:

- Python 3.6+

## Installation

First, clone this repository and cd into it.

```bash
git clone https://github.com/Ethos-lab/ares
cd ares
```

(Optional) Create a new Python virtual environment and activate it.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install all the package dependencies.

```bash
pip install -e .
```

## Structure

The framework is structured into three components:

- Attacker Agent
- Defender Agent
- Evaluation Scenario

Each component must be instantiated to create the reinforcement learning environment used to run Ares.

### Attacker Agent

The attacker agent (or simply attacker) is the adversary whose goal is to cause the defender to misclassify.

### Defender Agent

The defender agent (or simply defender) is the agent who needs to continuously classify the input example given by the attacker with the goal being to last as long as possible without misclassifying.

### Evaluation Scenario

The evaluation scenario (or simply scenario) defines the main parameters for the simulation.

## Config File

Ares uses a JSON configuration file to specify the parameters used to run the simulation. This JSON file has a field to instantiate each of the three components: attacker, defender, and scenario. For example configuration files, view the `configs/` directory.

## Usage

Based on our discussion in the [structure section](#structure), working with Ares is very simple. We utilize the config example as detailed in the [config file](#config-file) section. Extracting the config file for use in Ares uses this line of code

```python
from ares import utils

config = utils.get_config('/path/to/config.json')
```

One can start by initializing the attacker, defender, and scenario first.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create components
defender_agent = utils.get_defender_agent(config, device)
attacker_agent = utils.get_attacker_agent(config)
execution_scenario = utils.get_execution_scenario(config)
```

Next, we create the environment that houses these components

```python
# create environment
env = gym.make(
    'AresEnv-v0',
    attacker=attacker_agent,
    defender=defender_agent,
    scenario=execution_scenario
)
```

We can run the experiment for the specified number of trials (which can be set in the config file) and record the number of rounds until the attacker wins for each trial.

```python
episode_rewards = []
for episode in range(execution_scenario.num_episodes):
    print(f"=== Episode {episode + 1} ===")

    # initialize environment
    observation = env.reset()
    x = observation['x']
    y = observation['y']
    done = False

    # run simulation
    while not done:
        action = {
            'x': x,
            'y': y,
        }
        observation, reward, done, info = env.step(action)
        x = observation['x_adv']
        y_pred = observation['y_pred']
        winner = observation['winner']
        step_count = info["step_count"]

        print(f"Step {step_count:2}: ({y[0]} | {y_pred[0]})")

    print(f'Game end: {winner} wins after {episode} rounds')
    episode_rewards.append(reward)

# scenario stats
mean = np.mean(episode_rewards)
stddev = np.std(episode_rewards)
median = np.median(episode_rewards)
print(f'mean: {mean}, stddev: {stddev:.3f}, median: {median}')
```

For a more detailed example, view the demo Jupyter notebook in `notebooks/demo.ipynb`.

## Citation

If you use this code in your work please cite the following paper:

```bibtex
@inproceedings{ahmed2022ares,
      title={Ares: A System-Oriented Wargame Framework for Adversarial ML},
      author={Farhan Ahmed and Pratik Vaishnavi and Kevin Ekyholt and Amir Rahmati},
      booktitle={IEEE Security and Privacy Workshops (SPW)},
      year={2022},
}
```
