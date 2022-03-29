# Ares

Ares is a system-oriented wargame framework for adversarial ML used to simulate interactions between an attacker and defender in a real-time battle.

## Table of Contents

- [Ares](#ares)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Structure](#structure)
    - [Attacker Agent](#attacker-agent)
    - [Defender Agent](#defender-agent)
    - [Evaluation Scenario](#evaluation-scenario)
  - [Config File](#config-file)
  - [Usage](#usage)

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

Install the package dependencies into the virtual environment.

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

Ares uses a JSON configuration file to specify the parameters used to run the simulation. This JSON file has a field to instantiate each of the three components: attacker, defender, and scenario. An example configuration file is available in `ares/configs/default.json`.

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
counts = []
for trial in range(execution_scenario.num_trials):
    print(f'=== Trial {trial + 1} ===')

    # initialize environment
    observation = env.reset()
    image = observation['image']
    label = observation['label']
    done = False
    print(f'True label: {label[0]}')
    print('Preds: ', end='')

    # run simulation
    while not done:
        action = {
            'image': image,
            'label': label,
        }
        observation, reward, done, info = env.step(action)
        image = observation['image_adv']
        pred = observation['pred']
        winner = observation['winner']
        episode = info['step_count']

        print(f'{pred[0]}, ', end='')

    print(f'\nGame end: {winner} wins after {episode} rounds')
    counts.append(episode)
```
