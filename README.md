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

Install all the package dependencies with your choice of frameworks. For example, to use PyTorch and ART use the following.

```bash
pip install -e .[pytorch, art]
```

Ares supports the following frameworks:

- PyTorch
- IBM Adversarial Robustness Toolkit

## Config File

Ares uses a JSON configuration file to specify the parameters used to run the simulation. This JSON file has a field to instantiate each of the three components: attacker, defender, and scenario.

### Attacker Agent

The attacker agent (or simply attacker) is the adversary whose goal is to cause the defender to misclassify.

### Defender Agent

The defender agent (or simply defender) is the agent who needs to continuously classify the input example given by the attacker with the goal being to last as long as possible without misclassifying.

### Evaluation Scenario

The evaluation scenario (or simply scenario) defines the main parameters for the simulation.

Each component must be instantiated to create the reinforcement learning environment used to run Ares. For example configuration files, view the `configs/` directory.

## Usage

### Basic

To run a basic simulation, simply use the command line interface.

```bash
ares /path/to/config.json
```

### Intermediate

For more customizable simulations, the environment component needs to be constructed and run manually. For simplicity, two helper methods are provided to load the config file and construct the environment.

```python
from ares import load_config, construct

config = load_config('/path/to/config.json')
env = construct(config)
```

We can run the experiment for the specified number of trials (which can be set in the config file) and record the number of rounds until the attacker wins for each trial.

```python
episode_rewards = []
for episode in range(env.scenario.num_episodes):
    print(f'=== Episode {episode + 1} ===')

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
        step_count = info['step_count']

        print(f'Step {step_count:2}: ({y[0]} | {y_pred[0]})')

    print(f'Game end: {winner} wins after {reward} rounds')
    episode_rewards.append(reward)

# scenario stats
mean = np.mean(episode_rewards)
stddev = np.std(episode_rewards)
median = np.median(episode_rewards)
print(f'mean: {mean}, stddev: {stddev:.3f}, median: {median}')
```

### Advanced

Each of the components can also be constructed manually rather than using the helper functions. This does not require a config file and gives more freedom into their instantiation.

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
