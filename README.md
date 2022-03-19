# Ares
## Table of Contents
1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [Structure](#Structure)
4. [Configs](#Configs)
5. [Execution](#Execution)

### Requirements
Ares uses the following dependencies

- PyTorch
- OpenAI Gym
- IBM's Adversarial Robustness Toolbox
- Numpy
- Scipy
- Pillow

### Installation
First, clone this repository and cd into it. Then perform the following (we assume you have pip and virtualenv installed)

1. `virtualenv venv`
2. `source venv/bin/activate`
3. `pip install -e .`

### Structure

### Configs

### Execution
Based on our discussion in the [structure section](#Structure), working with Ares is very simple. We utilize the config example as detailed in the [configs](#Configs) section

One can start by initializing the attacker, defender, and scenario first.

```python
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