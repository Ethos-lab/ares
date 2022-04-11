import gym
import numpy as np
import torch

from ares import utils


def get_manual_config(attacks, defenses):
    attack_options = {
        # whitebox attack PGD
        'ProjectedGradientDescent': {
            "attack_type": "evasion",
            "attack_name": "ProjectedGradientDescent",
            "attack_params": {
                "norm": "inf",
                "eps": 0.03137254901,
                "eps_step": 0.00784313725,
                "num_random_init": 0,
                "max_iter": 1,
                "targeted": False,
                "batch_size": 1,
                "verbose": False
            }
        },
        # blackbox attack SquareAttack
        'SquareAttack': {
            "attack_type": "evasion",
            "attack_name": "SquareAttack",
            "attack_params": {
                "norm": "inf",
                "eps": 0.03137254901,
                "max_iter": 1,
                "batch_size": 1,
                "verbose": False
            }
        },
    }

    defenses_options = {
        # resnet 18 natural
        'resnet18_nat': {
            'model_file': './downloads/models/resnet.py',
            'model_name': 'resnet18_nat',
            'model_params': {
                'num_classes': 10
            },
            'model_state': './downloads/state_dicts/resnet18_nat.pth',
            'classifier_type': 'PyTorchClassifier',
            'classifier_params': {
                'loss': 'torch.nn.CrossEntropyLoss',
                'input_shape': [3, 32, 32],
                'nb_classes': 10,
                'clip_values': [0, 1]
            }
        },
        # resnet 18 adversarial
        'resnet18_adv': {
            'model_file': './downloads/models/resnet.py',
            'model_name': 'resnet18_adv',
            'model_params': {
                'num_classes': 10
            },
            'model_state': './downloads/state_dicts/resnet18_adv.pth',
            'classifier_type': 'PyTorchClassifier',
            'classifier_params': {
                'loss': 'torch.nn.CrossEntropyLoss',
                'input_shape': [3, 32, 32],
                'nb_classes': 10,
                'clip_values': [0, 1]
            }
        },
        # resnet 50 natural
        'resnet50_nat': {
            'model_file': './downloads/models/resnet.py',
            'model_name': 'resnet50_nat',
            'model_params': {
                'num_classes': 10
            },
            'model_state': './downloads/state_dicts/resnet50_nat.pth',
            'classifier_type': 'PyTorchClassifier',
            'classifier_params': {
                'loss': 'torch.nn.CrossEntropyLoss',
                'input_shape': [3, 32, 32],
                'nb_classes': 10,
                'clip_values': [0, 1]
            }
        },
        # resnet 50 adversarial
        'resnet50_adv': {
            'model_file': './downloads/models/resnet.py',
            'model_name': 'resnet50_adv',
            'model_params': {
                'num_classes': 10
            },
            'model_state': './downloads/state_dicts/resnet50_adv.pth',
            'classifier_type': 'PyTorchClassifier',
            'classifier_params': {
                'loss': 'torch.nn.CrossEntropyLoss',
                'input_shape': [3, 32, 32],
                'nb_classes': 10,
                'clip_values': [0, 1]
            }
        },
        # vgg11 natural
        'vgg11_nat': {
            'model_file': './downloads/models/vgg.py',
            'model_name': 'vgg11_nat',
            'model_params': {
                'num_classes': 10
            },
            'model_state': './downloads/state_dicts/vgg11_nat.pth',
            'classifier_type': 'PyTorchClassifier',
            'classifier_params': {
                'loss': 'torch.nn.CrossEntropyLoss',
                'input_shape': [3, 32, 32],
                'nb_classes': 10,
                'clip_values': [0, 1]
            }
        },
        # vgg11 adversarial
        'vgg11_adv': {
            'model_file': './downloads/models/vgg.py',
            'model_name': 'vgg11_adv',
            'model_params': {
                'num_classes': 10
            },
            'model_state': './downloads/state_dicts/vgg11_adv.pth',
            'classifier_type': 'PyTorchClassifier',
            'classifier_params': {
                'loss': 'torch.nn.CrossEntropyLoss',
                'input_shape': [3, 32, 32],
                'nb_classes': 10,
                'clip_values': [0, 1]
            }
        },
    }

    config = {
        'attacker': {
            "attacks": [attack_options[a] for a in attacks]
        },
        'defender': {
            'models': [defenses_options[d] for d in defenses]
        },
        'scenario': {
            'threat_model': 'white_box',
            'num_agents': 2,
            'reward': 0,
            'dataroot': '../downloads',
            'random_noise': True,
            'num_episodes': 50,
            'max_rounds': 50
        }
    }

    return config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    # config_path = "./configs/multiple_attacks.json"
    # config = utils.get_config(config_path)
    attacks = {
        'ProjectedGradientDescent',
        # 'SquareAttack',
    }
    defenses = {
        'resnet18_nat',
        # 'resnet18_adv',
        # 'resnet50_nat',
        # 'resnet50_adv',
        # 'vgg11_nat',
        # 'vgg11_adv',
    }
    config = get_manual_config(attacks, defenses)

    # create environment
    print('creating defender')
    defender_agent = utils.get_defender_agent(config, device)
    print('creating attacker')
    attacker_agent = utils.get_attacker_agent(config)
    print('creating execution scenario')
    execution_scenario = utils.get_execution_scenario(config)
    print('creating environment')
    env = gym.make("AresEnv-v0", attacker=attacker_agent, defender=defender_agent, scenario=execution_scenario)

    episode_rewards = []
    for episode in range(execution_scenario.num_episodes):
        print(f"=== Episode {episode + 1} ===")

        # initialize environment
        observation = env.reset()
        image = observation["image"]
        label = observation["label"]
        done = False

        # run simulation
        while not done:
            action = {
                "image": image,
                "label": label,
            }
            observation, reward, done, info = env.step(action)
            image = observation["image_adv"]
            pred = observation["pred"]
            winner = observation["winner"]
            step_count = info["step_count"]

            print(f"Step {step_count:2}: ({label[0]} | {pred[0]})")

        print(f"Game end: {winner} wins")
        episode_rewards.append(reward)

    # scenario stats
    print(episode_rewards)
    print(
        f"mean: {np.mean(episode_rewards)}, stddev: {np.std(episode_rewards):.3f}, median: {np.median(episode_rewards)}"
    )


if __name__ == "__main__":
    main()
