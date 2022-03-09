import gym
import numpy as np
import torch
import torch.nn as nn
# import torchvision

from ares import attacker, defender, utils


def main(args):
    # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    config_path = './ares/configs/default.json'
    config = utils.get_config(config_path)

    device = torch.device('cpu')

    defender_agent = utils.get_defender_agent(config, device)
    attacker_agent = utils.get_attacker_agent(config)

    print(defender_agent)
    print(attacker_agent)

    dataset_path = './downloads'
    dataset = utils.load_dataset(dataset_path)

    image, label = utils.get_valid_sample(dataset, defender_agent.classifiers)

    env = gym.make("AresEnv-v0", attacker=attacker_agent, defender=defender_agent, max_rounds=5)
    print(env.reset())

    for i in range(10):
        action = {
            'image': image,
            'label': label,
        }
        observation, reward, done, info = env.step(action)
        print(observation['image'].shape)
        print(observation['image_adv'].shape)
        print(observation['label'])
        print(observation['pred'])
        print(observation['winner'])
        print(done)

        image = observation['image_adv']


if __name__ == "__main__":
    args = None
    main(args)