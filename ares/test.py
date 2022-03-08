import json

# import gym
import numpy as np
import torch
import torch.nn as nn
# import torchvision

from ares import attacker, defender, utils


def main(args):
    # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # attacker_agent = attacker.AttackerAgent(device)
    # defender_agent = defender.DefenderAgent(models, device)
    # attacker_agent = None
    # defender_agent = None
    # env = gym.make("AresEnv-v0", attacker=attacker_agent, defender=defender_agent)
    # print(env.reset())
    # print(env.step({}))
    # print(env.step({}))
    # print(env.step({}))
    with open('./ares/configs/default.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cpu')

    model_file = config['defender'][0]['model_file']
    model_name = config['defender'][0]['model_name']
    model_params = config['defender'][0]['model_params']
    model_state = config['defender'][0]['model_state']
    classifier_type = config['defender'][0]['classifier_type']
    classifier_params = config['defender'][0]['classifier_params']
    classifier_params['loss'] = nn.CrossEntropyLoss()
    model = utils.load_torch_model(model_file, model_name, model_params, model_state, device)
    classifier = utils.get_classifier(model, classifier_type, classifier_params, device)
    
    # defender_agent = defender.DefenderAgent([classifier], [1], device)
    # print(defender_agent.change_model())
    # print(defender_agent.get_model())

    attack_type = config['attacker'][0]['attack_type']
    attack_name = config['attacker'][0]['attack_name']
    attack_params = config['attacker'][0]['attack_params']

    attack = utils.get_evasion_attack(attack_name, classifier, attack_params)
    print(attack)

    bx = attack.generate(np.random.rand(1, 3, 32, 32).astype(np.float32), np.array([2]))
    print(bx.shape)


if __name__ == "__main__":
    args = None
    main(args)