import argparse
import random

import gym
import numpy as np
import torch
import torchvision

from ares import attacker, defender, utils

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def get_args():
    parser = argparse.ArgumentParser(description="Entry point for MTD Eval")
    parser.add_argument("--dataroot", type=str, default="./downloads", help="CIFAR10 data location")
    parser.add_argument("--episodes", type=int, default=50, help="total training episodes")
    parser.add_argument("--n_trials", type=int, default=50, help="number of trials")
    parser.add_argument("--gpu", type=str, default="0", help="the gpu id used for predict")
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    models = [
        utils.load_resnet18(device),
        utils.load_resnet18(device, True),
        utils.load_resnet50(device),
        utils.load_resnet50(device, True),
        utils.load_vgg11(device),
        utils.load_vgg11(device, True),
    ]
    attacker_agent = attacker.AttackerAgent(device)
    defender_agent = defender.DefenderAgent(models, device)
    env = gym.make("AresEnv-v0", attacker=attacker_agent, defender=defender_agent)

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, transform=transforms, download=True)

    counts = []
    for trial in range(args.n_trials):
        print(f"=== Trial {trial + 1} ===")
        env.reset()

        image, label = utils.get_valid_sample(dataset, models, device)
        image_original = image.clone()

        with torch.no_grad():
            random_noise = torch.empty(image.size(), device=device).uniform_(-8 / 255, 8 / 255)
            image = image + random_noise

        for episode in range(args.episodes):
            action = {
                "image_original": image_original,
                "image": image,
                "label": label,
                "branch": False,
            }
            observation, _, _, _ = env.step(action)
            image = observation["image"]
            label = observation["label"]
            preds = observation["preds"]

            if any(label[0] == p for p in preds):
                print(f"Episode {episode + 1:2}: defender wins ({label[0]} | {preds})")
            else:
                print(f"Episode {episode + 1:2}: attacker wins ({label[0]} | {preds})")
                counts.append(episode + 1)
                break

        print("game end")

    print(counts)
    print(f"mean: {np.mean(counts)}, stddev: {np.std(counts):.3f}, median: {np.median(counts)}")
    print(f"{np.mean(counts)}\t{np.std(counts):.3f}\t{np.median(counts)}")


if __name__ == "__main__":
    args = get_args()
    main(args)
