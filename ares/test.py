import gym
import numpy as np
import torch

from ares import utils


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    config_path = "./configs/multiple_attacks.json"
    config = utils.get_config(config_path)

    # create environment
    defender_agent = utils.get_defender_agent(config, device)
    attacker_agent = utils.get_attacker_agent(config)
    execution_scenario = utils.get_execution_scenario(config)
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
