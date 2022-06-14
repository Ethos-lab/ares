import gym
import numpy as np
import torch

from ares import utils


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    config_path = "./configs/detector.json"
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
        x = observation["x"]
        y = observation["y"]
        done = False

        # run simulation
        while not done:
            action = {
                "x": x,
                "y": y,
            }
            observation, reward, done, info = env.step(action)
            x = observation["x_adv"]
            y_pred = observation["y_pred"]
            winner = observation["winner"]
            step_count = info["step_count"]

            print(f"Step {step_count:2}: ({y[0]} | {y_pred[0]})")

        print(f"Game end: {winner} wins after {episode} rounds")
        episode_rewards.append(reward)

    # scenario stats
    print(episode_rewards)
    mean = np.mean(episode_rewards)
    stddev = np.std(episode_rewards)
    median = np.median(episode_rewards)
    print(f"mean: {mean}, stddev: {stddev:.3f}, median: {median}")


if __name__ == "__main__":
    main()
