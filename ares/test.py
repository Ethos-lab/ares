import numpy as np
import torch

from ares import create_env, utils


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    config_path = "./configs/detector.json"
    config = utils.get_config(config_path)

    # create environment
    env = create_env(config, device)

    episode_rewards = []
    for episode in range(env.scenario.num_episodes):
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

        print(f"Game end: {winner} wins after {reward} rounds")
        episode_rewards.append(reward)

    # scenario stats
    print(episode_rewards)
    mean = np.mean(episode_rewards)
    stddev = np.std(episode_rewards)
    median = np.median(episode_rewards)
    print(f"mean: {mean}, stddev: {stddev:.3f}, median: {median}")


if __name__ == "__main__":
    main()
