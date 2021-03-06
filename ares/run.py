import argparse

import numpy as np

from ares.utils import construct, load_config


def get_args():
    parser = argparse.ArgumentParser(description="Ares default simulation run.")
    parser.add_argument("config", type=str, help="JSON config file path")
    args = parser.parse_args()
    return args


def run_simulation(args):
    # load config file
    config = load_config(args.config)

    # create environment
    env = construct(config)

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
            eps = observation["eps"]
            evaded = observation["evaded"]
            detected = observation["detected"]
            winner = observation["winner"]
            step_count = info["step_count"]

            if evaded:
                print(f"Step {step_count:2}: ({y[0]} | {y_pred[0]}), eps = {eps:.6f}, attacker evaded")
            elif detected:
                print(f"Step {step_count:2}: ({y[0]} | {y_pred[0]}), eps = {eps:.6f}, attacker was detected")
            else:
                print(f"Step {step_count:2}: ({y[0]} | {y_pred[0]}), eps = {eps:.6f}")

        print(f"Game end: {winner} wins after {reward} rounds")
        episode_rewards.append(reward)

    # scenario stats
    print(episode_rewards)
    mean = np.mean(episode_rewards)
    stddev = np.std(episode_rewards)
    median = np.median(episode_rewards)
    print(f"mean: {mean}, stddev: {stddev:.3f}, median: {median}")


def main():
    args = get_args()
    run_simulation(args)
