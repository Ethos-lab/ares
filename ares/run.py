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
    episode_queries = []
    attacker_rewards = []
    attacker_queries = []
    defender_rewards = []
    defender_queries = []

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
            defense = observation["defense"]
            attack = observation["attack"]
            x = observation["x_adv"]
            y_pred = observation["y_pred"]
            eps = observation["eps"]
            evaded = observation["evaded"]
            detected = observation["detected"]
            winner = observation["winner"]
            step_count = info["step_count"]
            queries = info["queries"]

            print(f"Round {step_count:2}: defense = {defense}, attack = {attack}")
            print(f"\t [label = {y[0]} | pred = {y_pred[0]}], eps = {eps:.6f}, queries = {queries}")
            if evaded:
                print("\t attacker evaded")
            elif detected:
                print("\t attacker was detected")

        print(f"Game end: {winner} wins after {reward} rounds and {queries} queries")
        episode_rewards.append(reward)
        episode_queries.append(queries)
        if winner == "attacker":
            attacker_rewards.append(reward)
            attacker_queries.append(queries)
        elif winner == "defender":
            defender_rewards.append(reward)
            defender_queries.append(queries)

    # scenario stats
    print_statistics("Simulation", episode_rewards, episode_queries)
    print_statistics("Attacker", attacker_rewards, attacker_queries)
    print_statistics("Defender", defender_rewards, defender_queries)


def print_statistics(title, episode_rewards, episode_queries):
    print(f"\n=== {title} Statistics ===")
    total = len(episode_rewards)
    print(f"Total: {total}")
    if total > 0:
        reward_mean = np.mean(episode_rewards)
        reward_stddev = np.std(episode_rewards)
        reward_median = np.median(episode_rewards)
        print(f"Rounds:  mean = {reward_mean:.3f}, stddev = {reward_stddev:.3f}, median = {reward_median}")
        queries_mean = np.mean(episode_queries)
        queries_stddev = np.std(episode_queries)
        queries_median = np.median(episode_queries)
        print(f"Queries: mean = {queries_mean:.3f}, stddev = {queries_stddev:.3f}, median = {queries_median}")


def main():
    args = get_args()
    run_simulation(args)
