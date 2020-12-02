import random
import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


class CoinToss:
    def __init__(self, head_probs: List, max_episode_steps: int = 30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action: int) -> Tuple[float, bool]:
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. Please reset env.")
        done = bool(self.toss_count == final)

        if action >= len(self.head_probs):
            raise Exception(f"The No.{action} coin doesn't exist.")
        head_prob = self.head_probs[action]
        if random.random() < head_prob:
            reward = 1.0
        else:
            reward = 0.0
        self.toss_count += 1
        return reward, done


class EpsilonGreedyAgent:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.V: List[float] = []

    def policy(self):
        if random.random() < self.epsilon:
            coins = range(len(self.V))
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def play(self, env: CoinToss):
        # Init estimation
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards


def main():
    env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
    epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
    game_steps = list(range(10, 310, 10))
    result = {}
    for e in epsilons:
        agent = EpsilonGreedyAgent(epsilon=e)
        means = []
        for s in game_steps:
            env.max_episode_steps = s
            rewards = agent.play(env)
            means.append(np.mean(rewards))
        result[f"epsilon={e}"] = means
    result["coin toss count"] = game_steps
    res = pd.DataFrame(result)
    res.set_index("coin toss count", drop=True, inplace=True)
    res.plot.line(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":
    main()
