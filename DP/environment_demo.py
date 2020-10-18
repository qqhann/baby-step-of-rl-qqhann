import random
from environment import Environment, State


class Agent:
    def __init__(self, env: Environment):
        self.actions = env.actions

    def policy(self, state: State):
        return random.choice(self.actions)


def main():
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 1],
    ]
    env = Environment(grid)
    agent = Agent(env)

    # try 10 games
    for i in range(10):
        # init agent position
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {i}: Agent gets {total_reward} rewards.")


if __name__ == "__main__":
    main()
