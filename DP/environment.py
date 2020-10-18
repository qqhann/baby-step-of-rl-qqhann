from enum import Enum
from typing import Dict

import numpy as np


class State:
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment:
    def __init__(self, grid, move_prob=0.8):
        """
        grid: 2d array. its values are treated as attributes
        attribute kinds:
        0: ordinary cell
        -1: damage cell (game end)
        1: reward cell (game end)
        9: block cell (can't locate agent)
        """
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        # a chance to move to the selected direction.
        # which means will move to other directions in (1 - move_prob)
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for r in range(self.row_length):
            for c in range(self.column_length):
                # Block cells not included to the state
                if self.grid[r][c] != 9:
                    states.append(State(r, c))
        return states

    def transit_func(self, state, action: Action) -> Dict[State, float]:
        transition_probs: Dict[State, float] = {}
        if not self.can_action_at(state):
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state: State):
        # if you are not at ordinary cell 0, you cannot action.
        # bc game is over in -1, 1 cells
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state: State, action: Action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # out of range
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state
        # block cell
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state: State):
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done

    def reset(self):
        # locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action: Action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done

    def transit(self, state: State, action: Action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        # next_states = []
        # probs = []
        # for s, p in transition_probs.items():
        #     next_states.append(s)
        #     probs.append(p)

        # next_state = np.random.choice(next_states, p=probs)
        next_state = np.random.choice(
            list(transition_probs.keys()), p=list(transition_probs.values())
        )
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
