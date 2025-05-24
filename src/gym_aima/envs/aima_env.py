# Copyright (c) 2018, Miguel Morales
# Copied from https://github.com/mimoralea/gym-aima/blob/master/gym_aima/envs/__init__.py
# License: MIT
# Updated to use gymnasium
"""Gymnasium environments for AIMA."""

import sys
from contextlib import closing

import numpy as np
from gymnasium import Env, spaces, utils
from six import StringIO

LEFT, DOWN, RIGHT, UP = range(4)


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution.

    Each row specifies class probabilities.
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()


class AIMAEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(
        self, map_name="3x4", noise=0.2, living_rew=0.0, sink=False, render_mode="human"
    ):
        """Initialize the environment."""
        super().__init__()
        self.map_name = map_name
        self.render_mode = render_mode

        desc = [
            "FFFG",
            "FWFH",
            "SFFF",
        ]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        self.s = None  # Current state
        self.lastaction = None  # Last action taken

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if sink:
                        if letter in b"W":
                            li.append((1.0, s, 0, True))
                        elif letter in b"G":
                            li.append((1.0, 5, 1, True))
                        elif letter in b"H":
                            li.append((1.0, 5, -1, True))
                        else:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                newstate = s if newletter in b"W" else newstate
                                rew = living_rew
                                li.append(
                                    (
                                        np.round(
                                            1.0 - noise if a == b else noise / 2.0, 2
                                        ),
                                        newstate,
                                        rew,
                                        False,
                                    )
                                )
                    elif letter in b"GHW":
                        li.append((1.0, s, 0, True))
                    else:
                        for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            newstate = s if newletter in b"W" else newstate
                            done = bytes(newletter) in b"GH"
                            rew = living_rew
                            rew += (
                                1.0
                                if newletter == b"G"
                                else -1.0
                                if newletter == b"H"
                                else 0
                            )
                            li.append(
                                (
                                    np.round(1.0 - noise if a == b else noise / 2.0, 2),
                                    newstate,
                                    rew,
                                    done,
                                )
                            )

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

    def render(self):
        mode = self.render_mode

        if mode == "None" and self.render_mode not in self.metadata["render_modes"]:
            msg = f"Unsupported render mode: {self.render_mode}"
            raise ValueError(msg)

        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
        return None

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, False, {"prob": p})

    def reset(self, seed: int | None = None, options: dict | None = None):  # noqa: ARG002
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        return int(self.s), {}

    def close(self):
        super().close()
