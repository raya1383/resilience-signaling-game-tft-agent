#!/usr/bin/env python3
"""Fair two-role head-to-head evaluation against Tit-for-Tat."""
from __future__ import annotations

import argparse

import numpy as np

from see.agents.checkpoint import CheckpointAgent
from see.agents.scripted import TitForTatAgent
from see.config import IRAN, US
from see.env import StrategicEnduranceEnv
from see.tournament.runner import play_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="submission.pt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed0", type=int, default=20_260_901)
    args = parser.parse_args()

    env = StrategicEnduranceEnv()
    submission = CheckpointAgent(args.checkpoint, sample=True)
    opponent = TitForTatAgent()
    seeds = range(args.seed0, args.seed0 + args.episodes)

    as_iran = [
        play_episode(env, {IRAN: submission, US: opponent}, seed)
        for seed in seeds
    ]
    as_us = [
        play_episode(env, {IRAN: opponent, US: submission}, seed)
        for seed in seeds
    ]
    ours = np.asarray(
        [row["u"][IRAN] for row in as_iran]
        + [row["u"][US] for row in as_us]
    )
    theirs = np.asarray(
        [row["u"][US] for row in as_iran]
        + [row["u"][IRAN] for row in as_us]
    )
    margin = ours - theirs
    print(f"episodes={margin.size}")
    print(f"submission_score={ours.mean():.3f}")
    print(f"tit_for_tat_score={theirs.mean():.3f}")
    print(f"margin={margin.mean():+.3f}")
    print(f"win_rate={(margin > 0).mean():.3%}")


if __name__ == "__main__":
    main()
