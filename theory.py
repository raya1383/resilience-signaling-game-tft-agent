"""Bayesian probe-and-countercycle theory for the SEE tournament.

The public history is used to distinguish a one-step signal-mirroring
opponent from a noisy/non-structural opponent.  The policy starts with a
costless probe.  Two exact, legally observable mirror responses are required
before entering the role-specific countercycle; any mismatch switches to a
high-pressure response.  No hidden state or tournament-only information is
used.
"""
from __future__ import annotations

import numpy as np

from see.config import IRAN, US
from see.training.theory_api import TheorySpec


class BayesianCounterCycleTheory(TheorySpec):
    """Explicit mirror belief plus a robust role-asymmetric countercycle."""

    name = "bayesian-probe-countercycle-v2"
    extra_feature_dim = 14
    gamma = 0.995

    @staticmethod
    def _other(player_id):
        return US if player_id == IRAN else IRAN

    def _mirror_evidence(self, player_id, public_state):
        opponent = self._other(player_id)
        history = public_state.get("history", [])
        matches = 0
        mismatches = 0
        log_odds = 0.0

        # TFT necessarily opens at zero because the initial public posture is
        # zero.  A non-zero opening is therefore immediate counter-evidence.
        if history:
            opening = float(history[0].get("sigma", {}).get(opponent, 0.0))
            if abs(opening) < 1e-8:
                log_odds += np.log(0.98 / 0.20)
            else:
                mismatches += 1
                log_odds += np.log(0.005 / 0.20)

        previous_own_sigma = 0.0
        for index, row in enumerate(history):
            sigma = row.get("sigma", {})
            if index > 0 and opponent in row.get("active", []):
                action = int(row.get("action", {}).get(opponent, 6))
                if 0 <= action < 5:
                    observed = float(sigma.get(opponent, 0.0))
                    if abs(observed - previous_own_sigma) < 1e-8:
                        matches += 1
                        log_odds += np.log(0.98 / 0.20)
                    else:
                        mismatches += 1
                        log_odds += np.log(0.005 / 0.20)
            previous_own_sigma = float(sigma.get(player_id, 0.0))

        probability = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20.0, 20.0)))
        return matches, mismatches, float(probability)

    @staticmethod
    def _recommended_action(player_id, stage, matches, mismatches):
        # Probe calmly until the public history contains two true response
        # opportunities.  A mismatch is evidence for a non-mirroring rival,
        # against which sustained 0.75 pressure performs best.
        if mismatches:
            return 3
        if matches < 2:
            return 0

        # Against a mirror, low-cost asymmetric cycles exploit the response
        # lag without paying for permanent mutual escalation.  The cycles are
        # role-specific because the canonical game is asymmetric.
        cycle = (0, 1, 0) if player_id == IRAN else (1, 0)
        return int(cycle[int(stage) % len(cycle)])

    def extra_features(self, player_id, obs, public_state):
        matches, mismatches, p_mirror = self._mirror_evidence(
            player_id, public_state
        )
        stage = int(public_state.get("t", round(float(obs[0]) * 40.0)))
        recommendation = self._recommended_action(
            player_id, stage, matches, mismatches
        )

        # The first seven entries are a transparent action-recommendation
        # basis.  The remaining entries expose the belief and the strategic
        # state, so the fixed recurrent policy can condition on confidence,
        # endurance, commitment, and escalation risk.
        features = np.zeros(14, dtype=np.float32)
        features[recommendation] = 1.0
        features[7:] = np.asarray([
            p_mirror,
            min(matches / 4.0, 1.0),
            min(mismatches, 1),
            np.clip(obs[3], 0.0, 1.2),
            np.clip(1.0 - obs[4], -0.5, 1.0),
            np.clip(obs[7], 0.0, 1.0),
            np.clip(obs[6] * obs[7], 0.0, 1.0),
        ], dtype=np.float32)
        return features

    @staticmethod
    def _potential(obs):
        flexibility = float(np.clip(1.0 - obs[4], -0.5, 1.0))
        mutual_heat = float(obs[6]) * float(obs[7])
        return float(
            3.5 * np.clip(obs[3], 0.0, 1.2)
            + 1.5 * flexibility
            + 0.8 * float(obs[8])
            - 1.1 * mutual_heat
            - 0.6 * np.clip(obs[15], 0.0, 2.0)
        )

    def shaping(self, player_id, obs, action, env_reward, next_obs,
                next_public, terminated):
        current = self._potential(obs)
        following = 0.0 if terminated else self._potential(next_obs)
        return self.gamma * following - current
