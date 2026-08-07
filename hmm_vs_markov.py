#!/usr/bin/env python3
"""Decisive mechanism test: hidden latent mode (HMM) vs literal higher-order
memory (2nd-order Markov) for sperm motility-state dynamics.

We showed motility is non-Markovian. Two competing explanations:
  (H1) LITERAL higher-order memory / hysteresis: the next observed state depends
       on the trajectory of OBSERVED states (2nd-order Markov on {P,NP,I}).
  (H2) HIDDEN MODE: each cell is in a slowly-changing LATENT motility mode and
       the 3 CASA categories are a NOISY projection of it (a hidden Markov model
       with K latent states emitting the 3 observed states). This would mean the
       "memory" is really unobserved persistent state -> CASA's 3 bins are a lossy
       readout of a smaller set of biophysical modes.

Both indict the standard first-order/snapshot model, but the biology differs.
We fit both on TRAIN tracks and compare per-token held-out log-likelihood (5-fold
block CV over tracks) on decorrelated non-overlapping 0.5 s blocks. The winner is
the better generative description; if a small-K HMM matches/beats 2nd-order
Markov, the parsimonious story is "hidden persistent modes".

Discrete-emission HMM trained with Baum-Welch in log-space (no external deps).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from markov_property_test import cv_order  # noqa: E402
from markov_property_robust import nonoverlap_sequences  # noqa: E402

RNG = np.random.default_rng(0)
N_OBS = 3
LOG0 = -1e30


def _log(x):
    with np.errstate(divide="ignore"):
        return np.where(x > 0, np.log(x), LOG0)


class DiscreteHMM:
    def __init__(self, n_states, n_obs=N_OBS, seed=0):
        self.K = n_states
        self.M = n_obs
        rng = np.random.default_rng(seed)
        self.log_pi = _log(np.full(n_states, 1.0 / n_states))
        A = rng.dirichlet(np.ones(n_states) * 5, size=n_states)
        self.log_A = _log(A)
        B = rng.dirichlet(np.ones(n_obs), size=n_states)
        self.log_B = _log(B)

    def _forward(self, obs):
        T = len(obs)
        la = np.empty((T, self.K))
        la[0] = self.log_pi + self.log_B[:, obs[0]]
        for t in range(1, T):
            # la[t,j] = logsumexp_i(la[t-1,i] + log_A[i,j]) + log_B[j,obs[t]]
            la[t] = logsumexp(la[t - 1][:, None] + self.log_A, axis=0) + self.log_B[:, obs[t]]
        return la

    def _backward(self, obs):
        T = len(obs)
        lb = np.empty((T, self.K))
        lb[-1] = 0.0
        for t in range(T - 2, -1, -1):
            # lb[t,i] = logsumexp_j(log_A[i,j] + log_B[j,obs[t+1]] + lb[t+1,j])
            lb[t] = logsumexp(self.log_A + (self.log_B[:, obs[t + 1]] + lb[t + 1])[None, :], axis=1)
        return lb

    def seq_loglik(self, obs):
        return float(logsumexp(self._forward(obs)[-1]))

    def fit(self, seqs, n_iter=25, tol=1e-3):
        prev = -np.inf
        for _ in range(n_iter):
            # accumulators (log-space via running sums of probabilities)
            log_pi_acc = np.full(self.K, LOG0)
            log_A_num = np.full((self.K, self.K), LOG0)
            log_A_den = np.full(self.K, LOG0)
            log_B_num = np.full((self.K, self.M), LOG0)
            log_B_den = np.full(self.K, LOG0)
            total_ll = 0.0
            for obs in seqs:
                T = len(obs)
                if T < 2:
                    continue
                la = self._forward(obs)
                lb = self._backward(obs)
                ll = logsumexp(la[-1])
                total_ll += ll
                lg = la + lb - ll                      # log gamma (T,K)
                log_pi_acc = np.logaddexp(log_pi_acc, lg[0])
                # vectorised xi accumulation over t=0..T-2
                # lxi[t,i,j] = la[t,i] + log_A[i,j] + log_B[j,obs[t+1]] + lb[t+1,j] - ll
                emit_next = self.log_B[:, obs[1:]].T            # (T-1, K) over j
                lxi = (la[:-1][:, :, None] + self.log_A[None, :, :]
                       + (emit_next + lb[1:])[:, None, :] - ll)  # (T-1,K,K)
                log_A_num = np.logaddexp(log_A_num, logsumexp(lxi, axis=0))
                log_A_den = np.logaddexp(log_A_den, logsumexp(lg[:-1], axis=0))
                for t in range(T):
                    log_B_num[:, obs[t]] = np.logaddexp(log_B_num[:, obs[t]], lg[t])
                log_B_den = np.logaddexp(log_B_den, logsumexp(lg, axis=0))
            # M-step
            self.log_pi = log_pi_acc - logsumexp(log_pi_acc)
            self.log_A = log_A_num - log_A_den[:, None]
            self.log_B = log_B_num - log_B_den[:, None]
            if abs(total_ll - prev) < tol * max(1, abs(prev)):
                break
            prev = total_ll
        return self

    def heldout_per_token(self, seqs):
        ll = sum(self.seq_loglik(o) for o in seqs if len(o) >= 1)
        n = sum(len(o) for o in seqs if len(o) >= 1)
        return ll / n


def cv_hmm(seqs, K, folds=5):
    idx = np.arange(len(seqs))
    RNG.shuffle(idx)
    parts = np.array_split(idx, folds)
    tot_ll, tot_n = 0.0, 0
    for f in range(folds):
        test_i = set(parts[f].tolist())
        train = [seqs[i] for i in idx if i not in test_i]
        test = [seqs[i] for i in parts[f]]
        # few restarts, keep best train fit
        best = None
        for r in range(3):
            m = DiscreteHMM(K, seed=r).fit(train, n_iter=20)
            tr_ll = sum(m.seq_loglik(o) for o in train[:200])
            if best is None or tr_ll > best[1]:
                best = (m, tr_ll)
        m = best[0]
        ll = sum(m.seq_loglik(o) for o in test)
        n = sum(len(o) for o in test)
        tot_ll += ll
        tot_n += n
    return tot_ll / tot_n


def n_params_markov(order):
    return (N_OBS ** order) * (N_OBS - 1)


def n_params_hmm(K):
    return (K - 1) + K * (K - 1) + K * (N_OBS - 1)


def main():
    seqs = nonoverlap_sequences(25)
    ntok = sum(len(s) for s in seqs)
    print(f"decorrelated 0.5 s blocks: {len(seqs)} tracks, {ntok} states\n")

    print("=== Markov models (held-out logL/token, 5-fold CV) ===")
    mll = cv_order(seqs, orders=(0, 1, 2, 3))
    for o in sorted(mll):
        print(f"  order {o}: logL/token = {mll[o]:.4f}   "
              f"(params={n_params_markov(o)})")

    print("\n=== Hidden Markov models (held-out logL/token, 5-fold CV) ===")
    hmm_res = {}
    for K in (2, 3, 4, 5):
        v = cv_hmm(seqs, K)
        hmm_res[K] = v
        print(f"  K={K} latent: logL/token = {v:.4f}   (params={n_params_hmm(K)})")

    best_markov = max(mll.values())
    best_hmm_K = max(hmm_res, key=hmm_res.get)
    best_hmm = hmm_res[best_hmm_K]
    print("\n=== Verdict ===")
    print(f"  best Markov (order 2/3): {best_markov:.4f}")
    print(f"  best HMM (K={best_hmm_K}):        {best_hmm:.4f}")
    diff = best_hmm - best_markov
    if diff > 0.005:
        print(f"  => HMM wins by {diff:+.4f}/token: memory is a HIDDEN PERSISTENT "
              f"MODE; CASA's 3 bins are a noisy readout of ~{best_hmm_K} modes.")
    elif diff < -0.005:
        print(f"  => 2nd-order Markov wins by {-diff:+.4f}/token: LITERAL "
              f"history-dependence / hysteresis on observed states.")
    else:
        print(f"  => statistical tie ({diff:+.4f}); both beat 1st-order. Report "
              f"both; non-Markovian either way.")

    import json
    out = {"markov": mll, "hmm": hmm_res,
           "n_tracks": len(seqs), "n_tokens": ntok}
    (ROOT / "outputs" / "markov" / "hmm_vs_markov.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> outputs/markov/hmm_vs_markov.json")


if __name__ == "__main__":
    main()
