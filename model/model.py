"""
Wh-question polarity model: clean O(N²) implementation.

This module implements a decision-theoretic RSA model predicting which question
a rational questioner will ask (e.g., "Which vials are contaminated?" vs
"Which are uncontaminated?") based on their goal and decision problem.

DESIGN PRINCIPLES:
1. Use ABSOLUTE categories (cont/uncont/unknown) throughout
2. Never use relative categories (match/other) that change meaning based on context
3. Let KL naturally penalize unlikely compositions (no hard validity constraints)

EXACTNESS NOTE:
This count-based model uses O(N²) complexity by representing responses as counts
rather than specific vial sets. This is mathematically EXACT (not an approximation)
because of exchangeability: all vials within a knowledge category are symmetric, so
compositions with the same count have identical KL, weight, and utility contributions.
The multinomial coefficient properly counts how many bitmask responses map to each
composition. Verified to match the O(2^N) bitmask implementation exactly.
"""

import numpy as np
from scipy.special import comb
from dataclasses import dataclass
from typing import Tuple, Dict

# Type aliases for clarity
QuestionType = int  # 0 = WHICH_CONT, 1 = WHICH_UNCONT
GoalType = int      # 0 = FIND_UNCONT, 1 = FIND_CONT


@dataclass(frozen=True)
class Knowledge:
    """Speaker's knowledge about vials (absolute categories)."""
    n_cont: int      # Number of vials known to be contaminated
    n_uncont: int    # Number of vials known to be uncontaminated
    n_total: int     # Total number of vials

    @property
    def n_unknown(self) -> int:
        return self.n_total - self.n_cont - self.n_uncont


@dataclass(frozen=True)
class Composition:
    """Response composition in absolute terms."""
    m_cont: int      # Vials mentioned from known-contaminated
    m_uncont: int    # Vials mentioned from known-uncontaminated
    m_unknown: int   # Vials mentioned from unknown

    @property
    def total(self) -> int:
        return self.m_cont + self.m_uncont + self.m_unknown


@dataclass
class Model:
    """
    Wh-question polarity model with clean parameterization.

    Hyperparameters:
        alpha_r: Speaker rationality
        alpha_q: Questioner rationality
        alpha_policy: Decision-maker rationality
        gamma: Speaker confidence P(correct | known)
        length_cost: Response brevity penalty
    """
    alpha_r: float = 5.0
    alpha_q: float = 5.0
    alpha_policy: float = 10.0
    gamma: float = 0.9
    length_cost: float = 0.1

    # === PUBLIC API ===

    def predict(self, goal: str, decision_type: str, p_uncont: float, n_vials: int) -> float:
        """Predict P(WHICH_UNCONT) for a given scenario."""
        g = 0 if goal == 'find_uncont' else 1
        result = self._run(n_vials, 1 - p_uncont, decision_type)
        key = 'find_uncontam' if g == 0 else 'find_contam'
        return result[key]['p_which_uncont']

    def predict_all(self, decision_type: str, p_uncont: float, n_vials: int) -> dict:
        """Predict for both goals at once."""
        result = self._run(n_vials, 1 - p_uncont, decision_type)
        return {
            'find_uncont': result['find_uncontam']['p_which_uncont'],
            'find_cont': result['find_contam']['p_which_uncont']
        }

    def _run(self, n_vials: int, contamination_rate: float, decision_type: str) -> dict:
        """Run the full model and return results."""
        n = n_vials
        p = 1 - contamination_rate  # P(uncontaminated)
        binom = self._binom_table(n)

        # All knowledge configurations
        configs = [Knowledge(n_c, n_u, n)
                   for n_c in range(n + 1)
                   for n_u in range(n + 1 - n_c)]

        # Compute R0 distribution for all (question, knowledge) pairs
        r0 = self._compute_r0_all(n, p, binom, configs)

        # Compute DPValue for all (goal, question, response_count) triples
        if decision_type == 'singleton':
            dpvalue = self._compute_dpvalue_singleton(n, p, binom)
        else:
            dpvalue = self._compute_dpvalue_set_id(n, p, binom)

        # Compute Q1 for each goal
        q1 = self._compute_q1(r0, dpvalue, configs, n)

        return {
            'n_vials': n,
            'contamination_rate': contamination_rate,
            'decision_type': decision_type,
            'find_uncontam': {
                'p_which_uncont': q1[0],
                'p_which_cont': 1 - q1[0],
            },
            'find_contam': {
                'p_which_uncont': q1[1],
                'p_which_cont': 1 - q1[1],
            }
        }

    # === CORE COMPUTATIONS ===

    def _compute_kl(self, q: QuestionType, know: Knowledge, comp: Composition, p: float) -> float:
        """
        Compute KL(literal_listener || speaker_belief) for a response composition.

        This is the heart of the speaker model. Key principles:
        1. Composition uses ABSOLUTE categories (cont/uncont/unknown)
        2. Speaker beliefs depend ONLY on knowledge, NOT on question type
        3. What the listener infers depends on question type
        4. NO hard validity constraints - KL naturally penalizes "wrong" mentions
        """
        gamma = self.gamma

        # === EMPTY RESPONSE ===
        if comp.total == 0:
            return self._kl_empty_response(q, know, p)

        # === MENTIONED VIALS ===
        # Literal listener conditions on: mentioned vials have queried property
        # KL contribution = -log P_speaker(vial has queried property)
        #
        # Speaker beliefs (independent of q):
        #   known-cont: P(uncont) = 1-γ, P(cont) = γ
        #   known-uncont: P(uncont) = γ, P(cont) = 1-γ
        #   unknown: P(uncont) = p, P(cont) = 1-p

        if q == 1:  # WHICH_UNCONT: listener infers mentioned vials are uncont
            # P_speaker(uncont) for each category:
            kl_mentioned = (
                comp.m_uncont * (-np.log(max(gamma, 1e-15))) +      # known-uncont: γ
                comp.m_cont * (-np.log(max(1 - gamma, 1e-15))) +    # known-cont: 1-γ
                comp.m_unknown * (-np.log(max(p, 1e-15)))           # unknown: p
            )
        else:  # WHICH_CONT: listener infers mentioned vials are cont
            # P_speaker(cont) for each category:
            kl_mentioned = (
                comp.m_cont * (-np.log(max(gamma, 1e-15))) +        # known-cont: γ
                comp.m_uncont * (-np.log(max(1 - gamma, 1e-15))) +  # known-uncont: 1-γ
                comp.m_unknown * (-np.log(max(1 - p, 1e-15)))       # unknown: 1-p
            )

        # === NON-MENTIONED VIALS ===
        # Listener has prior Bernoulli(p) for P(uncont)
        # Speaker belief depends on knowledge category (INDEPENDENT of q!)

        n_nonmention_uncont = know.n_uncont - comp.m_uncont
        n_nonmention_cont = know.n_cont - comp.m_cont
        # n_nonmention_unknown contributes 0 (KL of p||p)

        kl_nonmention = (
            n_nonmention_uncont * self._kl_bernoulli(p, gamma) +
            n_nonmention_cont * self._kl_bernoulli(p, 1 - gamma)
        )

        return kl_mentioned + kl_nonmention

    def _kl_empty_response(self, q: QuestionType, know: Knowledge, p: float) -> float:
        """KL for empty response ("none exist")."""
        gamma = self.gamma

        # Empty response means: no vials have queried property
        # Listener conditions on: ALL vials lack queried property

        if q == 1:  # WHICH_UNCONT with r=0 → all vials are cont
            # Listener: all have w_i = 0 (cont)
            # Speaker beliefs for P(w_i = 0):
            #   known-cont: P(cont) = γ
            #   known-uncont: P(cont) = 1-γ
            #   unknown: P(cont) = 1-p
            return (
                -know.n_cont * np.log(max(gamma, 1e-15)) +
                -know.n_uncont * np.log(max(1 - gamma, 1e-15)) +
                -know.n_unknown * np.log(max(1 - p, 1e-15))
            )
        else:  # WHICH_CONT with r=0 → all vials are uncont
            # Listener: all have w_i = 1 (uncont)
            # Speaker beliefs for P(w_i = 1):
            #   known-cont: P(uncont) = 1-γ
            #   known-uncont: P(uncont) = γ
            #   unknown: P(uncont) = p
            return (
                -know.n_cont * np.log(max(1 - gamma, 1e-15)) +
                -know.n_uncont * np.log(max(gamma, 1e-15)) +
                -know.n_unknown * np.log(max(p, 1e-15))
            )

    def _kl_bernoulli(self, p_listener: float, p_speaker: float) -> float:
        """KL divergence between Bernoulli distributions."""
        if p_speaker <= 0 or p_speaker >= 1:
            return float('inf')
        kl = 0.0
        if p_listener > 0:
            kl += p_listener * np.log(p_listener / p_speaker)
        if p_listener < 1:
            kl += (1 - p_listener) * np.log((1 - p_listener) / (1 - p_speaker))
        return kl

    def _compute_r0_all(self, n: int, p: float, binom: np.ndarray,
                        configs: list) -> np.ndarray:
        """Compute R0 distribution for all (question, knowledge, response_count)."""
        r0 = np.zeros((2, len(configs), n + 1))

        for q in range(2):
            for k_idx, know in enumerate(configs):
                weights = self._compute_response_weights(q, know, p, binom)
                total = weights.sum()
                if total > 0:
                    r0[q, k_idx, :] = weights / total

        return r0

    def _compute_response_weights(self, q: QuestionType, know: Knowledge,
                                   p: float, binom: np.ndarray) -> np.ndarray:
        """Compute unnormalized R0 weights for each response count m."""
        n = know.n_total
        weights = np.zeros(n + 1)

        # Iterate over all compositions (KL naturally penalizes unlikely ones)
        for comp in self._all_compositions(know):
            m = comp.total

            # Multinomial coefficient for this composition
            mult = (binom[know.n_cont, comp.m_cont] *
                    binom[know.n_uncont, comp.m_uncont] *
                    binom[know.n_unknown, comp.m_unknown])

            kl = self._compute_kl(q, know, comp, p)

            if np.isfinite(kl):
                w = mult * np.exp(self.alpha_r * (-kl - self.length_cost * m))
                weights[m] += w

        return weights

    def _all_compositions(self, know: Knowledge):
        """Generate all possible compositions (KL naturally penalizes unlikely ones)."""
        for m_c in range(know.n_cont + 1):
            for m_u in range(know.n_uncont + 1):
                for m_unk in range(know.n_unknown + 1):
                    yield Composition(m_cont=m_c, m_uncont=m_u, m_unknown=m_unk)

    def _compute_q1(self, r0: np.ndarray, dpvalue: np.ndarray,
                    configs: list, n: int) -> Dict[int, float]:
        """Compute Q1 (questioner model) for each goal."""
        q1 = {}
        for g in range(2):
            ev_q = np.zeros(2)
            for q in range(2):
                total_ev = 0.0
                for k_idx in range(len(configs)):
                    ev_config = sum(r0[q, k_idx, m] * dpvalue[g, q, m]
                                    for m in range(n + 1))
                    total_ev += ev_config / len(configs)
                ev_q[q] = total_ev

            # Softmax over questions
            exp_vals = np.exp(self.alpha_q * (ev_q - ev_q.max()))
            q1[g] = (exp_vals / exp_vals.sum())[1]

        return q1

    # === DECISION-MAKER MODEL ===

    def _compute_dpvalue_singleton(self, n: int, p: float, binom: np.ndarray) -> np.ndarray:
        """DPValue for singleton decision (pick one vial)."""
        dpvalue = np.zeros((2, 2, n + 1))

        for g in range(2):
            for q in range(2):
                goal_matches_question = (g == 0 and q == 1) or (g == 1 and q == 0)

                for m in range(n + 1):
                    if m == 0:
                        # Empty response: all vials lack queried property
                        if q == 1:  # All cont
                            dpvalue[g, q, 0] = 0.0 if g == 0 else 1.0
                        else:  # All uncont
                            dpvalue[g, q, 0] = 1.0 if g == 0 else 0.0
                    elif m == n:
                        # All mentioned: pick from mentioned
                        dpvalue[g, q, m] = 1.0 if goal_matches_question else 0.0
                    else:
                        # Mixed: softmax over mentioned vs remaining
                        u_mentioned = 1.0 if goal_matches_question else 0.0
                        u_remaining = p if g == 0 else (1 - p)

                        w_mentioned = m * np.exp(self.alpha_policy * u_mentioned)
                        w_remaining = (n - m) * np.exp(self.alpha_policy * u_remaining)

                        p_mentioned = w_mentioned / (w_mentioned + w_remaining)
                        dpvalue[g, q, m] = p_mentioned * u_mentioned + (1 - p_mentioned) * u_remaining

        return dpvalue

    def _compute_dpvalue_set_id(self, n: int, p: float, binom: np.ndarray) -> np.ndarray:
        """DPValue for set identification (label all vials)."""
        dpvalue = np.zeros((2, 2, n + 1))

        for g in range(2):
            for q in range(2):
                for m in range(n + 1):
                    n_r = n - m

                    # Action space: (k_m, k_r) = how many to label as uncont
                    utilities = []
                    multiplicities = []

                    for k_m in range(m + 1):
                        for k_r in range(n_r + 1):
                            ef1 = self._expected_f1(g, q, m, k_m, k_r, p, n, binom)
                            mult = binom[m, k_m] * binom[n_r, k_r]
                            utilities.append(ef1)
                            multiplicities.append(mult)

                    if not utilities:
                        continue

                    utilities = np.array(utilities)
                    multiplicities = np.array(multiplicities, dtype=float)

                    # Softmax with multiplicity weighting
                    exp_vals = multiplicities * np.exp(self.alpha_policy * (utilities - utilities.max()))
                    probs = exp_vals / exp_vals.sum()
                    dpvalue[g, q, m] = np.sum(probs * utilities)

        return dpvalue

    def _expected_f1(self, g: int, q: int, m: int, k_m: int, k_r: int,
                     p: float, n: int, binom: np.ndarray) -> float:
        """Expected F1 score for action (k_m, k_r) given response count m."""
        n_r = n - m
        k = k_m + k_r  # Total labeled as uncont

        if m == 0:
            return self._ef1_empty_response(g, q, k, p, n)

        # Literal listener semantics:
        # WHICH_UNCONT: t_m = m (all mentioned are uncont)
        # WHICH_CONT: t_m = 0 (all mentioned are cont)

        expected_f1 = 0.0
        total_prob = 0.0

        if q == 1:  # WHICH_UNCONT
            t_m = m  # All mentioned are uncont
            for W in range(m, n + 1):
                t_r = W - m
                prob_W = self._literal_listener_prob(W, m, n, p, binom, q)
                if prob_W < 1e-15:
                    continue
                total_prob += prob_W
                expected_f1 += prob_W * self._ef1_given_world(g, m, n_r, t_m, t_r, k_m, k_r, binom)
        else:  # WHICH_CONT
            t_m = 0  # All mentioned are cont
            for W in range(0, n - m + 1):
                t_r = W
                prob_W = self._literal_listener_prob(W, m, n, p, binom, q)
                if prob_W < 1e-15:
                    continue
                total_prob += prob_W
                expected_f1 += prob_W * self._ef1_given_world(g, m, n_r, t_m, t_r, k_m, k_r, binom)

        return expected_f1 / total_prob if total_prob > 0 else 0.0

    def _literal_listener_prob(self, W: int, m: int, n: int, p: float,
                                binom: np.ndarray, q: int) -> float:
        """P(W | m mentioned, question q) under literal listener."""
        prob_prior = binom[n, W] * (p ** W) * ((1 - p) ** (n - W))

        if q == 1:  # WHICH_UNCONT: need m uncont vials to mention
            prob_valid = binom[W, m] / binom[n, m] if binom[n, m] > 0 else 0.0
        else:  # WHICH_CONT: need m cont vials to mention
            prob_valid = binom[n - W, m] / binom[n, m] if binom[n, m] > 0 else 0.0

        return prob_prior * prob_valid

    def _ef1_given_world(self, g: int, m: int, n_r: int, t_m: int, t_r: int,
                          k_m: int, k_r: int, binom: np.ndarray) -> float:
        """E[F1] given world state (t_m uncont among mentioned, t_r among remaining)."""
        W = t_m + t_r
        k = k_m + k_r
        n = m + n_r

        # Mentioned vials: deterministic under literal semantics
        if t_m == m:  # All mentioned are uncont
            tp_m = k_m
        elif t_m == 0:  # All mentioned are cont
            tp_m = 0
        else:
            tp_m = min(k_m, t_m)

        # Remaining vials: marginalize over hypergeometric
        expected_f1 = 0.0
        for tp_r in range(max(0, k_r - (n_r - t_r)), min(k_r, t_r) + 1):
            prob = self._hypergeom_pmf(tp_r, n_r, t_r, k_r, binom)
            if prob < 1e-15:
                continue

            if g == 0:  # FIND_UNCONT
                f1 = self._f1_score(tp_m + tp_r, W, k)
            else:  # FIND_CONT
                tp_cont = (m - k_m - (t_m - tp_m)) + (n_r - k_r - (t_r - tp_r))
                f1 = self._f1_score(tp_cont, n - W, n - k)

            expected_f1 += prob * f1

        return expected_f1

    def _ef1_empty_response(self, g: int, q: int, k: int, p: float, n: int) -> float:
        """E[F1] for empty response (m=0)."""
        if q == 1:  # All vials are cont (W=0)
            if g == 0:  # FIND_UNCONT: nothing to find
                return 0.0
            else:  # FIND_CONT: all n are cont, we label n-k as cont
                return self._f1_score(n - k, n, n - k)
        else:  # All vials are uncont (W=n)
            if g == 1:  # FIND_CONT: nothing to find
                return 0.0
            else:  # FIND_UNCONT: all n are uncont, we label k as uncont
                return self._f1_score(k, n, k)

    # === UTILITIES ===

    def _binom_table(self, n_max: int) -> np.ndarray:
        """Precompute binomial coefficients."""
        return np.array([[comb(n, k, exact=True) if k <= n else 0
                          for k in range(n_max + 1)]
                         for n in range(n_max + 1)])

    def _hypergeom_pmf(self, k: int, N: int, K: int, n: int, binom: np.ndarray) -> float:
        """Hypergeometric PMF."""
        if N == 0:
            return 1.0 if k == 0 and n == 0 else 0.0
        if k < max(0, n - (N - K)) or k > min(n, K):
            return 0.0
        return (binom[K, k] * binom[N - K, n - k]) / binom[N, n]

    def _f1_score(self, tp: int, total_true: int, total_pred: int) -> float:
        """Compute F1 score."""
        if total_true == 0 or total_pred == 0 or tp == 0:
            return 0.0
        precision = tp / total_pred
        recall = tp / total_true
        return 2 * precision * recall / (precision + recall)


# Convenience function for backwards compatibility
def run_model(n_vials=10, contamination_rate=0.5, gamma=0.9, decision_type='singleton',
              alpha_r=5.0, alpha_q=5.0, alpha_policy=10.0, length_cost=0.1):
    """Run model with explicit parameters. Returns full result dict."""
    model = Model(alpha_r=alpha_r, alpha_q=alpha_q, alpha_policy=alpha_policy,
                  gamma=gamma, length_cost=length_cost)
    return model._run(n_vials, contamination_rate, decision_type)
