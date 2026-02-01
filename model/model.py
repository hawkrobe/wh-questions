"""
Wh-question polarity model: core implementation.

This module implements a decision-theoretic RSA model predicting which question
a rational questioner will ask (e.g., "Which vials are contaminated?" vs
"Which are uncontaminated?") based on their goal and decision problem.

Uses O(N²) count-based representation instead of O(2^N) bitmasks.
"""

import numpy as np
from scipy.special import comb
from dataclasses import dataclass


@dataclass
class Model:
    """
    Wh-question polarity model.

    Hyperparameters (set once):
        alpha_r: Speaker rationality
        alpha_q: Questioner rationality
        alpha_policy: Decision-maker rationality
        gamma: Speaker confidence (P(correct) for known vials)
        length_cost: Response brevity penalty
    """
    alpha_r: float = 5.0
    alpha_q: float = 5.0
    alpha_policy: float = 10.0
    gamma: float = 0.9
    length_cost: float = 0.1

    def predict(self, goal: str, decision_type: str, p_uncont: float, n_vials: int) -> float:
        """
        Predict P(WHICH_UNCONT) for a given scenario.

        Args:
            goal: 'find_uncont' or 'find_cont'
            decision_type: 'singleton' or 'set_id'
            p_uncont: Base rate P(uncontaminated) = 1 - contamination_rate
            n_vials: Number of vials

        Returns:
            P(WHICH_UNCONT) - probability of asking "Which are uncontaminated?"
        """
        g = 0 if goal == 'find_uncont' else 1
        result = self._run(n_vials, 1 - p_uncont, decision_type)
        return result['find_uncontam']['p_which_uncont'] if g == 0 else result['find_contam']['p_which_uncont']

    def predict_all(self, decision_type: str, p_uncont: float, n_vials: int) -> dict:
        """
        Predict for both goals at once.

        Returns:
            Dict with 'find_uncont' and 'find_cont' keys, each containing P(WHICH_UNCONT)
        """
        result = self._run(n_vials, 1 - p_uncont, decision_type)
        return {
            'find_uncont': result['find_uncontam']['p_which_uncont'],
            'find_cont': result['find_contam']['p_which_uncont']
        }

    def _run(self, n_vials: int, contamination_rate: float, decision_type: str) -> dict:
        """Run the full model and return results."""
        n = n_vials
        p = 1 - contamination_rate  # P(uncontaminated)

        # Precompute binomial coefficients
        binom = self._binom_table(n)

        # Knowledge configs
        configs = [(n_c, n_u) for n_c in range(n + 1) for n_u in range(n + 1 - n_c)]

        # Compute response weights for all (question, knowledge, response_count)
        response_weights = self._compute_response_weights(n, p, binom, configs)

        # Normalize to get R0 distribution
        r0 = np.zeros((2, len(configs), n + 1))
        for q in range(2):
            for k in range(len(configs)):
                total = response_weights[q, k, :].sum()
                if total > 0:
                    r0[q, k, :] = response_weights[q, k, :] / total

        # Compute DPValue
        if decision_type == 'singleton':
            dpvalue = self._compute_dpvalue_singleton(n, p, binom)
        else:
            dpvalue = self._compute_dpvalue_set_id(n, p, binom)

        # Compute Q1 for each goal
        q1 = {}
        for g in range(2):
            ev_q = np.zeros(2)
            for q in range(2):
                total_ev = 0.0
                for k_idx in range(len(configs)):
                    ev_config = 0.0
                    for m in range(n + 1):
                        ev_config += r0[q, k_idx, m] * dpvalue[g, q, m]
                    total_ev += ev_config / len(configs)
                ev_q[q] = total_ev

            # Softmax
            exp_vals = np.exp(self.alpha_q * (ev_q - ev_q.max()))
            probs = exp_vals / exp_vals.sum()
            q1[g] = probs[1]

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

    def _binom_table(self, n_max):
        """Precompute binomial coefficients."""
        return np.array([[comb(n, k, exact=True) if k <= n else 0
                          for k in range(n_max + 1)]
                         for n in range(n_max + 1)])

    def _kl_binary(self, p_l, p_s):
        """KL divergence for binary distributions."""
        if p_s <= 0 or p_s >= 1:
            return float('inf')
        kl = 0.0
        if p_l > 0:
            kl += p_l * np.log(p_l / p_s)
        if p_l < 1:
            kl += (1 - p_l) * np.log((1 - p_l) / (1 - p_s))
        return kl

    def _compute_kl_composition(self, q, n_c, n_u, m_match, m_other, m_unk, p, n):
        """Compute KL divergence for a response composition."""
        gamma = self.gamma
        n_q = n - n_c - n_u

        if q == 1:  # WHICH_UNCONT
            n_known_match, n_known_other = n_u, n_c
        else:  # WHICH_CONT
            n_known_match, n_known_other = n_c, n_u

        if m_match > n_known_match or m_other > n_known_other or m_unk > n_q:
            return float('inf')

        M = m_match + m_other + m_unk

        if M == 0:
            kl_known_match = n_known_match * (-np.log(1 - gamma)) if gamma < 1 else float('inf')
            kl_known_other = n_known_other * (-np.log(gamma)) if gamma > 0 else float('inf')
            kl_unknown = n_q * np.log(2)
            return kl_known_match + kl_known_other + kl_unknown

        kl_mentioned_match = m_match * (-np.log(gamma)) if gamma > 0 else (float('inf') if m_match > 0 else 0)
        kl_mentioned_other = m_other * (-np.log(1 - gamma)) if gamma < 1 else (float('inf') if m_other > 0 else 0)
        kl_mentioned_unk = m_unk * np.log(2)
        kl_nonmention_match = (n_known_match - m_match) * self._kl_binary(p, gamma)
        kl_nonmention_other = (n_known_other - m_other) * self._kl_binary(p, 1 - gamma)
        kl_nonmention_unk = (n_q - m_unk) * self._kl_binary(p, 0.5)

        return (kl_mentioned_match + kl_mentioned_other + kl_mentioned_unk +
                kl_nonmention_match + kl_nonmention_other + kl_nonmention_unk)

    def _compute_response_weights(self, n, p, binom, configs):
        """Precompute response weights."""
        weights = np.zeros((2, len(configs), n + 1))

        for q in range(2):
            p_match = p if q == 1 else (1 - p)

            for k_idx, (n_c, n_u) in enumerate(configs):
                n_q = n - n_c - n_u
                n_known_match = n_u if q == 1 else n_c
                n_known_other = n_c if q == 1 else n_u

                for m_match in range(n_known_match + 1):
                    for m_other in range(n_known_other + 1):
                        for m_unk in range(n_q + 1):
                            m_total = m_match + m_other + m_unk
                            if m_total > n:
                                continue

                            comb_factor = binom[n_known_match, m_match] * binom[n_known_other, m_other] * binom[n_q, m_unk]
                            kl = self._compute_kl_composition(q, n_c, n_u, m_match, m_other, m_unk, p_match, n)

                            if np.isfinite(kl):
                                w = comb_factor * np.exp(self.alpha_r * (-kl - self.length_cost * m_total))
                                weights[q, k_idx, m_total] += w

        return weights

    def _hypergeom_pmf(self, k, N, K, n, binom):
        """Hypergeometric PMF."""
        if k < max(0, n - (N - K)) or k > min(n, K):
            return 0.0
        return (binom[K, k] * binom[N - K, n - k]) / binom[N, n]

    def _compute_expected_f1(self, g, q, m, k, p, n, binom):
        """Compute expected F1 score for SET_ID using hypergeometric distribution."""
        n_remaining = n - m

        if q == 1:  # WHICH_UNCONT
            if k < m:
                return 0.0
            k_from_remaining = k - m
        else:  # WHICH_CONT
            if k > n_remaining:
                return 0.0
            k_from_remaining = k

        if m == 0:
            if q == 1:
                if g == 0:
                    return 0.0
                else:
                    if k == n:
                        return 0.0
                    recall = (n - k) / n
                    return 2 * recall / (1.0 + recall)
            else:
                if g == 0:
                    if k == 0:
                        return 0.0
                    recall = k / n
                    return 2 * recall / (1.0 + recall)
                else:
                    return 0.0

        expected_f1 = 0.0
        total_prob = 0.0

        for w_remaining in range(n_remaining + 1):
            prob_w = binom[n_remaining, w_remaining] * (p ** w_remaining) * ((1-p) ** (n_remaining - w_remaining))
            if prob_w < 1e-15:
                continue

            W = m + w_remaining if q == 1 else w_remaining
            expected_f1_this_world = 0.0

            for tp_remaining in range(min(k_from_remaining, w_remaining) + 1):
                prob_tp = self._hypergeom_pmf(tp_remaining, n_remaining, w_remaining, k_from_remaining, binom)
                if prob_tp < 1e-15:
                    continue

                if g == 0:  # FIND_UNCONT
                    tp_uncont = m + tp_remaining if q == 1 else tp_remaining
                    if W == 0 or k == 0:
                        f1 = 0.0
                    else:
                        precision = tp_uncont / k
                        recall = tp_uncont / W
                        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                else:  # FIND_CONT
                    n_cont = n - W
                    k_cont = n - k
                    cont_in_remaining = n_remaining - w_remaining
                    tp_cont_remaining = max(0, cont_in_remaining - k_from_remaining + tp_remaining)
                    tp_cont = tp_cont_remaining if q == 1 else m + tp_cont_remaining

                    if n_cont == 0 or k_cont == 0:
                        f1 = 0.0
                    else:
                        precision_cont = tp_cont / k_cont
                        recall_cont = tp_cont / n_cont
                        f1 = 2 * precision_cont * recall_cont / (precision_cont + recall_cont) if precision_cont + recall_cont > 0 else 0.0

                expected_f1_this_world += prob_tp * f1

            expected_f1 += prob_w * expected_f1_this_world
            total_prob += prob_w

        return expected_f1 / total_prob if total_prob > 0 else 0.0

    def _compute_dpvalue_singleton(self, n, p, binom):
        """Precompute DPValue for singleton decision."""
        dpvalue = np.zeros((2, 2, n + 1))

        for g in range(2):
            for q in range(2):
                goal_matches = (g == 0 and q == 1) or (g == 1 and q == 0)

                for m in range(n + 1):
                    n_remaining = n - m

                    if m == 0:
                        if q == 1:
                            dpvalue[g, q, 0] = 0.0 if g == 0 else 1.0
                        else:
                            dpvalue[g, q, 0] = 1.0 if g == 0 else 0.0
                    elif n_remaining == 0:
                        dpvalue[g, q, m] = 1.0 if goal_matches else 0.0
                    else:
                        util_mentioned = 1.0 if goal_matches else 0.0
                        util_remaining = p if g == 0 else (1 - p)
                        weight_mentioned = m * np.exp(self.alpha_policy * util_mentioned)
                        weight_remaining = n_remaining * np.exp(self.alpha_policy * util_remaining)
                        total = weight_mentioned + weight_remaining
                        p_pick_mentioned = weight_mentioned / total
                        p_pick_remaining = weight_remaining / total
                        dpvalue[g, q, m] = p_pick_mentioned * util_mentioned + p_pick_remaining * util_remaining

        return dpvalue

    def _compute_dpvalue_set_id(self, n, p, binom):
        """Precompute DPValue for set_id decision."""
        dpvalue = np.zeros((2, 2, n + 1))

        for g in range(2):
            for q in range(2):
                for m in range(n + 1):
                    utilities = []
                    for k in range(n + 1):
                        f1 = self._compute_expected_f1(g, q, m, k, p, n, binom)
                        valid = (k >= m) if q == 1 else (k <= n - m)
                        if valid:
                            utilities.append(f1)

                    if len(utilities) == 0:
                        dpvalue[g, q, m] = 0.0
                    else:
                        utilities = np.array(utilities)
                        weights = np.exp(self.alpha_policy * utilities)
                        probs = weights / np.sum(weights)
                        dpvalue[g, q, m] = np.sum(probs * utilities)

        return dpvalue


# Convenience function for backwards compatibility
def run_model(n_vials=10, contamination_rate=0.5, gamma=0.9, decision_type='singleton',
              alpha_r=5.0, alpha_q=5.0, alpha_policy=10.0, length_cost=0.1):
    """Run model with explicit parameters. Returns full result dict."""
    model = Model(alpha_r=alpha_r, alpha_q=alpha_q, alpha_policy=alpha_policy,
                  gamma=gamma, length_cost=length_cost)
    return model._run(n_vials, contamination_rate, decision_type)
