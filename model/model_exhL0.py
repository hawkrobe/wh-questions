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
        exh_scope: 'all' = unmentioned vials lack the queried property;
                   'known' = only unmentioned vials the speaker KNOWS lack it,
                   unknown unmentioned vials keep the prior p
    """
    alpha_r: float = 5.0
    alpha_q: float = 5.0
    alpha_policy: float = 10.0
    gamma: float = 0.9
    length_cost: float = 0.1
    exh_scope: str = 'all'

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
            dpvalue = self._compute_dpvalue_singleton(n, p, binom, configs)
        else:
            dpvalue = self._compute_dpvalue_set_id(n, p, binom, configs)

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
        if comp.total == 0 and self.exh_scope == 'all':
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
        #### NON-MENTIONED VIALS ARE ASSUMED TO NOT HAVE QUERIED PROPERTY ####
        n_nonmention_uncont = know.n_uncont - comp.m_uncont
        n_nonmention_cont = know.n_cont - comp.m_cont
        n_nonmention_unknown = know.n_unknown - comp.m_unknown
        if self.exh_scope == 'known':
            # Exhaustivity only over the speaker's knowledge: listener keeps prior p
            # for unmentioned unknown vials, so they contribute KL(p || p) = 0
            n_nonmention_unknown = 0
      
        if q == 1:  # WHICH_UNCONT: listener infers unmentioned vials are cont
        # P_speaker(cont) for each category:
            kl_nonmention = (
                n_nonmention_uncont * (-np.log(max(1 - gamma, 1e-15))) +    
                n_nonmention_cont * (-np.log(max(gamma, 1e-15))) +
                n_nonmention_unknown * (-np.log(max(1 - p, 1e-15)))            
            )
        else:  # WHICH_CONT: listener infers unmentioned vials are uncont
        # P_speaker(uncont) for each category:  
            kl_nonmention = (
                n_nonmention_cont * (-np.log(max(1- gamma, 1e-15))) + 
                n_nonmention_uncont * (-np.log(max(gamma, 1e-15))) +
                n_nonmention_unknown * (-np.log(max(p, 1e-15)))
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

        # Aggregate composition weights by total number of mentioned vials
        for comp, w in self._compute_composition_weights(q, know, p, binom).items():
            weights[comp.total] += w

        return weights

    def _compute_composition_weights(self, q: QuestionType, know: Knowledge,
                                     p: float, binom: np.ndarray) -> Dict[Composition, float]:
        """Unnormalized R0 weight for each composition (including multinomial count)."""
        weights = {}
        for comp in self._all_compositions(know):
            mult = (binom[know.n_cont, comp.m_cont] *
                    binom[know.n_uncont, comp.m_uncont] *
                    binom[know.n_unknown, comp.m_unknown])
            kl = self._compute_kl(q, know, comp, p)
            if np.isfinite(kl):
                weights[comp] = mult * np.exp(self.alpha_r * (-kl - self.length_cost * comp.total))
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

    def _compute_posterior(self, q: QuestionType, n: int, p: float, binom: np.ndarray,
                           configs: list) -> list:
        """
        Decision-maker's posterior over world counts given m mentioned vials.

        Marginalizes over the respondent's knowledge:
            P(w | r) = sum_k P(k | r) * P(w | k),  P(k | r) ∝ P(k) * R0(r | k)
        where P(k) is uniform over configs and P(w | k) is the speaker belief
        (each vial uncont with prob γ, 1-γ, or p depending on its knowledge category).

        Returns post[m]: array of shape (m+1, n-m+1), post[m][t_m, t_r] =
        P(t_m uncont among mentioned, t_r uncont among remaining | m).
        """
        gamma = self.gamma
        post = [np.zeros((m + 1, n - m + 1)) for m in range(n + 1)]

        for know in configs:
            weights = self._compute_composition_weights(q, know, p, binom)
            total = sum(weights.values())
            if total <= 0:
                continue

            for comp, w in weights.items():
                m = comp.total
                w = w / total / len(configs)    # P(k) * R0(comp | k)
                pmf_m = self._n_uncont_pmf(comp.m_cont, comp.m_uncont, comp.m_unknown, gamma, p)
                pmf_r = self._n_uncont_pmf(know.n_cont - comp.m_cont,
                                           know.n_uncont - comp.m_uncont,
                                           know.n_unknown - comp.m_unknown, gamma, p)
                post[m] += w * np.outer(pmf_m, pmf_r)

        for m in range(n + 1):
            z = post[m].sum()
            if z > 0:
                post[m] /= z

        return post

    def _n_uncont_pmf(self, n_c: int, n_u: int, n_k: int, gamma: float, p: float) -> np.ndarray:
        """PMF of # uncont vials among n_c known-cont, n_u known-uncont, n_k unknown."""
        pmf = np.ones(1)
        for count, p_uncont in ((n_c, 1 - gamma), (n_u, gamma), (n_k, p)):
            for _ in range(count):
                pmf = np.convolve(pmf, [1 - p_uncont, p_uncont])
        return pmf

    def _compute_dpvalue_singleton(self, n: int, p: float, binom: np.ndarray,
                                   configs: list) -> np.ndarray:
        """DPValue for singleton decision (pick one vial)."""
        dpvalue = np.zeros((2, 2, n + 1))

        for q in range(2):
            post = self._compute_posterior(q, n, p, binom, configs)

            for m in range(n + 1):
                # P(uncont) for a mentioned / remaining vial
                t_m = np.arange(m + 1)
                t_r = np.arange(n - m + 1)
                p_uncont_mentioned = post[m].sum(axis=1) @ t_m / m if m > 0 else 0.0
                p_uncont_remaining = post[m].sum(axis=0) @ t_r / (n - m) if m < n else 0.0

                for g in range(2):
                    # FIND_UNCONT scores uncont picks; FIND_CONT scores cont picks
                    u_mentioned = p_uncont_mentioned if g == 0 else 1 - p_uncont_mentioned
                    u_remaining = p_uncont_remaining if g == 0 else 1 - p_uncont_remaining

                    # Softmax over picking a mentioned vs a remaining vial
                    # (m = 0 or m = n leaves only one option)
                    w_mentioned = m * np.exp(self.alpha_policy * u_mentioned)
                    w_remaining = (n - m) * np.exp(self.alpha_policy * u_remaining)

                    p_mentioned = w_mentioned / (w_mentioned + w_remaining)
                    dpvalue[g, q, m] = p_mentioned * u_mentioned + (1 - p_mentioned) * u_remaining

        return dpvalue

    def _compute_dpvalue_set_id(self, n: int, p: float, binom: np.ndarray,
                                configs: list) -> np.ndarray:
        """DPValue for set identification (label all vials), scored by expected F1."""
        dpvalue = np.zeros((2, 2, n + 1))

        for q in range(2):
            post = self._compute_posterior(q, n, p, binom, configs)

            for m in range(n + 1):
                n_r = n - m
                actions = [(k_m, k_r) for k_m in range(m + 1) for k_r in range(n_r + 1)]
                multiplicities = np.array([binom[m, k_m] * binom[n_r, k_r]
                                           for k_m, k_r in actions], dtype=float)

                for g in range(2):
                    # Action space: (k_m, k_r) = how many to label as uncont
                    utilities = np.array([
                        sum(post[m][t_m, t_r] *
                            self._ef1_given_world(g, m, n_r, t_m, t_r, k_m, k_r, binom)
                            for t_m in range(m + 1) for t_r in range(n_r + 1)
                            if post[m][t_m, t_r] > 1e-15)
                        for k_m, k_r in actions])

                    # Softmax with multiplicity weighting
                    exp_vals = multiplicities * np.exp(self.alpha_policy * (utilities - utilities.max()))
                    probs = exp_vals / exp_vals.sum()
                    dpvalue[g, q, m] = np.sum(probs * utilities)

        return dpvalue

    def _ef1_given_world(self, g: int, m: int, n_r: int, t_m: int, t_r: int,
                          k_m: int, k_r: int, binom: np.ndarray) -> float:
        """E[F1] given world state (t_m uncont among mentioned, t_r among remaining)."""
        W = t_m + t_r
        k = k_m + k_r
        n = m + n_r

        # Labels within each group are a random subset w.r.t. true status: hypergeometric
        expected_f1 = 0.0
        for tp_m in range(max(0, k_m - (m - t_m)), min(k_m, t_m) + 1):
            prob_m = self._hypergeom_pmf(tp_m, m, t_m, k_m, binom)
            for tp_r in range(max(0, k_r - (n_r - t_r)), min(k_r, t_r) + 1):
                prob = prob_m * self._hypergeom_pmf(tp_r, n_r, t_r, k_r, binom)
                if prob < 1e-15:
                    continue

                if g == 0:  # FIND_UNCONT
                    f1 = self._f1_score(tp_m + tp_r, W, k)
                else:  # FIND_CONT
                    tp_cont = (m - k_m - (t_m - tp_m)) + (n_r - k_r - (t_r - tp_r))
                    f1 = self._f1_score(tp_cont, n - W, n - k)

                expected_f1 += prob * f1

        return expected_f1

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
              alpha_r=5.0, alpha_q=5.0, alpha_policy=10.0, length_cost=0.1, exh_scope='all'):
    """Run model with explicit parameters. Returns full result dict."""
    model = Model(alpha_r=alpha_r, alpha_q=alpha_q, alpha_policy=alpha_policy,
                  gamma=gamma, length_cost=length_cost, exh_scope=exh_scope)
    return model._run(n_vials, contamination_rate, decision_type)
