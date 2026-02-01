"""
Symmetric Wh-question model for question-asking behavior.

This model predicts which question a rational questioner will ask (e.g.,
"Which vials are contaminated?" vs "Which are uncontaminated?") based on
their goal and the decision problem they face.

Key insight: The speaker chooses a COMPOSITION (m_u, m_?) specifying how many
vials to mention from each knowledge category. The combinatorial factor
C(n_u, m_u) * C(n_?, m_?) accounts for equivalent specific responses.

The KL divergence decomposes per-vial for product distributions, so we can
compute it analytically for each composition. This reduces complexity from
O(2^N) to O(N^2), enabling efficient computation at larger N values.
"""

import sys
import argparse
import json
from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum
import numpy as onp  # For precomputation
from scipy.special import comb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Symmetric Wh-question model for question-asking behavior.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-n', '--n-vials',
        type=int,
        default=5,
        help='Number of vials in the scenario'
    )
    parser.add_argument(
        '-r', '--rate',
        type=float,
        default=0.5,
        help='Contamination rate (probability each vial is contaminated)'
    )
    parser.add_argument(
        '-d', '--decision-type',
        choices=['singleton', 'set_id'],
        default='singleton',
        help='Decision structure: singleton (pick one vial) or set_id (classify all)'
    )
    parser.add_argument(
        '-g', '--gamma',
        type=float,
        default=0.9,
        help='Speaker confidence parameter (probability speaker is correct about known vials)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug information about R0 and DPValue distributions'
    )
    return parser.parse_args()


# Parse arguments before any precomputation
_args = parse_args()
N_VIALS = _args.n_vials
CONTAMINATION_RATE = _args.rate
DECISION_TYPE = _args.decision_type
P_CONFIDENT = _args.gamma
DEBUG = _args.debug

# Model parameters
ALPHA_R = 5.0
ALPHA_Q = 5.0
LENGTH_COST = 0.1

class Question(IntEnum):
    WHICH_CONTAMINATED = 0
    WHICH_UNCONTAMINATED = 1

class Goal(IntEnum):
    FIND_UNCONTAMINATED = 0
    FIND_CONTAMINATED = 1

# =============================================================================
# STATE SPACES
# =============================================================================

# World = count of uncontaminated vials (0 to N)
World = np.arange(N_VIALS + 1)

# Response = count of vials mentioned (0 to N)
Response = np.arange(N_VIALS + 1)

# Knowledge configs: (n_cont_known, n_uncont_known)
KNOWLEDGE_CONFIGS = [(n_c, n_u)
                     for n_c in range(N_VIALS + 1)
                     for n_u in range(N_VIALS + 1 - n_c)]
N_CONFIGS = len(KNOWLEDGE_CONFIGS)
KnowledgeConfig = np.arange(N_CONFIGS)

# Composition: (m_u, m_?) = vials mentioned from known-uncont and unknown
# For WHICH_UNCONT, these are the vials claimed to be uncontaminated
COMPOSITIONS = [(m_u, m_q)
                for m_u in range(N_VIALS + 1)
                for m_q in range(N_VIALS + 1 - m_u)]
N_COMPOSITIONS = len(COMPOSITIONS)
Composition = np.arange(N_COMPOSITIONS)

# Lookup arrays
_N_CONT = np.array([c[0] for c in KNOWLEDGE_CONFIGS])
_N_UNCONT = np.array([c[1] for c in KNOWLEDGE_CONFIGS])
_M_U = np.array([c[0] for c in COMPOSITIONS])
_M_Q = np.array([c[1] for c in COMPOSITIONS])

# Binomial coefficients
_BINOM_NP = onp.array([[comb(n, k, exact=True) if k <= n else 0
                        for k in range(N_VIALS + 1)]
                       for n in range(N_VIALS + 1)])
_BINOM = np.array(_BINOM_NP)  # JAX version for traced indexing

@jax.jit
def get_n_cont(k):
    return _N_CONT[k]

@jax.jit
def get_n_uncont(k):
    return _N_UNCONT[k]

@jax.jit
def get_m_u(comp):
    return _M_U[comp]

@jax.jit
def get_m_q(comp):
    return _M_Q[comp]

# =============================================================================
# PRECOMPUTE KL DIVERGENCES AND RESPONSE WEIGHTS
# =============================================================================

def kl_binary(p_l, p_s):
    """KL divergence for binary distributions."""
    if p_s <= 0 or p_s >= 1:
        return float('inf')
    kl = 0.0
    if p_l > 0:
        kl += p_l * onp.log(p_l / p_s)
    if p_l < 1:
        kl += (1 - p_l) * onp.log((1 - p_l) / (1 - p_s))
    return kl


def compute_kl_full_composition(q, n_c, n_u, m_match, m_other, m_unk, gamma, p):
    """
    Compute KL[P_L || P_S] for a full response composition.

    The speaker can mention vials from ALL three categories:
    - m_match: vials from "known-match" (speaker thinks they match the question)
    - m_other: vials from "known-other" (speaker thinks they DON'T match the question)
    - m_unk: vials from unknown

    This handles the case where the speaker might mention a vial they think is
    contaminated when asked "which are uncontaminated" (pays higher KL cost).

    For WHICH_UNCONT: known-match = known-uncont (gamma), known-other = known-cont (1-gamma)
    For WHICH_CONT: known-match = known-cont (gamma), known-other = known-uncont (1-gamma)
    """
    n_q = N_VIALS - n_c - n_u  # unknown vials

    # Determine category sizes based on question
    if q == Question.WHICH_UNCONTAMINATED:
        n_known_match = n_u  # speaker thinks uncont, P=gamma
        n_known_other = n_c  # speaker thinks cont, P=1-gamma
    else:  # WHICH_CONTAMINATED
        n_known_match = n_c  # speaker thinks cont, P=gamma
        n_known_other = n_u  # speaker thinks uncont, P=1-gamma

    # Check validity
    if m_match > n_known_match or m_other > n_known_other or m_unk > n_q:
        return float('inf')

    M = m_match + m_other + m_unk  # total mentioned

    # SPECIAL CASE: m=0 means "there are none"
    if M == 0:
        # Listener believes ALL vials are the OPPOSITE type
        # KL contribution: listener says P(match)=0, so KL = -log(1 - P_S(match))

        # Known-match vials: speaker thinks P(match) = gamma
        kl_known_match = n_known_match * (-onp.log(1 - gamma)) if gamma < 1 else float('inf')

        # Known-other vials: speaker thinks P(match) = 1-gamma
        kl_known_other = n_known_other * (-onp.log(gamma)) if gamma > 0 else float('inf')

        # Unknown vials: speaker thinks P(match) = 0.5
        kl_unknown = n_q * onp.log(2)

        return kl_known_match + kl_known_other + kl_unknown

    # NORMAL CASE: m > 0
    # KL contributions from MENTIONED vials:
    # - Listener says P(match) = 1 for all mentioned
    # - Speaker belief varies by category

    # m_match from known-match: speaker says P(match) = gamma
    kl_mentioned_match = m_match * (-onp.log(gamma)) if gamma > 0 else (float('inf') if m_match > 0 else 0)

    # m_other from known-other: speaker says P(match) = 1-gamma
    kl_mentioned_other = m_other * (-onp.log(1 - gamma)) if gamma < 1 else (float('inf') if m_other > 0 else 0)

    # m_unk from unknown: speaker says P(match) = 0.5
    kl_mentioned_unk = m_unk * onp.log(2)

    # KL contributions from NON-MENTIONED vials:
    # Listener uses prior p for match property

    # Non-mentioned known-match: (n_known_match - m_match)
    kl_nonmention_match = (n_known_match - m_match) * kl_binary(p, gamma)

    # Non-mentioned known-other: (n_known_other - m_other)
    kl_nonmention_other = (n_known_other - m_other) * kl_binary(p, 1 - gamma)

    # Non-mentioned unknown: (n_q - m_unk)
    kl_nonmention_unk = (n_q - m_unk) * kl_binary(p, 0.5)

    total_kl = (kl_mentioned_match + kl_mentioned_other + kl_mentioned_unk +
                kl_nonmention_match + kl_nonmention_other + kl_nonmention_unk)

    return total_kl


def precompute_response_weights():
    """
    Precompute the marginalized weight for each (question, knowledge, m_total).

    Weight(q, k, m) = Σ_{compositions (m_match, m_other, m_unk) with sum = m}
                      C(n_match, m_match) * C(n_other, m_other) * C(n_q, m_unk)
                      * exp(-α * (KL + λm))

    Now considers all three knowledge categories (match, other, unknown).
    """
    gamma = P_CONFIDENT
    p = 1 - CONTAMINATION_RATE  # P(uncontaminated) under prior

    weights = onp.zeros((2, N_CONFIGS, N_VIALS + 1))

    for q in range(2):
        # p_match: listener's prior for the queried property
        p_match = p if q == Question.WHICH_UNCONTAMINATED else (1 - p)

        for k_idx, (n_c, n_u) in enumerate(KNOWLEDGE_CONFIGS):
            n_q = N_VIALS - n_c - n_u

            # Determine category sizes
            if q == Question.WHICH_UNCONTAMINATED:
                n_known_match = n_u
                n_known_other = n_c
            else:
                n_known_match = n_c
                n_known_other = n_u

            # Iterate over all possible compositions
            for m_match in range(n_known_match + 1):
                for m_other in range(n_known_other + 1):
                    for m_unk in range(n_q + 1):
                        m_total = m_match + m_other + m_unk

                        if m_total > N_VIALS:
                            continue

                        # Combinatorial factor
                        comb_factor = (_BINOM_NP[n_known_match, m_match] *
                                       _BINOM_NP[n_known_other, m_other] *
                                       _BINOM_NP[n_q, m_unk])

                        # KL divergence
                        kl = compute_kl_full_composition(q, n_c, n_u, m_match, m_other,
                                                         m_unk, gamma, p_match)

                        if onp.isfinite(kl):
                            w = comb_factor * onp.exp(ALPHA_R * (-kl - LENGTH_COST * m_total))
                            weights[q, k_idx, m_total] += w

    return np.array(weights)

_RESPONSE_WEIGHTS = precompute_response_weights()

@jax.jit
def response_weight(q, k, m):
    """Marginalized weight for response count m given question q and knowledge k."""
    return _RESPONSE_WEIGHTS[q, k, m]

# =============================================================================
# WORLD PRIOR AND LITERAL POSTERIOR
# =============================================================================

def precompute_world_prior():
    """Binomial prior over count of uncontaminated vials."""
    p = 1 - CONTAMINATION_RATE
    return np.array([_BINOM_NP[N_VIALS, w] * (p ** w) * ((1-p) ** (N_VIALS - w))
                     for w in range(N_VIALS + 1)])

_WORLD_PRIOR = precompute_world_prior()

@jax.jit
def world_prior(w):
    return _WORLD_PRIOR[w]

@jax.jit
def literal_posterior(q, m, w):
    """
    Listener's posterior P(W = w | m vials mentioned as having property).

    Given m specific vials have the queried property, the remaining N-m
    vials follow the prior. So:
    P(W = w | m mentioned) = Binomial(w - m; N - m, p) for WHICH_UNCONT
    P(W = w | m mentioned) = Binomial(N - w - m; N - m, 1-p) for WHICH_CONT
    """
    p = 1 - CONTAMINATION_RATE
    n_remaining = N_VIALS - m

    # For WHICH_UNCONT: m uncont mentioned, so W >= m
    # P(W = w) = P(w - m uncont among remaining n_remaining)
    w_remaining_uncont = w - m
    valid_uncont = (w_remaining_uncont >= 0) & (w_remaining_uncont <= n_remaining)
    prob_uncont = np.where(
        valid_uncont,
        _BINOM[n_remaining, w_remaining_uncont] *
        (p ** w_remaining_uncont) * ((1-p) ** (n_remaining - w_remaining_uncont)),
        0.0
    )

    # For WHICH_CONT: m cont mentioned, so N - W >= m, i.e., W <= N - m
    # P(W = w) = P(w uncont among remaining n_remaining)
    valid_cont = (w >= 0) & (w <= n_remaining)
    prob_cont = np.where(
        valid_cont,
        _BINOM[n_remaining, w] * (p ** w) * ((1-p) ** (n_remaining - w)),
        0.0
    )

    # Handle m = 0 case: "there are none"
    # For WHICH_UNCONT with m=0: only valid if w = 0
    # For WHICH_CONT with m=0: only valid if w = N
    prob_uncont_m0 = np.where(w == 0, 1.0, 0.0)
    prob_cont_m0 = np.where(w == N_VIALS, 1.0, 0.0)

    prob_uncont_final = np.where(m > 0, prob_uncont, prob_uncont_m0)
    prob_cont_final = np.where(m > 0, prob_cont, prob_cont_m0)

    return np.where(q == Question.WHICH_UNCONTAMINATED, prob_uncont_final, prob_cont_final)

# =============================================================================
# DECISION UTILITY
# =============================================================================

ALPHA_POLICY = 10.0  # Soft-max temperature for action choice


def _hypergeom_pmf(k, N, K, n):
    """
    Hypergeometric PMF: P(X=k) where X ~ Hypergeometric(N, K, n).

    N = population size
    K = number of success states in population
    n = number of draws
    k = number of observed successes
    """
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return (_BINOM_NP[K, k] * _BINOM_NP[N - K, n - k]) / _BINOM_NP[N, n]


def _compute_expected_f1(g, q, m, k, p):
    """
    Compute expected F1 score for SET_ID decision structure.

    Uses proper E[F1] by summing over hypergeometric distribution of TP values,
    since F1 is non-linear and E[F1] ≠ F1(E[TP]).

    Args:
        g: Goal (0=FIND_UNCONT, 1=FIND_CONT)
        q: Question (0=WHICH_CONT, 1=WHICH_UNCONT)
        m: Number of vials mentioned in response
        k: Number of vials guessed as uncontaminated (action)
        p: Base rate P(uncontaminated)

    Returns:
        Expected F1 score on the goal-relevant category.
    """
    n_remaining = N_VIALS - m

    # Determine constraints based on question type
    if q == 1:  # WHICH_UNCONT: m vials are definitely uncontaminated
        # Must include all m mentioned as "uncont", so k >= m
        if k < m:
            return 0.0  # Invalid action
        k_from_remaining = k - m  # Additional guesses from remaining
    else:  # WHICH_CONT: m vials are definitely contaminated
        # Cannot include mentioned vials as "uncont", so k <= N - m
        if k > n_remaining:
            return 0.0  # Invalid action
        k_from_remaining = k  # All guesses come from remaining

    # Handle edge case: m=0 means "there are none"
    if m == 0:
        if q == 1:  # WHICH_UNCONT with m=0: all are contaminated
            if g == 0:  # FIND_UNCONT
                return 0.0  # No uncont vials exist
            else:  # FIND_CONT
                if k == N_VIALS:
                    return 0.0
                recall = (N_VIALS - k) / N_VIALS
                return 2 * 1.0 * recall / (1.0 + recall)
        else:  # WHICH_CONT with m=0: all are uncontaminated
            if g == 0:  # FIND_UNCONT
                if k == 0:
                    return 0.0
                recall = k / N_VIALS
                return 2 * 1.0 * recall / (1.0 + recall)
            else:  # FIND_CONT
                return 0.0  # No cont vials exist

    # Normal case: m > 0
    # Compute E[F1] over world posterior AND over hypergeometric TP distribution
    expected_f1 = 0.0
    total_prob = 0.0

    for w_remaining in range(n_remaining + 1):
        # Probability of this world state (binomial over remaining vials)
        prob_w = _BINOM_NP[n_remaining, w_remaining] * (p ** w_remaining) * ((1-p) ** (n_remaining - w_remaining))

        if prob_w < 1e-15:
            continue

        if q == 1:  # WHICH_UNCONT
            W = m + w_remaining  # Total uncontaminated
        else:  # WHICH_CONT
            W = w_remaining  # Total uncontaminated (mentioned are cont)

        # For this world, compute E[F1] over hypergeometric distribution of TP
        # TP from remaining ~ Hypergeometric(n_remaining, w_remaining, k_from_remaining)

        expected_f1_this_world = 0.0

        for tp_remaining in range(min(k_from_remaining, w_remaining) + 1):
            # Hypergeometric probability
            prob_tp = _hypergeom_pmf(tp_remaining, n_remaining, w_remaining, k_from_remaining)

            if prob_tp < 1e-15:
                continue

            # Compute exact F1 for this TP value
            if g == 0:  # FIND_UNCONT
                if q == 1:  # WHICH_UNCONT
                    tp_uncont = m + tp_remaining  # m mentioned + tp from remaining
                else:  # WHICH_CONT
                    tp_uncont = tp_remaining

                if W == 0 or k == 0:
                    f1 = 0.0
                else:
                    precision = tp_uncont / k
                    recall = tp_uncont / W
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
            else:  # FIND_CONT
                n_cont = N_VIALS - W
                k_cont = N_VIALS - k

                # TP for contaminated = correctly identified as cont
                # FP_uncont (guessed uncont but actually cont) = k_from_remaining - tp_remaining
                # So TP_cont from remaining = (n_remaining - k_from_remaining) guessed as cont
                #                             intersected with (n_remaining - w_remaining) true cont
                cont_in_remaining = n_remaining - w_remaining
                guessed_cont_from_remaining = n_remaining - k_from_remaining

                # TP_cont from remaining ~ Hypergeometric, but we can derive it:
                # We guessed k_from_remaining as uncont, got tp_remaining correct
                # So we guessed (k_from_remaining - tp_remaining) cont vials as uncont (false negatives)
                # Remaining cont vials correctly identified = cont_in_remaining - (k_from_remaining - tp_remaining)
                #                                          = cont_in_remaining - k_from_remaining + tp_remaining
                tp_cont_remaining = max(0, cont_in_remaining - k_from_remaining + tp_remaining)

                if q == 1:  # WHICH_UNCONT
                    tp_cont = tp_cont_remaining
                else:  # WHICH_CONT
                    tp_cont = m + tp_cont_remaining  # m mentioned are definitely cont

                if n_cont == 0 or k_cont == 0:
                    f1 = 0.0
                else:
                    precision_cont = tp_cont / k_cont
                    recall_cont = tp_cont / n_cont
                    if precision_cont + recall_cont > 0:
                        f1 = 2 * precision_cont * recall_cont / (precision_cont + recall_cont)
                    else:
                        f1 = 0.0

            expected_f1_this_world += prob_tp * f1

        expected_f1 += prob_w * expected_f1_this_world
        total_prob += prob_w

    return expected_f1 / total_prob if total_prob > 0 else 0.0


def _precompute_dpvalue_singleton():
    """Precompute DPValue for SINGLETON decision structure."""
    p = 1 - CONTAMINATION_RATE

    dpvalue = onp.zeros((2, 2, N_VIALS + 1))

    for g in range(2):
        for q in range(2):
            goal_matches = (g == 0 and q == 1) or (g == 1 and q == 0)

            for m in range(N_VIALS + 1):
                n_remaining = N_VIALS - m

                if m == 0:
                    if q == 1:  # WHICH_UNCONT with m=0: all are cont
                        dpvalue[g, q, 0] = 0.0 if g == 0 else 1.0
                    else:  # WHICH_CONT with m=0: all are uncont
                        dpvalue[g, q, 0] = 1.0 if g == 0 else 0.0
                elif n_remaining == 0:
                    dpvalue[g, q, m] = 1.0 if goal_matches else 0.0
                else:
                    util_mentioned = 1.0 if goal_matches else 0.0
                    util_remaining = p if g == 0 else (1 - p)

                    weight_mentioned = m * onp.exp(ALPHA_POLICY * util_mentioned)
                    weight_remaining = n_remaining * onp.exp(ALPHA_POLICY * util_remaining)
                    total = weight_mentioned + weight_remaining

                    p_pick_mentioned = weight_mentioned / total
                    p_pick_remaining = weight_remaining / total

                    dpvalue[g, q, m] = p_pick_mentioned * util_mentioned + p_pick_remaining * util_remaining

    return dpvalue


def _precompute_dpvalue_set_id():
    """Precompute DPValue for SET_ID decision structure."""
    p = 1 - CONTAMINATION_RATE

    dpvalue = onp.zeros((2, 2, N_VIALS + 1))

    for g in range(2):
        for q in range(2):
            for m in range(N_VIALS + 1):
                # Compute expected utility for each action k, then soft-max
                utilities = []
                valid_actions = []

                for k in range(N_VIALS + 1):
                    f1 = _compute_expected_f1(g, q, m, k, p)
                    # Check validity based on constraints
                    if q == 1:  # WHICH_UNCONT: k >= m
                        valid = (k >= m)
                    else:  # WHICH_CONT: k <= N - m
                        valid = (k <= N_VIALS - m)

                    if valid:
                        utilities.append(f1)
                        valid_actions.append(k)

                if len(utilities) == 0:
                    dpvalue[g, q, m] = 0.0
                    continue

                utilities = onp.array(utilities)
                # Soft-max over valid actions
                weights = onp.exp(ALPHA_POLICY * utilities)
                probs = weights / onp.sum(weights)

                # Expected utility under soft-max policy
                dpvalue[g, q, m] = onp.sum(probs * utilities)

    return dpvalue


def _precompute_dpvalue():
    """Precompute DPValue based on decision type."""
    if DECISION_TYPE == "singleton":
        return np.array(_precompute_dpvalue_singleton())
    else:  # set_id
        return np.array(_precompute_dpvalue_set_id())

_DPVALUE_TABLE = _precompute_dpvalue()

@jax.jit
def dpvalue_lookup(g, q, m):
    """Look up precomputed DPValue."""
    return _DPVALUE_TABLE[g, q, m]

# =============================================================================
# MODEL
# =============================================================================

@memo
def R0[q: Question, k: KnowledgeConfig, m: Response]():
    """Speaker's response distribution (marginalized over compositions)."""
    speaker: knows(q, k)
    speaker: chooses(m in Response, wpp=response_weight(q, k, m))
    return Pr[speaker.m == m]


@memo
def DPValue[g: Goal, q: Question, m: Response]():
    """Value of decision problem given goal, question, and response count."""
    decider: knows(g, q, m)
    # Use precomputed DPValue table
    # The soft-max is based on expected utility (expectation over world posterior),
    # which doesn't depend on the specific world w, only on (g, q, m)
    return dpvalue_lookup(g, q, m)


@memo
def Q1[g: Goal, q: Question]():
    """Questioner's choice distribution over questions."""
    questioner: knows(g)
    questioner: chooses(q in Question, wpp=exp({ALPHA_Q} * imagine[
        scenario: knows(q),
        scenario: given(k in KnowledgeConfig, wpp=1),
        scenario: chooses(m in Response, wpp=R0[q, k, m]()),
        E[DPValue[g, scenario.q, scenario.m]()]
    ]))
    return Pr[questioner.q == q]


if __name__ == "__main__":
    print(f"Running model: N={N_VIALS}, rate={CONTAMINATION_RATE}, "
          f"decision={DECISION_TYPE}, gamma={P_CONFIDENT}", file=sys.stderr)

    # Debug output
    if DEBUG:
        r0 = R0()
        print(f"\nDebug: R0 distributions", file=sys.stderr)
        for k_idx, (n_c, n_u) in enumerate(KNOWLEDGE_CONFIGS[:3]):  # First 3 configs
            print(f"\nConfig (n_c={n_c}, n_u={n_u}):", file=sys.stderr)
            for q, qname in [(0, "WHICH_CONT"), (1, "WHICH_UNCONT")]:
                print(f"  {qname}:", file=sys.stderr)
                for m in range(N_VIALS + 1):
                    prob = float(r0[q, k_idx, m])
                    if prob > 1e-6:
                        print(f"    m={m}: P={prob:.4f}", file=sys.stderr)

        dp = DPValue()
        print("\nDebug: DPValue for FIND_UNCONT (g=0):", file=sys.stderr)
        for q, qname in [(0, "WHICH_CONT"), (1, "WHICH_UNCONT")]:
            print(f"  {qname}:", file=sys.stderr)
            for m in range(min(N_VIALS + 1, 6)):  # First 6 values
                val = float(dp[0, q, m])
                print(f"    m={m}: DPValue={val:.4f}", file=sys.stderr)

    q1 = Q1()

    results = {
        'n_vials': N_VIALS,
        'contamination_rate': CONTAMINATION_RATE,
        'decision_type': DECISION_TYPE,
        'gamma': P_CONFIDENT,
        'find_uncontam': {
            'p_which_cont': float(q1[Goal.FIND_UNCONTAMINATED, Question.WHICH_CONTAMINATED]),
            'p_which_uncont': float(q1[Goal.FIND_UNCONTAMINATED, Question.WHICH_UNCONTAMINATED])
        },
        'find_contam': {
            'p_which_cont': float(q1[Goal.FIND_CONTAMINATED, Question.WHICH_CONTAMINATED]),
            'p_which_uncont': float(q1[Goal.FIND_CONTAMINATED, Question.WHICH_UNCONTAMINATED])
        }
    }

    print(json.dumps(results))
