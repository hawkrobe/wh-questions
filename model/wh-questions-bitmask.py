#!/usr/bin/env python3
"""
Exact bitmask implementation of wh-question polarity model.

This is the O(2^N) reference implementation that enumerates all possible
world states and responses as bitmasks. Use for verification at small N.

The main model (wh-questions.py) uses O(N²) count-based representation with
hypergeometric weighting, which is faster but uses a different (principled)
marginalization over unobserved vial identities.
"""
import argparse
import json
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Exact bitmask wh-question model')
parser.add_argument('-n', '--n-vials', type=int, default=5, help='Number of vials (default: 5, max ~10)')
parser.add_argument('-r', '--rate', type=float, default=0.5, help='Contamination rate (default: 0.5)')
parser.add_argument('-g', '--gamma', type=float, default=0.9, help='Speaker confidence (default: 0.9)')
parser.add_argument('-d', '--decision-type', choices=['singleton', 'set_id'], default='singleton')
args = parser.parse_args()

N = args.n_vials
P_UNCONT = 1 - args.rate
GAMMA = args.gamma
DECISION_TYPE = args.decision_type

# Model parameters (fixed)
ALPHA_R = 5.0
ALPHA_Q = 5.0
ALPHA_POLICY = 10.0
LENGTH_COST = 0.1


def popcount(x):
    """Count set bits."""
    return bin(x).count('1')


def to_bits(x):
    """Convert int to bit array."""
    return [(x >> i) & 1 for i in range(N)]


def world_prior(w):
    """P(world w) where w is bitmask of uncontaminated vials."""
    bits = to_bits(w)
    p = 1.0
    for b in bits:
        p *= P_UNCONT if b else (1 - P_UNCONT)
    return p


def meaning(q, r, w):
    """Is response r a valid answer to question q in world w?"""
    if q == 1:  # WHICH_UNCONT
        queried = w
    else:  # WHICH_CONT
        queried = ((1 << N) - 1) ^ w

    if r == 0:
        return queried == 0  # "none" valid only if there are none
    return (r & queried) == r  # r must be subset of queried


def f1_score(tp, total_real, total_guess):
    """Compute F1 score."""
    # Note: returns 0 when either is empty (conservative choice)
    if total_real == 0 or total_guess == 0:
        return 0.0
    precision = tp / total_guess
    recall = tp / total_real
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def utility(g, w, a):
    """Utility of action a in world w for goal g."""
    if DECISION_TYPE == 'singleton':
        # Binary: did we pick a vial matching our goal?
        # a is a vial index (0 to N-1), w is bitmask of uncontaminated
        is_uncont = (w >> a) & 1
        if g == 0:  # FIND_UNCONT
            return 1.0 if is_uncont else 0.0
        else:  # FIND_CONT
            return 0.0 if is_uncont else 1.0
    else:  # set_id
        n_uncont = popcount(w)
        n_cont = N - n_uncont
        n_guess_uncont = popcount(a)
        n_guess_cont = N - n_guess_uncont
        tp_uncont = popcount(w & a)
        tp_cont = popcount(((1 << N) - 1 - w) & ((1 << N) - 1 - a))

        if g == 0:  # FIND_UNCONT
            return f1_score(tp_uncont, n_uncont, n_guess_uncont)
        else:  # FIND_CONT
            return f1_score(tp_cont, n_cont, n_guess_cont)


def compute_dpvalue(g, q, r):
    """Compute decision-maker's expected utility given question and response."""
    # Posterior over worlds given response
    posterior = {}
    total = 0.0
    for w in range(1 << N):
        if meaning(q, r, w):
            p = world_prior(w)
            posterior[w] = p
            total += p

    if total == 0:
        return 0.0

    for w in posterior:
        posterior[w] /= total

    # Action space depends on decision type
    if DECISION_TYPE == 'singleton':
        # Actions are single vials (0 to N-1)
        actions = list(range(N))
    else:
        # Actions are subsets (bitmasks)
        actions = list(range(1 << N))

    # Expected utility for each action under softmax policy
    action_values = []
    for a in actions:
        eu = sum(posterior[w] * utility(g, w, a) for w in posterior)
        action_values.append(eu)

    action_values = np.array(action_values)
    exp_vals = np.exp(ALPHA_POLICY * (action_values - action_values.max()))
    probs = exp_vals / exp_vals.sum()

    return np.sum(probs * action_values)


def speaker_belief(w, n_cont_known, n_uncont_known):
    """Speaker's belief about world w given their knowledge."""
    bits = to_bits(w)
    p = 1.0
    for i in range(N):
        if i < n_cont_known:
            # Known contaminated: believe uncont with prob 1-GAMMA
            p *= (1 - GAMMA) if bits[i] else GAMMA
        elif i < n_cont_known + n_uncont_known:
            # Known uncontaminated: believe uncont with prob GAMMA
            p *= GAMMA if bits[i] else (1 - GAMMA)
        else:
            # Unknown: use prior
            p *= P_UNCONT if bits[i] else (1 - P_UNCONT)
    return p


def compute_r0(q, n_cont_known, n_uncont_known):
    """Compute speaker response distribution for knowledge config."""
    response_weights = []

    for r in range(1 << N):
        # Listener posterior given response r
        listener_post = {}
        listener_total = 0.0
        for w in range(1 << N):
            if meaning(q, r, w):
                p = world_prior(w)
                listener_post[w] = p
                listener_total += p

        if listener_total == 0:
            response_weights.append(0.0)
            continue

        for w in listener_post:
            listener_post[w] /= listener_total

        # KL divergence: KL(listener || speaker)
        kl = 0.0
        for w in listener_post:
            lp = listener_post[w]
            sp = speaker_belief(w, n_cont_known, n_uncont_known)
            if lp > 1e-15:
                if sp < 1e-15:
                    kl = float('inf')
                    break
                kl += lp * np.log(lp / sp)

        if np.isinf(kl):
            response_weights.append(0.0)
        else:
            length = popcount(r)
            response_weights.append(np.exp(ALPHA_R * (-kl - LENGTH_COST * length)))

    total = sum(response_weights)
    if total == 0:
        return {0: 1.0}
    return {r: w / total for r, w in enumerate(response_weights) if w > 1e-10}


def compute_q1(g):
    """Compute questioner's probability of asking WHICH_UNCONT."""
    # All knowledge configs (uniform prior)
    configs = [(n_c, n_u) for n_c in range(N + 1) for n_u in range(N + 1 - n_c)]

    expected_value = {0: 0.0, 1: 0.0}

    for q in [0, 1]:
        total_ev = 0.0
        for n_c, n_u in configs:
            r0 = compute_r0(q, n_c, n_u)
            ev_config = sum(prob_r * compute_dpvalue(g, q, r) for r, prob_r in r0.items())
            total_ev += ev_config / len(configs)
        expected_value[q] = total_ev

    # Softmax over questions
    max_val = max(expected_value.values())
    exp_vals = np.array([np.exp(ALPHA_Q * (expected_value[q] - max_val)) for q in [0, 1]])
    probs = exp_vals / exp_vals.sum()

    return probs[1]  # P(WHICH_UNCONT)


if __name__ == "__main__":
    p_uncont_find_uncont = compute_q1(0)
    p_uncont_find_cont = compute_q1(1)

    result = {
        'model': 'bitmask',
        'n_vials': N,
        'contamination_rate': args.rate,
        'decision_type': DECISION_TYPE,
        'gamma': GAMMA,
        'find_uncontam': {
            'p_which_uncont': p_uncont_find_uncont,
            'p_which_cont': 1 - p_uncont_find_uncont,
        },
        'find_contam': {
            'p_which_uncont': p_uncont_find_cont,
            'p_which_cont': 1 - p_uncont_find_cont,
        }
    }

    print(json.dumps(result, indent=2))
