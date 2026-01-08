"""
Wh-question model with decision problem structure manipulation

Two decision problems:
1. SINGLETON: Pick ONE vial (current model) - utility from single choice
2. SET_ID: Classify ALL vials - utility from overall accuracy

The prediction: Goal-alignment effect should be larger for SINGLETON
(where mention-some suffices) than for SET_ID (where exhaustivity matters more).
"""

import sys
import json
from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

# Parameters from command line
N_VIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CONTAMINATION_RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
DECISION_TYPE = sys.argv[3] if len(sys.argv) > 3 else "singleton"  # "singleton" or "set_id"

N_WORLDS = 2 ** N_VIALS
ALL_VIALS = (1 << N_VIALS) - 1  # Bitmask with all vial bits set

World = np.arange(N_WORLDS)
Response = np.arange(N_WORLDS)
Vial = np.arange(N_VIALS)

# Action space depends on decision type
Action = Vial if DECISION_TYPE == "singleton" else np.arange(N_WORLDS)

KNOWLEDGE_CONFIGS = [(n_cont, n_uncont)
                     for n_cont in range(N_VIALS + 1)
                     for n_uncont in range(N_VIALS + 1 - n_cont)]
N_CONFIGS = len(KNOWLEDGE_CONFIGS)
KnowledgeConfig = np.arange(N_CONFIGS)

_N_CONT_ARRAY = np.array([c[0] for c in KNOWLEDGE_CONFIGS])
_N_UNCONT_ARRAY = np.array([c[1] for c in KNOWLEDGE_CONFIGS])

# Model parameters
ALPHA_R = 5.0
ALPHA_POLICY = 10.0
ALPHA_Q = 5.0
LENGTH_COST = 0.1
P_CONFIDENT = 0.9

class Question(IntEnum):
    WHICH_CONTAMINATED = 0
    WHICH_UNCONTAMINATED = 1

class Goal(IntEnum):
    FIND_UNCONTAMINATED = 0
    AVOID_CONTAMINATION = 1


# Bitmask helpers
@jax.jit
def to_bits(x):
    """Convert bitmask to array of bits."""
    return np.array([(x >> i) & 1 for i in range(N_VIALS)])

@jax.jit
def popcount(x):
    """Count set bits in bitmask."""
    return np.sum(to_bits(x))


# Knowledge config accessors
@jax.jit
def get_n_cont(k):
    return _N_CONT_ARRAY[k]

@jax.jit
def get_n_uncont(k):
    return _N_UNCONT_ARRAY[k]


# Semantics and priors
@jax.jit
def meaning(q, r, w):
    queried = np.where(q == Question.WHICH_UNCONTAMINATED, w, ALL_VIALS ^ w)
    return np.where(r > 0, (r & queried) == r, queried == 0)

@jax.jit
def response_length(r):
    return popcount(r)

@jax.jit
def world_prior(w):
    bits = to_bits(w)
    p_uncontaminated = 1.0 - CONTAMINATION_RATE
    return np.prod(np.where(bits, p_uncontaminated, CONTAMINATION_RATE))

@jax.jit
def get_speaker_belief(w, n_cont, n_uncont, p_conf):
    bits = to_bits(w)
    indices = np.arange(N_VIALS)
    p_uncontaminated = np.where(
        indices < n_cont,
        1.0 - p_conf,
        np.where(indices < n_cont + n_uncont, p_conf, 0.5)
    )
    return np.prod(np.where(bits, p_uncontaminated, 1.0 - p_uncontaminated))

# Utility functions
@jax.jit
def singleton_utility(g, w, a):
    """Utility of picking vial a in world w for goal g."""
    is_uncontaminated = (w >> a) & 1
    return np.where(g == Goal.FIND_UNCONTAMINATED, is_uncontaminated, 1 - is_uncontaminated)

@jax.jit
def set_id_utility(g, w, a):
    """Utility of guessing bitmask a as uncontaminated vials. Returns recall on goal-relevant category."""
    # FIND: recall on uncontaminated vials
    tp_uncont = popcount(w & a)
    total_uncont = popcount(w)
    recall_uncont = np.where(total_uncont > 0, tp_uncont / total_uncont, 1.0)

    # AVOID: recall on contaminated vials
    tp_cont = popcount((ALL_VIALS ^ w) & (ALL_VIALS ^ a))
    total_cont = popcount(ALL_VIALS ^ w)
    recall_cont = np.where(total_cont > 0, tp_cont / total_cont, 1.0)

    return np.where(g == Goal.FIND_UNCONTAMINATED, recall_uncont, recall_cont)


@jax.jit
def action_utility(g, w, a):
    """Dispatch to appropriate utility function based on decision type."""
    if DECISION_TYPE == "singleton":
        return singleton_utility(g, w, a)
    else:
        return set_id_utility(g, w, a)


@memo
def R0[q: Question, k: KnowledgeConfig, r: Response]():
    speaker: knows(q, k)
    speaker: thinks[
        world: knows(k),
        world: chooses(w in World, wpp=get_speaker_belief(w, get_n_cont(k), get_n_uncont(k), {P_CONFIDENT}))
    ]
    speaker: chooses(r in Response, wpp=exp({ALPHA_R} * imagine[
        literal: knows(q, r),
        literal: given(w in World, wpp=meaning(q, r, w) * world_prior(w)),
        -KL[literal.w | world.w] - {LENGTH_COST} * response_length(r)
    ]))
    return Pr[speaker.r == r]


@memo
def DPValue[g: Goal, q: Question, r: Response]():
    decider: knows(g, q, r)
    decider: chooses(a in Action, wpp=exp({ALPHA_POLICY} * imagine[
        world: knows(g, q, r),
        world: chooses(w in World, wpp=meaning(q, r, w) * world_prior(w)),
        E[action_utility(g, world.w, a)]
    ]))
    return imagine[
        world: knows(g, q, r),
        world: chooses(w in World, wpp=meaning(q, r, w) * world_prior(w)),
        E[action_utility(g, world.w, decider.a)]
    ]


@memo
def Q1[g: Goal, q: Question]():
    questioner: knows(g)
    questioner: chooses(q in Question, wpp=exp({ALPHA_Q} * imagine[
        scenario: knows(q),
        scenario: given(k in KnowledgeConfig, wpp=1),
        scenario: chooses(r in Response, wpp=R0[q, k, r]()),
        E[DPValue[g, scenario.q, scenario.r]()]
    ]))
    return Pr[questioner.q == q]


if __name__ == "__main__":
    q1 = Q1()
    
    results = {
        'n_vials': N_VIALS,
        'contamination_rate': CONTAMINATION_RATE,
        'decision_type': DECISION_TYPE,
        'find_uncontam': {
            'p_which_cont': float(q1[Goal.FIND_UNCONTAMINATED, Question.WHICH_CONTAMINATED]),
            'p_which_uncont': float(q1[Goal.FIND_UNCONTAMINATED, Question.WHICH_UNCONTAMINATED])
        },
        'avoid_contam': {
            'p_which_cont': float(q1[Goal.AVOID_CONTAMINATION, Question.WHICH_CONTAMINATED]),
            'p_which_uncont': float(q1[Goal.AVOID_CONTAMINATION, Question.WHICH_UNCONTAMINATED])
        }
    }
    
    print(json.dumps(results))
