from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

# =============================================================================
# Setup: Domain definitions
# =============================================================================

N_VIALS = 5                     # Number of vials in the scenario
N_WORLDS = 2 ** N_VIALS         # Possible states of the vials

World = np.arange(N_WORLDS)     # Each world is an integer 0..(2^N_VIALS - 1)
Response = np.arange(N_WORLDS)  # Responses are also subsets, encoded the same way
Vial = np.arange(N_VIALS)       # Individual vials: 0, 1, 2, 3, 4
Action = Vial                   # Questioner's action = picking a vial

# Knowledge configurations: all valid (n_cont, n_uncont) pairs where n_cont + n_uncont <= N_VIALS
# Enumerated as a single domain since the two values constrain each other
KNOWLEDGE_CONFIGS = [(n_cont, n_uncont)
                     for n_cont in range(N_VIALS + 1)
                     for n_uncont in range(N_VIALS + 1 - n_cont)]
N_CONFIGS = len(KNOWLEDGE_CONFIGS)  # (N+1)(N+2)/2 = 21 for N_VIALS=5
KnowledgeConfig = np.arange(N_CONFIGS)

# Precompute arrays for fast lookup inside JIT functions
_N_CONT_ARRAY = np.array([c[0] for c in KNOWLEDGE_CONFIGS])
_N_UNCONT_ARRAY = np.array([c[1] for c in KNOWLEDGE_CONFIGS])

@jax.jit
def get_n_cont(k):
    """Extract n_contaminated from knowledge config index k."""
    return _N_CONT_ARRAY[k]

@jax.jit
def get_n_uncont(k):
    """Extract n_uncontaminated from knowledge config index k."""
    return _N_UNCONT_ARRAY[k]

def config_to_str(k):
    """Human-readable string for knowledge config k."""
    n_cont, n_uncont = KNOWLEDGE_CONFIGS[k]
    n_unc = N_VIALS - n_cont - n_uncont
    return f"({n_cont}, {n_uncont}, {n_unc})"

# =============================================================================
# Model Parameters
# =============================================================================

CONTAMINATION_RATE = 0.5        # P(any given vial is contaminated), i.i.d. Bernoulli
ALPHA_R = 5.0                   # R0's softmax temperature when choosing responses
ALPHA_POLICY = 10.0             # Q1's softmax for picking a vial after getting an answer
ALPHA_Q = 5.0                   # Q1's rationality when choosing which question to ask
LENGTH_COST = 0.1               # Penalty per vial mentioned (encourages brevity)
P_CONFIDENT = 0.9               # R0's confidence level (0.9 = "90% sure")

# There are two wh-questions Q1 can ask
class Question(IntEnum):
    WHICH_CONTAMINATED = 0      # "Which vials are contaminated?"
    WHICH_UNCONTAMINATED = 1    # "Which vials are uncontaminated?"

# Q1's possible goals
class Goal(IntEnum):
    FIND_UNCONTAMINATED = 0     # Want to pick an uncontaminated vial
    AVOID_CONTAMINATION = 1     # Want to avoid contaminated vials 


# =============================================================================
# Semantics: Truth conditions and utility functions
# =============================================================================

@jax.jit
def meaning(q, r, w):
    """Is response r true in world w for question q? (r must be subset of queried vials)"""
    all_vials = (1 << N_VIALS) - 1
    queried = np.where(q == Question.WHICH_UNCONTAMINATED, w, all_vials ^ w)
    return (r & queried) == r


@jax.jit
def vial_uncontaminated(w, i):
    """
    Checks if vial i is uncontaminated in world w.
    Uses bit extraction: shift w right by i positions, then check the lowest bit.
    """
    return (w >> i) & 1


# =============================================================================
# L0: Literal Listener
# =============================================================================

@jax.jit
def world_prior(w):
    """Prior P(w) under i.i.d. Bernoulli contamination model."""
    bits = np.array([(w >> i) & 1 for i in range(N_VIALS)])
    p_uncontaminated = 1.0 - CONTAMINATION_RATE
    return np.prod(np.where(bits, p_uncontaminated, CONTAMINATION_RATE))


@memo
def L0[q: Question, r: Response, w: World]():
    """Literal listener: P(w | q, r) via Bayesian update."""
    listener: knows(q, r)
    listener: chooses(w in World, wpp=meaning(q, r, w) * world_prior(w))
    return Pr[listener.w == w]


# =============================================================================
# R0: Respondent with Partial Knowledge
# =============================================================================

@jax.jit
def get_speaker_belief(w, n_cont, n_uncont, p_conf):
    """
    Compute R0's belief that the world is w.
      - First n_cont vials: R0 believes ARE contaminated with probability p_conf
      - Next n_uncont vials: R0 believes are NOT contaminated with probability p_conf
      - Remaining vials: R0 is uncertain (50/50)
    """
    bits = np.array([(w >> i) & 1 for i in range(N_VIALS)])
    indices = np.arange(N_VIALS)
    p_uncontaminated = np.where(
        indices < n_cont,
        1.0 - p_conf,
        np.where(indices < n_cont + n_uncont, p_conf, 0.5)
    )
    return np.prod(np.where(bits, p_uncontaminated, 1.0 - p_uncontaminated))


@jax.jit
def response_length(r):
    """
    Count how many vials are mentioned in response r (population count).

    Used for brevity cost: longer responses are penalized.
    Example: r=5 (binary 101) mentions 2 vials -> returns 2
    """
    return np.sum(np.array([(r >> i) & 1 for i in range(N_VIALS)]))


@memo
def R0[q: Question, k: KnowledgeConfig, r: Response]():
    """Speaker chooses response to minimize KL(L0's posterior || own beliefs) + length cost."""
    speaker: knows(q, k)
    speaker: thinks[
        world: knows(k),
        world: chooses(w in World, wpp=get_speaker_belief(w, get_n_cont(k), get_n_uncont(k), {P_CONFIDENT}))
    ]
    speaker: chooses(r in Response, wpp=exp({ALPHA_R} * imagine[
        listener: knows(q, r),
        listener: chooses(w in World, wpp=L0[q, r, w]()),
        -KL[listener.w | world.w] - {LENGTH_COST} * response_length(r)
    ]))
    return Pr[speaker.r == r]


# =============================================================================
# Q1: Questioner with Goals (marginalizes over unknown R0 knowledge)
# =============================================================================

@jax.jit
def action_utility(g, w, a):
    """Utility of picking vial a in world w for goal g."""
    is_uncontaminated = vial_uncontaminated(w, a)
    return np.where(g == Goal.FIND_UNCONTAMINATED, is_uncontaminated, 1 - is_uncontaminated)

def DPValue(g, q, r):
    """Value of decision problem after hearing response r to question q."""
    l0 = L0()
    posterior = np.array([l0[q, r, w] for w in World])
    expected_utilities = np.array([
        np.sum(posterior * np.array([action_utility(g, w, a) for w in World]))
        for a in Action
    ])
    policy = jax.nn.softmax(ALPHA_POLICY * expected_utilities)
    return np.sum(policy * expected_utilities)

@memo
def Q1[g: Goal, q: Question]():
    """Questioner chooses question, marginalizing over R0's unknown knowledge config."""
    questioner: knows(g)
    questioner: chooses(q in Question, wpp=exp({ALPHA_Q} * imagine[
        scenario: knows(q),
        scenario: given(k in KnowledgeConfig, wpp=1),
        scenario: chooses(r in Response, wpp=R0[q, k, r]()),
        E[DPValue(g, scenario.q, scenario.r)]
    ]))
    return Pr[questioner.q == q]


# =============================================================================
# Demo: Helper functions and main entry point
# =============================================================================

def vials_to_str(x):
    """Convert a bitmask to a human-readable set notation."""
    vials = [i for i in range(N_VIALS) if (x >> i) & 1]
    return "{" + ",".join(str(v) for v in vials) + "}" if vials else "{}"


def print_r0_responses():
    """Show how R0's responses vary by knowledge configuration."""
    r0 = R0()
    for k in [0, 7, 14, 20]:
        if k >= N_CONFIGS:
            continue
        print(f"  {config_to_str(k)}:")
        for q in Question:
            top_resp = max(Response, key=lambda r: float(r0[q, k, r]))
            prob = float(r0[q, k, top_resp])
            print(f"    {q.name}: '{vials_to_str(top_resp)}' ({prob:.2f})")


def print_confidence_scaling():
    """Show how Q1's preferences vary with R0's confidence level (P_CONFIDENT)."""
    global P_CONFIDENT
    original = P_CONFIDENT

    print("\nQ1's preferences by R0 confidence level:")
    print(f"  {'P_CONFIDENT':<12} {'FIND_UNCONTAM':<20} {'AVOID_CONTAM':<20}")
    print(f"  {'':12} {'P(Which uncont?)':<20} {'P(Which cont?)':<20}")
    print("  " + "-" * 52)

    for p_conf in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        P_CONFIDENT = p_conf
        q1 = Q1()
        p_find = float(q1[Goal.FIND_UNCONTAMINATED, Question.WHICH_UNCONTAMINATED])
        p_avoid = float(q1[Goal.AVOID_CONTAMINATION, Question.WHICH_CONTAMINATED])
        print(f"  {p_conf:<12.2f} {p_find:<20.3f} {p_avoid:<20.3f}")

    P_CONFIDENT = original  # restore


def main():
    print(f"Wh-Questions Model: {N_VIALS} vials, {N_WORLDS} worlds, {N_CONFIGS} knowledge configs\n")
    print("R0's responses for selected knowledge configs:")
    print("(N_conf_contaminated, N_conf_uncontaminated, uncertain)")
    print_r0_responses()

    q1 = Q1()
    print("\nQ1's question preferences (marginalizing over R0's knowledge):")
    for g in Goal:
        p_cont = float(q1[g, Question.WHICH_CONTAMINATED])
        p_uncont = float(q1[g, Question.WHICH_UNCONTAMINATED])
        print(f"  {g.name}: P(Which cont?)={p_cont:.2f}, P(Which uncont?)={p_uncont:.2f}")

    print_confidence_scaling()



if __name__ == "__main__":
    main()
