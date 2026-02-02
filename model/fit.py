#!/usr/bin/env python3
"""
Model fitting for wh-question polarity experiments.

Fits each model separately to the experimental data using maximum likelihood.
RSA models have k=2 free parameters: alpha (tied) and gamma.
"""

import numpy as np
from scipy.optimize import minimize
from model import Model

# ============================================================================
# EXPERIMENTAL DATA
# ============================================================================

# Experiment 1: Goal × Base Rate (singleton only)
# Format: (goal, base_rate): (n_chose_uncont, n_total)
EXP1_DATA = {
    ('uncont', 0.2): (10, 23),
    ('uncont', 0.5): (17, 36),
    ('uncont', 0.8): (21, 31),
    ('cont', 0.2): (4, 28),
    ('cont', 0.5): (2, 21),
    ('cont', 0.8): (7, 24),
}

# Experiment 2: Goal × Decision Structure (base_rate = 0.5)
# Format: decision_structure -> goal -> (n_chose_uncont, n_total)
EXP2_DATA = {
    'singleton': {'find_uncont': (29, 38), 'find_cont': (9, 42)},
    'set_id': {'find_uncont': (24, 36), 'find_cont': (18, 44)}
}

N_VIALS = 10
LENGTH_COST = 0.01  # Fixed, not fit


# ============================================================================
# LIKELIHOOD FUNCTIONS
# ============================================================================

def neg_log_lik_exp1(predictions: dict, data: dict = EXP1_DATA) -> float:
    """Negative log-likelihood for Experiment 1."""
    nll = 0.0
    for (goal, br), (s, t) in data.items():
        p = np.clip(predictions[(goal, br)], 1e-10, 1 - 1e-10)
        nll -= s * np.log(p) + (t - s) * np.log(1 - p)
    return nll


def neg_log_lik_exp2(predictions: dict, data: dict = EXP2_DATA) -> float:
    """Negative log-likelihood for Experiment 2."""
    nll = 0.0
    for dt in ['singleton', 'set_id']:
        for goal in ['find_uncont', 'find_cont']:
            s, t = data[dt][goal]
            p = np.clip(predictions[dt][goal], 1e-10, 1 - 1e-10)
            nll -= s * np.log(p) + (t - s) * np.log(1 - p)
    return nll


def aic(nll: float, k: int) -> float:
    """AIC = -2 * log-likelihood + 2 * k."""
    return 2 * nll + 2 * k


# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

def predict_exp1_rsa(alpha: float, gamma: float, decision_type: str = 'singleton') -> dict:
    """Get RSA predictions for Experiment 1."""
    model = Model(alpha_r=alpha, alpha_q=alpha, alpha_policy=alpha,
                  gamma=gamma, length_cost=LENGTH_COST)
    predictions = {}
    for (goal, br) in EXP1_DATA:
        g_name = 'find_uncont' if goal == 'uncont' else 'find_cont'
        predictions[(goal, br)] = model.predict(g_name, decision_type, p_uncont=1-br, n_vials=N_VIALS)
    return predictions


def predict_exp2_rsa(alpha: float, gamma: float, use_full: bool = True) -> dict:
    """Get RSA predictions for Experiment 2.

    Args:
        alpha: Tied rationality parameter
        gamma: Speaker confidence parameter
        use_full: If True, use appropriate decision type for each condition.
                  If False, use same decision type for both (for ablations).
    """
    model = Model(alpha_r=alpha, alpha_q=alpha, alpha_policy=alpha,
                  gamma=gamma, length_cost=LENGTH_COST)
    predictions = {}
    for dt in ['singleton', 'set_id']:
        # For ablations, override the decision type
        if use_full:
            model_dt = dt
        else:
            model_dt = dt  # Will be overridden by caller
        preds = model.predict_all(model_dt, p_uncont=0.5, n_vials=N_VIALS)
        predictions[dt] = preds
    return predictions


def predict_exp2_ablation(alpha: float, gamma: float, semantics: str) -> dict:
    """RSA ablation: use same semantics for both conditions."""
    model = Model(alpha_r=alpha, alpha_q=alpha, alpha_policy=alpha,
                  gamma=gamma, length_cost=LENGTH_COST)
    preds = model.predict_all(semantics, p_uncont=0.5, n_vials=N_VIALS)
    return {'singleton': preds, 'set_id': preds}


# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def fit_rsa_exp1(decision_type: str = 'singleton') -> dict:
    """Fit RSA model to Experiment 1 data."""
    def objective(params):
        alpha, gamma = params
        if alpha <= 0 or gamma <= 0.5 or gamma >= 1:
            return 1e10
        try:
            preds = predict_exp1_rsa(alpha, gamma, decision_type)
            return neg_log_lik_exp1(preds)
        except Exception:
            return 1e10

    # Use L-BFGS-B with bounds
    result = minimize(objective, [3.0, 0.8],
                      method='L-BFGS-B',
                      bounds=[(0.1, 20), (0.51, 0.99)],
                      options={'maxiter': 100})

    alpha, gamma = result.x
    return {
        'alpha': alpha,
        'gamma': gamma,
        'nll': result.fun,
        'aic': aic(result.fun, k=2),  # k=2: alpha, gamma
        'predictions': predict_exp1_rsa(alpha, gamma, decision_type)
    }


def fit_rsa_exp2_full() -> dict:
    """Fit Full RSA model to Experiment 2 (correct semantics for each condition)."""
    def objective(params):
        alpha, gamma = params
        if alpha <= 0 or gamma <= 0.5 or gamma >= 1:
            return 1e10
        try:
            model = Model(alpha_r=alpha, alpha_q=alpha, alpha_policy=alpha,
                          gamma=gamma, length_cost=LENGTH_COST)
            preds = {
                'singleton': model.predict_all('singleton', p_uncont=0.5, n_vials=N_VIALS),
                'set_id': model.predict_all('set_id', p_uncont=0.5, n_vials=N_VIALS)
            }
            return neg_log_lik_exp2(preds)
        except Exception:
            return 1e10

    # Use L-BFGS-B with bounds (faster than Nelder-Mead grid search)
    result = minimize(objective, [3.0, 0.8],
                      method='L-BFGS-B',
                      bounds=[(0.1, 20), (0.51, 0.99)],
                      options={'maxiter': 100})

    alpha, gamma = result.x
    model = Model(alpha_r=alpha, alpha_q=alpha, alpha_policy=alpha,
                  gamma=gamma, length_cost=LENGTH_COST)
    preds = {
        'singleton': model.predict_all('singleton', p_uncont=0.5, n_vials=N_VIALS),
        'set_id': model.predict_all('set_id', p_uncont=0.5, n_vials=N_VIALS)
    }

    return {
        'alpha': alpha,
        'gamma': gamma,
        'nll': result.fun,
        'aic': aic(result.fun, k=2),  # k=2: alpha, gamma
        'predictions': preds
    }


def fit_rsa_exp2_ablation(semantics: str) -> dict:
    """Fit ablated RSA model (same semantics for both conditions)."""
    def objective(params):
        alpha, gamma = params
        if alpha <= 0 or gamma <= 0.5 or gamma >= 1:
            return 1e10
        try:
            preds = predict_exp2_ablation(alpha, gamma, semantics)
            return neg_log_lik_exp2(preds)
        except Exception:
            return 1e10

    # Use L-BFGS-B with bounds
    result = minimize(objective, [3.0, 0.8],
                      method='L-BFGS-B',
                      bounds=[(0.1, 20), (0.51, 0.99)],
                      options={'maxiter': 100})

    alpha, gamma = result.x
    preds = predict_exp2_ablation(alpha, gamma, semantics)

    return {
        'alpha': alpha,
        'gamma': gamma,
        'nll': result.fun,
        'aic': aic(result.fun, k=2),  # k=2: alpha, gamma
        'predictions': preds
    }


# ============================================================================
# BASELINE MODELS
# ============================================================================

def fit_null_exp1() -> dict:
    """Null model: 50% baseline for all conditions."""
    predictions = {key: 0.5 for key in EXP1_DATA}
    nll = neg_log_lik_exp1(predictions)
    return {'nll': nll, 'aic': aic(nll, k=0), 'predictions': predictions}


def fit_null_exp2() -> dict:
    """Null model: 50% baseline for all conditions."""
    predictions = {
        'singleton': {'find_uncont': 0.5, 'find_cont': 0.5},
        'set_id': {'find_uncont': 0.5, 'find_cont': 0.5}
    }
    nll = neg_log_lik_exp2(predictions)
    return {'nll': nll, 'aic': aic(nll, k=0), 'predictions': predictions}


def fit_goal_matching_exp1() -> dict:
    """Goal-matching heuristic: MLE estimate per goal (k=2)."""
    # MLE: p = successes / total for each goal
    uncont_s = sum(EXP1_DATA[k][0] for k in EXP1_DATA if k[0] == 'uncont')
    uncont_t = sum(EXP1_DATA[k][1] for k in EXP1_DATA if k[0] == 'uncont')
    cont_s = sum(EXP1_DATA[k][0] for k in EXP1_DATA if k[0] == 'cont')
    cont_t = sum(EXP1_DATA[k][1] for k in EXP1_DATA if k[0] == 'cont')

    p_uncont = uncont_s / uncont_t
    p_cont = cont_s / cont_t

    predictions = {k: (p_uncont if k[0] == 'uncont' else p_cont) for k in EXP1_DATA}
    nll = neg_log_lik_exp1(predictions)

    return {
        'p_uncont_goal': p_uncont,
        'p_cont_goal': p_cont,
        'nll': nll,
        'aic': aic(nll, k=2),  # k=2: p_uncont_goal, p_cont_goal
        'predictions': predictions
    }


def fit_goal_matching_exp2() -> dict:
    """Goal-matching heuristic: MLE estimate per goal, same for both structures (k=2)."""
    # Pool across decision structures
    uncont_s = EXP2_DATA['singleton']['find_uncont'][0] + EXP2_DATA['set_id']['find_uncont'][0]
    uncont_t = EXP2_DATA['singleton']['find_uncont'][1] + EXP2_DATA['set_id']['find_uncont'][1]
    cont_s = EXP2_DATA['singleton']['find_cont'][0] + EXP2_DATA['set_id']['find_cont'][0]
    cont_t = EXP2_DATA['singleton']['find_cont'][1] + EXP2_DATA['set_id']['find_cont'][1]

    p_uncont = uncont_s / uncont_t
    p_cont = cont_s / cont_t

    predictions = {
        'singleton': {'find_uncont': p_uncont, 'find_cont': p_cont},
        'set_id': {'find_uncont': p_uncont, 'find_cont': p_cont}
    }
    nll = neg_log_lik_exp2(predictions)

    return {
        'p_uncont_goal': p_uncont,
        'p_cont_goal': p_cont,
        'nll': nll,
        'aic': aic(nll, k=2),  # k=2: p_uncont_goal, p_cont_goal
        'predictions': predictions
    }


# ============================================================================
# SINGLE ENTRY POINT
# ============================================================================

def fit_all_models() -> dict:
    """Fit all models for both experiments. Returns dict with all results."""
    return {
        'exp1': {
            'null': fit_null_exp1(),
            'goal_matching': fit_goal_matching_exp1(),
            'rsa': fit_rsa_exp1('singleton'),
        },
        'exp2': {
            'null': fit_null_exp2(),
            'goal_matching': fit_goal_matching_exp2(),
            'exhaustive_only': fit_rsa_exp2_ablation('set_id'),
            'mention_some_only': fit_rsa_exp2_ablation('singleton'),
            'full_rsa': fit_rsa_exp2_full(),
        }
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MODEL FITTING: Wh-Question Polarity Experiments")
    print("=" * 70)

    # ========== EXPERIMENT 1 ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Goal × Base Rate")
    print("=" * 70)

    print("\nFitting models...")
    exp1_null = fit_null_exp1()
    exp1_goal = fit_goal_matching_exp1()
    exp1_rsa = fit_rsa_exp1('singleton')

    print(f"\n{'Model':<20} {'k':>3} {'AIC':>8} {'Parameters'}")
    print("-" * 60)
    print(f"{'Null':<20} {0:>3} {exp1_null['aic']:>8.1f}")
    print(f"{'Goal-matching':<20} {2:>3} {exp1_goal['aic']:>8.1f} "
          f"p_uncont={exp1_goal['p_uncont_goal']:.2f}, p_cont={exp1_goal['p_cont_goal']:.2f}")
    print(f"{'Full RSA':<20} {2:>3} {exp1_rsa['aic']:>8.1f} "
          f"α={exp1_rsa['alpha']:.1f}, γ={exp1_rsa['gamma']:.2f}")

    # ========== EXPERIMENT 2 ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Goal × Decision Structure")
    print("=" * 70)

    print("\nFitting models...")
    exp2_null = fit_null_exp2()
    exp2_goal = fit_goal_matching_exp2()
    exp2_exh = fit_rsa_exp2_ablation('set_id')
    exp2_ms = fit_rsa_exp2_ablation('singleton')
    exp2_full = fit_rsa_exp2_full()

    print(f"\n{'Model':<25} {'k':>3} {'AIC':>8} {'Parameters'}")
    print("-" * 70)
    print(f"{'Null':<25} {0:>3} {exp2_null['aic']:>8.1f}")
    print(f"{'Goal-matching':<25} {2:>3} {exp2_goal['aic']:>8.1f} "
          f"p_uncont={exp2_goal['p_uncont_goal']:.2f}, p_cont={exp2_goal['p_cont_goal']:.2f}")
    print(f"{'Exhaustive-only RSA':<25} {2:>3} {exp2_exh['aic']:>8.1f} "
          f"α={exp2_exh['alpha']:.1f}, γ={exp2_exh['gamma']:.2f}")
    print(f"{'Mention-some only RSA':<25} {2:>3} {exp2_ms['aic']:>8.1f} "
          f"α={exp2_ms['alpha']:.1f}, γ={exp2_ms['gamma']:.2f}")
    print(f"{'Full RSA':<25} {2:>3} {exp2_full['aic']:>8.1f} "
          f"α={exp2_full['alpha']:.1f}, γ={exp2_full['gamma']:.2f}")

    # ========== PREDICTIONS ==========
    print("\n" + "=" * 70)
    print("FULL RSA MODEL PREDICTIONS (Exp 2)")
    print("=" * 70)

    preds = exp2_full['predictions']
    print(f"\nSingleton:")
    print(f"  find_uncont: {preds['singleton']['find_uncont']:.1%}")
    print(f"  find_cont:   {preds['singleton']['find_cont']:.1%}")
    print(f"  Effect:      {preds['singleton']['find_uncont'] - preds['singleton']['find_cont']:.1%}")

    print(f"\nSet ID:")
    print(f"  find_uncont: {preds['set_id']['find_uncont']:.1%}")
    print(f"  find_cont:   {preds['set_id']['find_cont']:.1%}")
    print(f"  Effect:      {preds['set_id']['find_uncont'] - preds['set_id']['find_cont']:.1%}")

    singleton_effect = preds['singleton']['find_uncont'] - preds['singleton']['find_cont']
    setid_effect = preds['set_id']['find_uncont'] - preds['set_id']['find_cont']
    print(f"\nInteraction: {singleton_effect - setid_effect:.1%}")

    # ========== DATA COMPARISON ==========
    print("\n" + "=" * 70)
    print("OBSERVED DATA (Exp 2)")
    print("=" * 70)

    obs_singleton_uncont = EXP2_DATA['singleton']['find_uncont'][0] / EXP2_DATA['singleton']['find_uncont'][1]
    obs_singleton_cont = EXP2_DATA['singleton']['find_cont'][0] / EXP2_DATA['singleton']['find_cont'][1]
    obs_setid_uncont = EXP2_DATA['set_id']['find_uncont'][0] / EXP2_DATA['set_id']['find_uncont'][1]
    obs_setid_cont = EXP2_DATA['set_id']['find_cont'][0] / EXP2_DATA['set_id']['find_cont'][1]

    print(f"\nSingleton:")
    print(f"  find_uncont: {obs_singleton_uncont:.1%}")
    print(f"  find_cont:   {obs_singleton_cont:.1%}")
    print(f"  Effect:      {obs_singleton_uncont - obs_singleton_cont:.1%}")

    print(f"\nSet ID:")
    print(f"  find_uncont: {obs_setid_uncont:.1%}")
    print(f"  find_cont:   {obs_setid_cont:.1%}")
    print(f"  Effect:      {obs_setid_uncont - obs_setid_cont:.1%}")

    obs_singleton_effect = obs_singleton_uncont - obs_singleton_cont
    obs_setid_effect = obs_setid_uncont - obs_setid_cont
    print(f"\nInteraction: {obs_singleton_effect - obs_setid_effect:.1%}")


if __name__ == "__main__":
    main()
