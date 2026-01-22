"""
Test suite for the symmetric wh-questions model.

The symmetric model exploits exchangeability of vials to reduce state space
from O(2^N) to O(N), enabling efficient computation at larger N values.
"""

import subprocess
import json
import sys
import time

# Tolerance for numerical comparison (percentage points)
TOLERANCE = 0.5


def run_model(n_vials, contamination_rate, gamma=0.9, decision_type='singleton'):
    """Run the symmetric model and return the results dict."""
    result = subprocess.run(
        ['python3', 'wh-questions.py',
         '-n', str(n_vials),
         '-r', str(contamination_rate),
         '-d', decision_type,
         '-g', str(gamma)],
        capture_output=True,
        text=True,
        timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Model failed: {result.stderr}")

    # Parse JSON from stdout (may have stderr mixed in)
    for line in result.stdout.strip().split('\n'):
        if line.startswith('{'):
            return json.loads(line)
    raise RuntimeError(f"No JSON output found: {result.stdout}")


def test_runs_at_various_n():
    """Test that model runs correctly at various N values."""
    print("Testing model runs at various N values:")
    results = []

    for n in [3, 4, 5, 6, 10]:
        print(f"  N={n}...", end=" ")
        start = time.time()
        try:
            result = run_model(n, 0.5, 0.9)
            elapsed = time.time() - start

            # Check basic validity
            valid = (
                'find_uncontam' in result and
                'find_contam' in result and
                0 <= result['find_uncontam']['p_which_uncont'] <= 1 and
                0 <= result['find_contam']['p_which_cont'] <= 1
            )

            if valid:
                print(f"PASS ({elapsed:.1f}s)")
                results.append(True)
            else:
                print(f"FAIL (invalid output)")
                results.append(False)
        except Exception as e:
            print(f"FAIL ({e})")
            results.append(False)

    return all(results)


def test_extreme_gamma_values():
    """Test model at extreme gamma values."""
    print("Testing extreme gamma values:")
    results = []

    for gamma in [0.5, 0.6, 0.7, 0.9, 0.99]:
        print(f"  gamma={gamma}...", end=" ")
        try:
            result = run_model(5, 0.5, gamma)

            # At gamma=0.5 (no speaker knowledge advantage), the model should
            # still show goal effect due to decision problem structure
            valid = (
                'find_uncontam' in result and
                result['find_uncontam']['p_which_uncont'] > 0.5
            )

            if valid:
                print(f"PASS (p_which_uncont={result['find_uncontam']['p_which_uncont']*100:.1f}%)")
                results.append(True)
            else:
                print(f"FAIL")
                results.append(False)
        except Exception as e:
            print(f"FAIL ({e})")
            results.append(False)

    return all(results)


def test_extreme_base_rates():
    """Test model at extreme base rates."""
    print("Testing extreme base rates:")
    results = []

    for rate in [0.1, 0.2, 0.5, 0.8, 0.9]:
        print(f"  rate={rate}...", end=" ")
        try:
            result = run_model(5, rate, 0.9)

            # Check validity
            valid = (
                'find_uncontam' in result and
                0 <= result['find_uncontam']['p_which_uncont'] <= 1
            )

            if valid:
                print(f"PASS (p_which_uncont={result['find_uncontam']['p_which_uncont']*100:.1f}%)")
                results.append(True)
            else:
                print(f"FAIL")
                results.append(False)
        except Exception as e:
            print(f"FAIL ({e})")
            results.append(False)

    return all(results)


def test_symmetry():
    """Test that find_uncont @ rate p equals find_cont @ rate 1-p."""
    print("Testing symmetry (find_uncont@p == find_cont@(1-p)):")
    results = []

    for rate in [0.2, 0.3, 0.5, 0.7, 0.8]:
        print(f"  rate={rate} vs {1-rate}...", end=" ")
        try:
            result_p = run_model(5, rate, 0.9)
            result_1p = run_model(5, 1 - rate, 0.9)

            # find_uncont @ rate p should equal find_cont @ rate 1-p
            p_uncont_at_p = result_p['find_uncontam']['p_which_uncont']
            p_cont_at_1p = result_1p['find_contam']['p_which_cont']

            diff = abs(p_uncont_at_p - p_cont_at_1p) * 100

            if diff <= TOLERANCE:
                print(f"PASS (diff={diff:.2f} pts)")
                results.append(True)
            else:
                print(f"FAIL (diff={diff:.2f} pts)")
                print(f"    find_uncont@{rate}: {p_uncont_at_p*100:.2f}%")
                print(f"    find_cont@{1-rate}: {p_cont_at_1p*100:.2f}%")
                results.append(False)
        except Exception as e:
            print(f"FAIL ({e})")
            results.append(False)

    return all(results)


def test_goal_effect():
    """Test that the model shows expected goal effect at 0.5 base rate."""
    print("Testing goal effect at base rate 0.5...", end=" ")

    result = run_model(10, 0.5, 0.9)

    # At 0.5 base rate, should prefer goal-matching question
    find_uncont_prefers_which_uncont = result['find_uncontam']['p_which_uncont'] > 0.6
    find_cont_prefers_which_cont = result['find_contam']['p_which_cont'] > 0.6

    passed = find_uncont_prefers_which_uncont and find_cont_prefers_which_cont

    if passed:
        print(f"PASS (goal effect: {result['find_uncontam']['p_which_uncont']*100:.1f}%)")
    else:
        print(f"FAIL")
        print(f"  find_uncontam prefers which_uncont: {result['find_uncontam']['p_which_uncont']*100:.1f}%")
        print(f"  find_contam prefers which_cont: {result['find_contam']['p_which_cont']*100:.1f}%")

    return passed


def test_base_rate_effect():
    """Test that the model shows base rate modulation."""
    print("Testing base rate effect...", end=" ")

    r02 = run_model(10, 0.2, 0.9)
    r08 = run_model(10, 0.8, 0.9)

    # At 0.2 (few contaminated), FIND_UNCONT should prefer WHICH_CONT more
    # At 0.8 (many contaminated), FIND_UNCONT should strongly prefer WHICH_UNCONT
    p_which_uncont_at_02 = r02['find_uncontam']['p_which_uncont']
    p_which_uncont_at_08 = r08['find_uncontam']['p_which_uncont']

    # Should see higher preference for goal-matching at extreme base rates
    passed = p_which_uncont_at_08 > p_which_uncont_at_02

    if passed:
        print(f"PASS (0.2: {p_which_uncont_at_02*100:.1f}%, 0.8: {p_which_uncont_at_08*100:.1f}%)")
    else:
        print(f"FAIL")
        print(f"  base_rate=0.2: {p_which_uncont_at_02*100:.1f}% prefer which_uncont")
        print(f"  base_rate=0.8: {p_which_uncont_at_08*100:.1f}% prefer which_uncont")

    return passed


def test_efficiency_at_n10():
    """Test that model runs efficiently at N=10."""
    print("Testing efficiency at N=10...", end=" ")

    start = time.time()
    result = run_model(10, 0.5, 0.9)
    elapsed = time.time() - start

    # Should complete in under 30 seconds
    passed = elapsed < 30 and 'find_uncontam' in result

    if passed:
        print(f"PASS ({elapsed:.1f}s)")
    else:
        print(f"FAIL (took {elapsed:.1f}s)")

    return passed


def test_set_id_runs():
    """Test that SET_ID decision structure runs correctly."""
    print("Testing SET_ID decision structure:")
    results = []

    for n in [5, 10]:
        print(f"  N={n}...", end=" ")
        start = time.time()
        try:
            result = run_model(n, 0.5, 0.9, decision_type='set_id')
            elapsed = time.time() - start

            # Check basic validity
            valid = (
                'find_uncontam' in result and
                'find_contam' in result and
                0 <= result['find_uncontam']['p_which_uncont'] <= 1 and
                0 <= result['find_contam']['p_which_cont'] <= 1
            )

            if valid:
                print(f"PASS ({elapsed:.1f}s, goal_effect={result['find_uncontam']['p_which_uncont']*100:.1f}%)")
                results.append(True)
            else:
                print(f"FAIL (invalid output)")
                results.append(False)
        except Exception as e:
            print(f"FAIL ({e})")
            results.append(False)

    return all(results)


def test_set_id_weaker_goal_effect():
    """Test that SET_ID shows weaker goal effect than SINGLETON."""
    print("Testing SET_ID has weaker goal effect than SINGLETON...", end=" ")

    singleton = run_model(10, 0.5, 0.9, decision_type='singleton')
    set_id = run_model(10, 0.5, 0.9, decision_type='set_id')

    singleton_effect = singleton['find_uncontam']['p_which_uncont']
    set_id_effect = set_id['find_uncontam']['p_which_uncont']

    # SET_ID should have weaker goal alignment (closer to 0.5)
    passed = singleton_effect > set_id_effect and set_id_effect > 0.5

    if passed:
        print(f"PASS (singleton={singleton_effect*100:.1f}%, set_id={set_id_effect*100:.1f}%)")
    else:
        print(f"FAIL")
        print(f"  singleton goal effect: {singleton_effect*100:.1f}%")
        print(f"  set_id goal effect: {set_id_effect*100:.1f}%")

    return passed


def test_set_id_symmetry():
    """Test that SET_ID preserves symmetry property."""
    print("Testing SET_ID symmetry...", end=" ")

    result_p = run_model(5, 0.3, 0.9, decision_type='set_id')
    result_1p = run_model(5, 0.7, 0.9, decision_type='set_id')

    p_uncont = result_p['find_uncontam']['p_which_uncont']
    p_cont = result_1p['find_contam']['p_which_cont']

    diff = abs(p_uncont - p_cont) * 100

    if diff <= TOLERANCE:
        print(f"PASS (diff={diff:.2f} pts)")
        return True
    else:
        print(f"FAIL (diff={diff:.2f} pts)")
        return False


def main():
    print("=" * 60)
    print("Symmetric Wh-Questions Model Test Suite")
    print("=" * 60)
    print()

    results = []

    results.append(("Various N values", test_runs_at_various_n()))

    print()
    results.append(("Extreme gamma values", test_extreme_gamma_values()))

    print()
    results.append(("Extreme base rates", test_extreme_base_rates()))

    print()
    results.append(("Symmetry property", test_symmetry()))

    print()
    print("Qualitative behavior tests (SINGLETON):")
    results.append(("Goal effect", test_goal_effect()))
    results.append(("Base rate effect", test_base_rate_effect()))
    results.append(("Efficiency at N=10", test_efficiency_at_n10()))

    print()
    print("SET_ID decision structure tests:")
    results.append(("SET_ID runs", test_set_id_runs()))
    results.append(("SET_ID weaker goal effect", test_set_id_weaker_goal_effect()))
    results.append(("SET_ID symmetry", test_set_id_symmetry()))

    print()
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Results: {passed}/{total} test categories passed")

    if passed < total:
        print("\nFailed tests:")
        for name, r in results:
            if not r:
                print(f"  - {name}")

    print("=" * 60)

    return 0 if all(r for _, r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
