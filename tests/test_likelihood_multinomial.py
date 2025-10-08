#!/usr/bin/env python3
"""
Unit tests for likelihood_multinomial() function

This test suite verifies:
1. Backward compatibility with existing behavior
2. Proper handling of NaN values in P_obs
3. Functionality with various parameters (class_id, entropyFilter, etc.)
"""

import numpy as np
import sys
import os

# Add integrate module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrate.integrate_rejection import likelihood_multinomial


def test_basic_example():
    """Test basic example from docstring - ensures backward compatibility"""
    print("\n" + "="*60)
    print("Test 1: Basic Example (from docstring)")
    print("="*60)

    D = np.array([[1, 2], [2, 1]])
    P_obs = np.array([[0.3, 0.7], [0.7, 0.3]])

    print(f"D shape: {D.shape}")
    print(f"D:\n{D}")
    print(f"P_obs shape: {P_obs.shape}")
    print(f"P_obs:\n{P_obs}")

    logL = likelihood_multinomial(D, P_obs)

    print(f"logL: {logL}")
    print(f"All finite? {np.all(np.isfinite(logL))}")

    # Expected: Should return finite log-likelihood values
    assert np.all(np.isfinite(logL)), "FAILED: logL contains NaN or Inf"
    print("✓ PASSED: Returns finite logL values")

    return logL


def test_nan_columns():
    """Test with some NaN columns in P_obs"""
    print("\n" + "="*60)
    print("Test 2: P_obs with NaN Columns")
    print("="*60)

    D = np.array([[1, 2, 1], [2, 1, 2]])
    P_obs = np.array([[0.3, np.nan, 0.5],
                      [0.7, np.nan, 0.5]])

    print(f"D shape: {D.shape}")
    print(f"D:\n{D}")
    print(f"P_obs shape: {P_obs.shape}")
    print(f"P_obs:\n{P_obs}")
    print(f"NaN columns: {np.where(np.any(np.isnan(P_obs), axis=0))[0]}")

    logL = likelihood_multinomial(D, P_obs)

    print(f"logL: {logL}")
    print(f"All finite? {np.all(np.isfinite(logL))}")

    # After fix: Should filter out NaN column and compute on columns 0 and 2
    if np.all(np.isfinite(logL)):
        print("✓ PASSED: Correctly handles NaN columns (returns finite logL)")
    else:
        print("✗ FAILED: Returns NaN (BUG - needs fix)")

    return logL


def test_all_nan():
    """Test with all NaN in P_obs"""
    print("\n" + "="*60)
    print("Test 3: P_obs All NaN")
    print("="*60)

    D = np.array([[1, 2], [2, 1]])
    P_obs = np.array([[np.nan, np.nan],
                      [np.nan, np.nan]])

    print(f"D shape: {D.shape}")
    print(f"P_obs (all NaN):\n{P_obs}")

    logL = likelihood_multinomial(D, P_obs)

    print(f"logL: {logL}")
    print(f"All NaN? {np.all(np.isnan(logL))}")

    # Expected: Should return NaN (no valid features)
    if np.all(np.isnan(logL)):
        print("✓ PASSED: Correctly returns NaN when no valid features")
    else:
        print("✗ FAILED: Should return NaN when all features are NaN")

    return logL


def test_with_class_id():
    """Test with explicit class_id parameter"""
    print("\n" + "="*60)
    print("Test 4: With class_id Parameter")
    print("="*60)

    D = np.array([[1, 2], [2, 1]])
    P_obs = np.array([[0.3, 0.7], [0.7, 0.3]])
    class_id = np.array([1, 2])

    print(f"D:\n{D}")
    print(f"P_obs:\n{P_obs}")
    print(f"class_id: {class_id}")

    logL = likelihood_multinomial(D, P_obs, class_id=class_id)

    print(f"logL: {logL}")
    print(f"All finite? {np.all(np.isfinite(logL))}")

    assert np.all(np.isfinite(logL)), "FAILED: logL contains NaN or Inf"
    print("✓ PASSED: Works with class_id parameter")

    return logL


def test_with_entropy_filter():
    """Test with entropyFilter enabled"""
    print("\n" + "="*60)
    print("Test 5: With entropyFilter")
    print("="*60)

    # Create data with varying entropy
    D = np.array([[1, 2, 1, 1], [2, 1, 2, 1]])
    P_obs = np.array([[0.9, 0.5, 0.8, 0.5],  # High certainty, medium, high, medium
                      [0.1, 0.5, 0.2, 0.5]])  # Low entropy, high entropy, low, high

    print(f"D:\n{D}")
    print(f"P_obs:\n{P_obs}")

    # Compute entropy for each feature
    from scipy.stats import entropy
    H = entropy(P_obs.T)
    print(f"Entropy per feature: {H}")

    logL = likelihood_multinomial(D, P_obs, entropyFilter=True, entropyThreshold=0.5)

    print(f"logL: {logL}")
    print(f"All finite? {np.all(np.isfinite(logL))}")

    assert np.all(np.isfinite(logL)), "FAILED: logL contains NaN or Inf"
    print("✓ PASSED: Works with entropyFilter")

    return logL


def test_nan_with_entropy_filter():
    """Test combined NaN filtering and entropy filtering"""
    print("\n" + "="*60)
    print("Test 6: NaN Columns + entropyFilter")
    print("="*60)

    D = np.array([[1, 2, 1, 2], [2, 1, 2, 1]])
    P_obs = np.array([[0.9, np.nan, 0.5, 0.8],  # Column 1 has NaN
                      [0.1, np.nan, 0.5, 0.2]])

    print(f"D:\n{D}")
    print(f"P_obs:\n{P_obs}")
    print(f"NaN columns: {np.where(np.any(np.isnan(P_obs), axis=0))[0]}")

    logL = likelihood_multinomial(D, P_obs, entropyFilter=True, entropyThreshold=0.5)

    print(f"logL: {logL}")
    print(f"All finite? {np.all(np.isfinite(logL))}")

    if np.all(np.isfinite(logL)):
        print("✓ PASSED: Handles both NaN filtering and entropy filtering")
    else:
        print("✗ FAILED: Cannot handle combined filtering (needs fix)")

    return logL


def run_all_tests():
    """Run all tests and summarize results"""
    print("\n" + "#"*60)
    print("# Testing likelihood_multinomial() Function")
    print("#"*60)

    results = {}

    try:
        results['test1'] = test_basic_example()
    except Exception as e:
        print(f"✗ Test 1 FAILED with exception: {e}")
        results['test1'] = None

    try:
        results['test2'] = test_nan_columns()
    except Exception as e:
        print(f"✗ Test 2 FAILED with exception: {e}")
        results['test2'] = None

    try:
        results['test3'] = test_all_nan()
    except Exception as e:
        print(f"✗ Test 3 FAILED with exception: {e}")
        results['test3'] = None

    try:
        results['test4'] = test_with_class_id()
    except Exception as e:
        print(f"✗ Test 4 FAILED with exception: {e}")
        results['test4'] = None

    try:
        results['test5'] = test_with_entropy_filter()
    except Exception as e:
        print(f"✗ Test 5 FAILED with exception: {e}")
        results['test5'] = None

    try:
        results['test6'] = test_nan_with_entropy_filter()
    except Exception as e:
        print(f"✗ Test 6 FAILED with exception: {e}")
        results['test6'] = None

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)

    test_names = [
        "Basic example (docstring)",
        "P_obs with NaN columns",
        "P_obs all NaN",
        "With class_id parameter",
        "With entropyFilter",
        "NaN + entropyFilter combined"
    ]

    for i, (key, result) in enumerate(results.items(), 1):
        if result is None:
            status = "EXCEPTION"
        elif np.all(np.isnan(result)):
            status = "NaN" if i == 3 else "FAIL (NaN)"
        elif np.all(np.isfinite(result)):
            status = "PASS"
        else:
            status = "FAIL"

        print(f"Test {i} ({test_names[i-1]}): {status}")

    return results


if __name__ == '__main__':
    results = run_all_tests()
