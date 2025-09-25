#!/usr/bin/env python3
"""
Test script for the enhanced Bias-Variance Playground app
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from app import (generate_data, standardize_train_test, fit_and_metrics, 
                sweep_lambda, generate_explanation, create_snapshot_data, PRESET_SCENARIOS)

def test_preset_scenarios():
    """Test preset scenarios are properly defined"""
    print("Testing preset scenarios...")
    
    required_presets = [
        "Comfortable OLS (n ≫ p)",
        "Edge Case (p ≈ n) — OLS", 
        "Edge Case (p ≈ n) — Ridge",
        "Underdetermined (p > n) — LASSO",
        "High Correlation — Ridge",
        "High Correlation — LASSO"
    ]
    
    for preset_name in required_presets:
        assert preset_name in PRESET_SCENARIOS, f"Missing preset: {preset_name}"
        preset = PRESET_SCENARIOS[preset_name]
        
        # Check required fields
        required_fields = ['name', 'n', 'p', 'model', 'rho', 'sigma', 'seed', 'rationale']
        for field in required_fields:
            assert field in preset, f"Missing field {field} in preset {preset_name}"
        
        # Check model types
        assert preset['model'] in ['OLS', 'Ridge', 'LASSO'], f"Invalid model in {preset_name}"
        
        # Check ranges
        assert 30 <= preset['n'] <= 1000, f"Invalid n in {preset_name}"
        assert 2 <= preset['p'] <= 600, f"Invalid p in {preset_name}"
        assert 0.0 <= preset['rho'] <= 0.9, f"Invalid rho in {preset_name}"
        assert 0.1 <= preset['sigma'] <= 3.0, f"Invalid sigma in {preset_name}"
    
    print("✓ Preset scenarios test passed")

def test_reproducibility():
    """Test that presets produce reproducible results"""
    print("Testing reproducibility...")
    
    # Test Comfortable OLS preset
    preset = PRESET_SCENARIOS["Comfortable OLS (n ≫ p)"]
    
    # Generate data twice with same seed
    X1, y1, beta1 = generate_data(preset['n'], preset['p'], preset['rho'], preset['sigma'], preset['seed'])
    X2, y2, beta2 = generate_data(preset['n'], preset['p'], preset['rho'], preset['sigma'], preset['seed'])
    
    # Should be identical
    assert np.allclose(X1, X2), "Data generation not reproducible"
    assert np.allclose(y1, y2), "Response generation not reproducible"
    assert np.allclose(beta1, beta2), "True coefficients not reproducible"
    
    print("✓ Reproducibility test passed")

def test_explanations():
    """Test explanation generation"""
    print("Testing explanations...")
    
    # Generate test data
    X, y, beta_true = generate_data(100, 10, 0.5, 1.0, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(X_train, X_test, y_train, y_test)
    
    # Test with different models
    metrics_ridge = fit_and_metrics('Ridge', 1.0, X_train_s, y_train_c, X_test_s, y_test_c)
    metrics_lasso = fit_and_metrics('LASSO', 0.1, X_train_s, y_train_c, X_test_s, y_test_c)
    
    # Generate explanations
    explanations_ridge = generate_explanation(metrics_ridge, 'Ridge', 1.0, 100, 10, 0.5, 1.0, metrics_ridge['beta_hat'])
    explanations_lasso = generate_explanation(metrics_lasso, 'LASSO', 0.1, 100, 10, 0.5, 1.0, metrics_lasso['beta_hat'])
    
    # Check that explanations are generated
    assert len(explanations_ridge) > 0, "No explanations generated for Ridge"
    assert len(explanations_lasso) > 0, "No explanations generated for LASSO"
    
    # Check that explanations contain expected content
    ridge_text = ' '.join(explanations_ridge)
    lasso_text = ' '.join(explanations_lasso)
    
    assert 'Ridge' in ridge_text, "Ridge explanation missing model name"
    assert 'LASSO' in lasso_text, "LASSO explanation missing model name"
    
    print("✓ Explanations test passed")

def test_snapshots():
    """Test snapshot functionality"""
    print("Testing snapshots...")
    
    # Create test metrics
    test_metrics = {
        'rmse_train': 0.5,
        'rmse_test': 0.6,
        'r2_train': 0.8,
        'r2_test': 0.7
    }
    
    test_beta = np.array([1.0, 0.0, 0.5, 0.0, 0.2])  # Some zeros for LASSO
    
    test_config = {
        'n': 100,
        'p': 5,
        'model': 'LASSO',
        'lambda': 0.1,
        'rho': 0.3,
        'sigma': 1.0,
        'seed': 42
    }
    
    # Create snapshot
    snapshot = create_snapshot_data("Test Preset", test_config, test_metrics, test_beta, "2024-01-01 12:00:00")
    
    # Check snapshot structure
    required_fields = ['timestamp', 'preset', 'n', 'p', 'model', 'lambda', 'rho', 'sigma', 'seed', 
                      'train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'nonzero_coefs', 'p_n_ratio']
    
    for field in required_fields:
        assert field in snapshot, f"Missing field {field} in snapshot"
    
    # Check specific values
    assert snapshot['nonzero_coefs'] == 3, f"Expected 3 non-zero coefficients, got {snapshot['nonzero_coefs']}"
    assert snapshot['p_n_ratio'] == "0.05", f"Expected p/n ratio 0.05, got {snapshot['p_n_ratio']}"
    
    print("✓ Snapshots test passed")

def test_edge_cases():
    """Test edge case handling"""
    print("Testing edge cases...")
    
    # Test p > n case
    X, y, beta_true = generate_data(20, 50, 0.0, 1.0, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(X_train, X_test, y_train, y_test)
    
    # OLS should fail gracefully
    metrics_ols = fit_and_metrics('OLS', 0.0, X_train_s, y_train_c, X_test_s, y_test_c)
    # This might fail or succeed depending on numerical precision
    
    # Ridge should work
    metrics_ridge = fit_and_metrics('Ridge', 1.0, X_train_s, y_train_c, X_test_s, y_test_c)
    assert metrics_ridge['converged'], "Ridge should converge even when p > n"
    
    # LASSO should work
    metrics_lasso = fit_and_metrics('LASSO', 0.1, X_train_s, y_train_c, X_test_s, y_test_c)
    assert metrics_lasso['converged'], "LASSO should converge even when p > n"
    
    print("✓ Edge cases test passed")

def test_preset_consistency():
    """Test that preset configurations are consistent"""
    print("Testing preset consistency...")
    
    # Test edge case presets use same data
    ols_preset = PRESET_SCENARIOS["Edge Case (p ≈ n) — OLS"]
    ridge_preset = PRESET_SCENARIOS["Edge Case (p ≈ n) — Ridge"]
    
    # Should have same data parameters
    assert ols_preset['n'] == ridge_preset['n'], "Edge case presets should have same n"
    assert ols_preset['p'] == ridge_preset['p'], "Edge case presets should have same p"
    assert ols_preset['rho'] == ridge_preset['rho'], "Edge case presets should have same rho"
    assert ols_preset['sigma'] == ridge_preset['sigma'], "Edge case presets should have same sigma"
    assert ols_preset['seed'] == ridge_preset['seed'], "Edge case presets should have same seed"
    
    # Test high correlation presets use same data
    ridge_corr = PRESET_SCENARIOS["High Correlation — Ridge"]
    lasso_corr = PRESET_SCENARIOS["High Correlation — LASSO"]
    
    assert ridge_corr['n'] == lasso_corr['n'], "High correlation presets should have same n"
    assert ridge_corr['p'] == lasso_corr['p'], "High correlation presets should have same p"
    assert ridge_corr['rho'] == lasso_corr['rho'], "High correlation presets should have same rho"
    assert ridge_corr['sigma'] == lasso_corr['sigma'], "High correlation presets should have same sigma"
    assert ridge_corr['seed'] == lasso_corr['seed'], "High correlation presets should have same seed"
    
    print("✓ Preset consistency test passed")

if __name__ == "__main__":
    print("Running enhanced Bias-Variance Playground tests...")
    print("=" * 60)
    
    try:
        test_preset_scenarios()
        test_reproducibility()
        test_explanations()
        test_snapshots()
        test_edge_cases()
        test_preset_consistency()
        
        print("=" * 60)
        print("🎉 All enhanced features tests passed! The app is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
