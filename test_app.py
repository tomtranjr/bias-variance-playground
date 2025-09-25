#!/usr/bin/env python3
"""
Test script for the Bias-Variance Playground app
"""

import sys
import numpy as np
from app import generate_data, standardize_train_test, fit_and_metrics, sweep_lambda

def test_data_generation():
    """Test data generation functions"""
    print("Testing data generation...")
    
    # Test basic data generation
    X, y, beta_true = generate_data(n=100, p=10, rho=0.5, sigma=1.0, seed=42)
    
    assert X.shape == (100, 10), f"Expected X shape (100, 10), got {X.shape}"
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    assert beta_true.shape == (10,), f"Expected beta_true shape (10,), got {beta_true.shape}"
    
    # Check that first 5 coefficients are non-zero
    assert np.sum(np.abs(beta_true[:5]) > 0) == 5, "First 5 coefficients should be non-zero"
    assert np.sum(np.abs(beta_true[5:]) > 0) == 0, "Remaining coefficients should be zero"
    
    print("✓ Data generation test passed")

def test_standardization():
    """Test standardization functions"""
    print("Testing standardization...")
    
    X, y, beta_true = generate_data(n=100, p=10, rho=0.0, sigma=1.0, seed=42)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(
        X_train, X_test, y_train, y_test
    )
    
    # Check standardization
    assert np.allclose(np.mean(X_train_s, axis=0), 0, atol=1e-10), "Training features should have mean 0"
    assert np.allclose(np.std(X_train_s, axis=0), 1, atol=1e-10), "Training features should have std 1"
    assert np.allclose(np.mean(y_train_c), 0, atol=1e-10), "Training response should have mean 0"
    
    print("✓ Standardization test passed")

def test_model_fitting():
    """Test model fitting functions"""
    print("Testing model fitting...")
    
    X, y, beta_true = generate_data(n=100, p=10, rho=0.0, sigma=1.0, seed=42)
    
    # Split and standardize
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(
        X_train, X_test, y_train, y_test
    )
    
    # Test OLS
    metrics_ols = fit_and_metrics('OLS', 0.0, X_train_s, y_train_c, X_test_s, y_test_c)
    assert metrics_ols['converged'], "OLS should converge"
    assert not np.isnan(metrics_ols['rmse_train']), "OLS should have valid train RMSE"
    
    # Test Ridge
    metrics_ridge = fit_and_metrics('Ridge', 1.0, X_train_s, y_train_c, X_test_s, y_test_c)
    assert metrics_ridge['converged'], "Ridge should converge"
    assert not np.isnan(metrics_ridge['rmse_train']), "Ridge should have valid train RMSE"
    
    # Test LASSO
    metrics_lasso = fit_and_metrics('LASSO', 0.1, X_train_s, y_train_c, X_test_s, y_test_c)
    assert metrics_lasso['converged'], "LASSO should converge"
    assert not np.isnan(metrics_lasso['rmse_train']), "LASSO should have valid train RMSE"
    
    print("✓ Model fitting test passed")

def test_lambda_sweep():
    """Test lambda sweep function"""
    print("Testing lambda sweep...")
    
    X, y, beta_true = generate_data(n=100, p=10, rho=0.0, sigma=1.0, seed=42)
    
    # Split and standardize
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(
        X_train, X_test, y_train, y_test
    )
    
    # Test Ridge sweep
    lambdas = np.logspace(-2, 2, 10)
    error_df = sweep_lambda('Ridge', lambdas, X_train_s, y_train_c, X_test_s, y_test_c)
    
    assert len(error_df) == 10, f"Expected 10 rows, got {len(error_df)}"
    assert 'lambda' in error_df.columns, "Missing lambda column"
    assert 'rmse_train' in error_df.columns, "Missing rmse_train column"
    assert 'rmse_test' in error_df.columns, "Missing rmse_test column"
    
    print("✓ Lambda sweep test passed")

def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    # Test p > n case
    X, y, beta_true = generate_data(n=20, p=50, rho=0.0, sigma=1.0, seed=42)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(
        X_train, X_test, y_train, y_test
    )
    
    # OLS should fail gracefully
    metrics_ols = fit_and_metrics('OLS', 0.0, X_train_s, y_train_c, X_test_s, y_test_c)
    # This might fail or succeed depending on numerical precision
    
    # Ridge should work
    metrics_ridge = fit_and_metrics('Ridge', 1.0, X_train_s, y_train_c, X_test_s, y_test_c)
    assert metrics_ridge['converged'], "Ridge should converge even when p > n"
    
    print("✓ Edge cases test passed")

if __name__ == "__main__":
    print("Running Bias-Variance Playground tests...")
    print("=" * 50)
    
    try:
        test_data_generation()
        test_standardization()
        test_model_fitting()
        test_lambda_sweep()
        test_edge_cases()
        
        print("=" * 50)
        print("🎉 All tests passed! The app is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
