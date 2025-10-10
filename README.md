# Bias-Variance Playground

An interactive Plotly Dash web application that demonstrates how model performance changes when the number of features (p) approaches or exceeds the number of samples (n), and how regularization (Ridge/LASSO) stabilizes out-of-sample error.

[Link here](https://tomtranjr-bias-variance-playground.share.connect.posit.cloud/)

**Enhanced version** with preset scenarios, explanatory text, and results snapshots for educational use.

## Features

### Core Features
- **Interactive Controls**: Adjust number of samples (n), features (p), regularization strength (λ), model type, feature correlation (ρ), noise level (σ), and random seed
- **Real-time Visualization**: 
  - Train vs Test error metrics (RMSE and R²)
  - Error vs regularization strength curves with optimal λ annotations
  - Coefficient magnitude plots showing shrinkage/sparsity
  - Predicted vs actual scatter plots with R² annotations
- **Model Support**: OLS, Ridge, and LASSO regression
- **Edge Case Handling**: Proper warnings when p ≥ n for OLS models

### New Educational Features
- **Preset Scenarios**: 6 predefined scenarios with reproducible configurations
  - Comfortable OLS (n ≫ p): Stable estimation with plenty of data
  - Edge Case (p ≈ n): Compare OLS vs Ridge on same data
  - Underdetermined (p > n): LASSO feature selection
  - High Correlation: Compare Ridge vs LASSO with multicollinearity
- **Auto-Generated Explanations**: Data-driven summaries explaining bias-variance tradeoff, coefficient behavior, and data characteristics
- **Results Snapshot Panel**: Save and compare different configurations side-by-side
- **CSV Export**: Download snapshots for further analysis
- **Tooltips**: Helpful explanations for key parameters

## Installation

This project uses UV for dependency management. Make sure you have UV installed, then:

```bash
# Clone or download the project
cd bias-variance-playground

# Install dependencies
uv sync

# Run the app
uv run python app.py
```

The app will be available at `http://localhost:8050`

## Usage

### 1. Preset Scenarios
Select from 6 predefined scenarios to quickly explore different situations:
- **Comfortable OLS**: n=300, p=20 - demonstrates stable OLS estimation
- **Edge Case OLS**: n=120, p=100 - shows OLS instability when p ≈ n
- **Edge Case Ridge**: Same data as above but with Ridge regularization
- **Underdetermined LASSO**: n=80, p=200 - demonstrates automatic feature selection
- **High Correlation Ridge**: ρ=0.85 - shows Ridge handling multicollinearity
- **High Correlation LASSO**: Same data but with LASSO sparsity

### 2. Interactive Exploration
- Use sliders to adjust parameters and see real-time updates
- Hover over plots for detailed information
- Read auto-generated explanations in the "Explain This View" section
- Save interesting configurations using the snapshot panel

### 3. Results Analysis
- Compare different scenarios using the snapshot table
- Download results as CSV for further analysis
- Observe how regularization affects bias-variance tradeoff

## Key Educational Insights

- **Bias-Variance Tradeoff**: Higher regularization reduces variance but increases bias
- **Curse of Dimensionality**: When p approaches n, models become unstable without regularization
- **Feature Selection**: LASSO performs automatic feature selection by setting coefficients to zero
- **Regularization Benefits**: Ridge and LASSO provide stable solutions even when p > n
- **Multicollinearity**: High correlation between features requires regularization for stable estimates

## Preset Scenarios Details

| Scenario | n | p | Model | λ | ρ | σ | Key Learning |
|----------|---|---|------|---|---|---|--------------|
| Comfortable OLS | 300 | 20 | OLS | - | 0.2 | 1.0 | Stable estimation with n >> p |
| Edge Case OLS | 120 | 100 | OLS | - | 0.5 | 1.2 | OLS instability when p ≈ n |
| Edge Case Ridge | 120 | 100 | Ridge | 1.0 | 0.5 | 1.2 | Regularization stabilizes solution |
| Underdetermined LASSO | 80 | 200 | LASSO | 0.3 | 0.3 | 1.0 | Feature selection with p > n |
| High Correlation Ridge | 200 | 60 | Ridge | 3.0 | 0.85 | 1.0 | Ridge handles multicollinearity |
| High Correlation LASSO | 200 | 60 | LASSO | 0.15 | 0.85 | 1.0 | LASSO selects from correlated groups |

## Technical Details

- **Data Generation**: Synthetic data from linear model y = Xβ + ε with correlated features
- **Preprocessing**: Features standardized, response centered using training set statistics
- **Models**: scikit-learn implementations with proper hyperparameter handling
- **Visualization**: Plotly for interactive, publication-quality plots
- **UI**: Dash Bootstrap Components for responsive design
- **Reproducibility**: Fixed random seeds ensure identical results across runs

## Reproducibility

All preset scenarios use fixed random seeds to ensure reproducible results:
- Same data generation across runs
- Identical train/test splits
- Consistent coefficient estimates
- Comparable performance metrics
