"""
Bias-Variance Playground: Interactive demonstration of regularization effects
in linear models when p approaches or exceeds n.

Enhanced version with preset scenarios, explanatory text, and results snapshots.
"""

import base64
import io
import json
import os
import warnings
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from plotly.subplots import make_subplots
from scipy.linalg import cholesky
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Bias-Variance Playground"

# Global variables for caching
_lambda_grid = None
_error_cache = {}
_snapshots = []

# Preset scenarios with exact configurations
PRESET_SCENARIOS = {
    "Custom": None,
    "Comfortable OLS (n ≫ p)": {
        "name": "Comfortable OLS",
        "n": 300,
        "p": 20,
        "model": "OLS",
        "lambda": None,
        "rho": 0.2,
        "sigma": 1.0,
        "seed": 123,
        "rationale": "Here n >> p (300 >> 20), so OLS has plenty of data to estimate coefficients stably. The low correlation (ρ=0.2) and moderate noise (σ=1.0) make this a 'comfortable' scenario where OLS performs well."
    },
    "Edge Case (p ≈ n) — OLS": {
        "name": "Edge (p≈n) OLS",
        "n": 120,
        "p": 100,
        "model": "OLS",
        "lambda": None,
        "rho": 0.5,
        "sigma": 1.2,
        "seed": 456,
        "rationale": "This is the 'edge case' where p ≈ n (100 ≈ 120). OLS becomes unstable because XᵀX is nearly singular. Notice how test error is much higher than train error, indicating overfitting."
    },
    "Edge Case (p ≈ n) — Ridge": {
        "name": "Edge (p≈n) Ridge",
        "n": 120,
        "p": 100,
        "model": "Ridge",
        "lambda": 1.0,
        "rho": 0.5,
        "sigma": 1.2,
        "seed": 456,
        "rationale": "Same data as above but with Ridge regularization (λ=1.0). Ridge stabilizes the solution by adding penalty to large coefficients, reducing overfitting and improving test performance."
    },
    "Underdetermined (p > n) — LASSO": {
        "name": "Underdetermined LASSO",
        "n": 80,
        "p": 200,
        "model": "LASSO",
        "lambda": 0.3,
        "rho": 0.3,
        "sigma": 1.0,
        "seed": 789,
        "rationale": "Here p > n (200 > 80), making the system underdetermined. LASSO performs automatic feature selection by setting many coefficients to exactly zero, finding a sparse solution."
    },
    "High Correlation — Ridge": {
        "name": "High Correlation Ridge",
        "n": 200,
        "p": 60,
        "model": "Ridge",
        "lambda": 3.0,
        "rho": 0.85,
        "sigma": 1.0,
        "seed": 31415,
        "rationale": "Strong correlation (ρ=0.85) between features creates multicollinearity. Ridge regularization (λ=3.0) shrinks correlated coefficients together, stabilizing the solution."
    },
    "High Correlation — LASSO": {
        "name": "High Correlation LASSO",
        "n": 200,
        "p": 60,
        "model": "LASSO",
        "lambda": 0.15,
        "rho": 0.85,
        "sigma": 1.0,
        "seed": 31415,
        "rationale": "Same high correlation scenario but with LASSO (λ=0.15). LASSO tends to select one feature from each correlated group, creating a sparse model."
    }
}

def make_covariance(p, rho):
    """
    Create covariance matrix with correlation structure Σ_ij = ρ^|i-j|
    
    Args:
        p: number of features
        rho: correlation parameter (0 <= rho < 1)
    
    Returns:
        (p, p) covariance matrix
    """
    indices = np.arange(p)
    cov_matrix = np.zeros((p, p))
    
    for i in range(p):
        for j in range(p):
            cov_matrix[i, j] = rho ** abs(i - j)
    
    return cov_matrix

def generate_data(n, p, rho, sigma, seed):
    """
    Generate synthetic data from linear model y = Xβ + ε
    
    Args:
        n: number of samples
        p: number of features
        rho: correlation parameter for features
        sigma: noise standard deviation
        seed: random seed
    
    Returns:
        X: (n, p) design matrix
        y: (n,) response vector
        beta_true: (p,) true coefficients
    """
    np.random.seed(seed)
    
    # Generate correlated features
    if rho > 0:
        cov_matrix = make_covariance(p, rho)
        # Use Cholesky decomposition for numerical stability
        try:
            L = cholesky(cov_matrix, lower=True)
            X = np.random.randn(n, p) @ L.T
        except np.linalg.LinAlgError:
            # Fallback to identity if Cholesky fails
            X = np.random.randn(n, p)
    else:
        X = np.random.randn(n, p)
    
    # Generate sparse true coefficients
    k_true = min(5, p)
    beta_true = np.zeros(p)
    beta_true[:k_true] = np.random.randn(k_true)
    
    # Generate response with noise
    y = X @ beta_true + sigma * np.random.randn(n)
    
    return X, y, beta_true

def standardize_train_test(X_train, X_test, y_train, y_test):
    """
    Standardize features and center response using training set statistics
    
    Args:
        X_train, X_test: training and test feature matrices
        y_train, y_test: training and test response vectors
    
    Returns:
        Standardized matrices and centered responses
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Center response
    y_mean = np.mean(y_train)
    y_train_c = y_train - y_mean
    y_test_c = y_test - y_mean
    
    return X_train_s, X_test_s, y_train_c, y_test_c, y_mean

def fit_and_metrics(model_name, lambda_value, X_train, y_train, X_test, y_test):
    """
    Fit model and compute metrics
    
    Args:
        model_name: 'OLS', 'Ridge', or 'LASSO'
        lambda_value: regularization parameter
        X_train, y_train: training data
        X_test, y_test: test data
    
    Returns:
        Dictionary with metrics and predictions
    """
    # Initialize model
    if model_name == 'OLS':
        model = LinearRegression(fit_intercept=False)
    elif model_name == 'Ridge':
        model = Ridge(alpha=lambda_value, fit_intercept=False, solver='auto', random_state=42)
    elif model_name == 'LASSO':
        model = Lasso(alpha=lambda_value, fit_intercept=False, max_iter=20000, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Fit model
    try:
        model.fit(X_train, y_train)
        beta_hat = model.coef_
        
        # Predictions
        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)
        
        # Metrics
        rmse_train = np.sqrt(mean_squared_error(y_train, yhat_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, yhat_test))
        r2_train = r2_score(y_train, yhat_train)
        r2_test = r2_score(y_test, yhat_test)
        
        return {
            'beta_hat': beta_hat,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'yhat_test': yhat_test,
            'converged': True
        }
    
    except Exception as e:
        return {
            'beta_hat': np.zeros(X_train.shape[1]),
            'rmse_train': np.nan,
            'rmse_test': np.nan,
            'r2_train': np.nan,
            'r2_test': np.nan,
            'yhat_test': np.zeros(len(y_test)),
            'converged': False,
            'error': str(e)
        }

def sweep_lambda(model_name, lambdas, X_train, y_train, X_test, y_test):
    """
    Compute error metrics across a grid of lambda values
    
    Args:
        model_name: 'Ridge' or 'LASSO'
        lambdas: array of lambda values
        X_train, y_train: training data
        X_test, y_test: test data
    
    Returns:
        DataFrame with lambda, log10_lambda, rmse_train, rmse_test
    """
    results = []
    
    for lam in lambdas:
        metrics = fit_and_metrics(model_name, lam, X_train, y_train, X_test, y_test)
        results.append({
            'lambda': lam,
            'log10_lambda': np.log10(lam),
            'rmse_train': metrics['rmse_train'],
            'rmse_test': metrics['rmse_test']
        })
    
    return pd.DataFrame(results)

def create_lambda_grid():
    """Create the standard lambda grid for error curves"""
    global _lambda_grid
    if _lambda_grid is None:
        _lambda_grid = np.logspace(-4, 3, 50)
    return _lambda_grid

def generate_explanation(metrics, model, lambda_value, n, p, rho, sigma, beta_hat):
    """Generate explanatory text based on current state"""
    
    # Bias-Variance analysis
    train_rmse = metrics['rmse_train']
    test_rmse = metrics['rmse_test']
    train_r2 = metrics['r2_train']
    test_r2 = metrics['r2_test']
    
    # Coefficient analysis
    n_nonzero = np.sum(np.abs(beta_hat) > 1e-6)
    max_coef = np.max(np.abs(beta_hat))
    
    explanations = []
    
    # Bias-Variance summary
    if not np.isnan(train_rmse) and not np.isnan(test_rmse):
        gap = test_rmse - train_rmse
        if gap > 0.1:
            explanations.append(f"📊 **Overfitting detected**: Test RMSE ({test_rmse:.3f}) is {gap:.3f} higher than train RMSE ({train_rmse:.3f}). This suggests high variance.")
        elif gap < -0.05:
            explanations.append(f"📊 **Underfitting detected**: Train RMSE ({train_rmse:.3f}) is higher than test RMSE ({test_rmse:.3f}). This suggests high bias.")
        else:
            explanations.append(f"📊 **Good generalization**: Train and test RMSE are similar ({train_rmse:.3f} vs {test_rmse:.3f}), indicating good bias-variance balance.")
    
    # Coefficient story
    if model == 'LASSO':
        explanations.append(f"🔍 **LASSO sparsity**: {n_nonzero}/{p} coefficients are non-zero (sparsity: {(p-n_nonzero)/p*100:.1f}%). Max coefficient: {max_coef:.3f}")
    elif model == 'Ridge':
        explanations.append(f"📏 **Ridge shrinkage**: All {p} coefficients are non-zero but shrunk. Max coefficient: {max_coef:.3f} (λ={lambda_value:.3f})")
    else:
        explanations.append(f"📐 **OLS coefficients**: All {p} coefficients estimated. Max coefficient: {max_coef:.3f}")
    
    # Data characteristics
    if p >= n:
        explanations.append(f"⚠️ **Underdetermined system**: p={p} ≥ n={n}. OLS is ill-posed; regularization is essential.")
    elif p > 0.8 * n:
        explanations.append(f"⚖️ **Edge case**: p={p} ≈ n={n} (ratio: {p/n:.2f}). System is near-singular.")
    else:
        explanations.append(f"✅ **Well-determined**: n={n} >> p={p} (ratio: {n/p:.1f}). Plenty of data for stable estimation.")
    
    # Correlation effects
    if rho > 0.7:
        explanations.append(f"🔗 **High correlation**: ρ={rho:.2f} creates multicollinearity. Regularization helps stabilize estimates.")
    elif rho > 0.3:
        explanations.append(f"🔗 **Moderate correlation**: ρ={rho:.2f} between features.")
    else:
        explanations.append(f"🔗 **Low correlation**: ρ={rho:.2f} between features.")
    
    return explanations

def create_snapshot_data(preset_name, config, metrics, beta_hat, timestamp):
    """Create a snapshot data entry"""
    n_nonzero = np.sum(np.abs(beta_hat) > 1e-6) if beta_hat is not None else 0
    
    return {
        'timestamp': timestamp,
        'preset': preset_name,
        'n': config['n'],
        'p': config['p'],
        'model': config['model'],
        'lambda': config.get('lambda', 'N/A'),
        'rho': config['rho'],
        'sigma': config['sigma'],
        'seed': config['seed'],
        'train_rmse': f"{metrics['rmse_train']:.3f}" if not np.isnan(metrics['rmse_train']) else "N/A",
        'test_rmse': f"{metrics['rmse_test']:.3f}" if not np.isnan(metrics['rmse_test']) else "N/A",
        'train_r2': f"{metrics['r2_train']:.3f}" if not np.isnan(metrics['r2_train']) else "N/A",
        'test_r2': f"{metrics['r2_test']:.3f}" if not np.isnan(metrics['r2_test']) else "N/A",
        'nonzero_coefs': n_nonzero,
        'p_n_ratio': f"{config['p']/config['n']:.2f}"
    }

# Create lambda grid
lambda_grid = create_lambda_grid()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Bias-Variance Playground", className="text-center mb-4"),
            html.P("Interactive demonstration of regularization effects in linear models", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Preset scenarios
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Preset Scenarios"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Choose a scenario:"),
                            dcc.Dropdown(
                                id='preset-dropdown',
                                options=[{'label': k, 'value': k} for k in PRESET_SCENARIOS.keys()],
                                value='Custom',
                                clearable=False
                            )
                        ], width=8),
                        dbc.Col([
                            html.Br(),
                            dbc.Button("Apply Preset", id='apply-preset-btn', color='primary', className='mt-2')
                        ], width=4)
                    ]),
                    html.Div(id='preset-rationale', className='mt-3')
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Main content row
    dbc.Row([
        # Left column - Controls and plots
        dbc.Col([
            # Controls panel
            dbc.Card([
                dbc.CardHeader("Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Number of samples (n)", title="More samples generally improve model stability"),
                            dcc.Slider(
                                id='n-slider',
                                min=30, max=1000, step=10, value=200,
                                marks={i: str(i) for i in range(30, 1001, 100)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Number of features (p)", title="More features increase model complexity"),
                            dcc.Slider(
                                id='p-slider',
                                min=2, max=600, step=1, value=50,
                                marks={i: str(i) for i in range(0, 601, 100)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Regularization strength (λ)", title="Higher λ increases bias but reduces variance"),
                            dcc.Slider(
                                id='lambda-slider',
                                min=-4, max=3, step=0.1, value=0,
                                marks={i: f"10^{i}" for i in range(-4, 4)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Model Type"),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[
                                    {'label': 'OLS', 'value': 'OLS'},
                                    {'label': 'Ridge', 'value': 'Ridge'},
                                    {'label': 'LASSO', 'value': 'LASSO'}
                                ],
                                value='Ridge',
                                clearable=False
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Feature correlation (ρ)", title="Higher ρ creates multicollinearity"),
                            dcc.Slider(
                                id='rho-slider',
                                min=0.0, max=0.9, step=0.05, value=0.5,
                                marks={i: f"{i:.1f}" for i in np.arange(0, 1, 0.2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Noise level (σ)", title="Higher σ makes prediction harder"),
                            dcc.Slider(
                                id='sigma-slider',
                                min=0.1, max=3.0, step=0.1, value=1.0,
                                marks={i: f"{i:.1f}" for i in np.arange(0, 3.1, 0.5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Random seed"),
                            dbc.Input(
                                id='seed-input',
                                type='number',
                                value=42,
                                min=1, max=10000
                            )
                        ], width=6),
                        dbc.Col([
                            html.Br(),
                            dbc.Button("Regenerate Data", id='regenerate-btn', color='primary', className='mt-2')
                        ], width=6)
                    ])
                ])
            ], className="mb-4"),
            
            # KPI cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Train RMSE", className="card-title"),
                            html.H2(id='train-rmse', className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Test RMSE", className="card-title"),
                            html.H2(id='test-rmse', className="text-danger")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Train R²", className="card-title"),
                            html.H2(id='train-r2', className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Test R²", className="card-title"),
                            html.H2(id='test-r2', className="text-info")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Plots
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='error-curve')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='coefficient-plot')
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='prediction-scatter')
                ], width=12)
            ], className="mb-4"),
            
            # Explanations
            dbc.Card([
                dbc.CardHeader("Explain This View"),
                dbc.CardBody([
                    html.Div(id='explanations')
                ])
            ], className="mb-4")
            
        ], width=8),
        
        # Right column - Snapshots
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Results Snapshot"),
                dbc.CardBody([
                    dbc.Button("Save Snapshot", id='save-snapshot-btn', color='success', className='mb-3'),
                    html.Div(id='snapshot-table'),
                    html.Hr(),
                    dbc.Button("Download CSV", id='download-csv-btn', color='info', className='mb-2'),
                    dcc.Download(id='download-csv')
                ])
            ])
        ], width=4)
    ]),
    
    # Status and warnings
    dbc.Row([
        dbc.Col([
            dbc.Alert(id='status-alert', is_open=False, dismissable=True)
        ])
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(id='footer-info', className="text-muted small")
        ])
    ])
], fluid=True)

@app.callback(
    [Output('n-slider', 'value'),
     Output('p-slider', 'value'),
     Output('lambda-slider', 'value'),
     Output('model-dropdown', 'value'),
     Output('rho-slider', 'value'),
     Output('sigma-slider', 'value'),
     Output('seed-input', 'value'),
     Output('preset-rationale', 'children')],
    [Input('apply-preset-btn', 'n_clicks')],
    [State('preset-dropdown', 'value')]
)
def apply_preset(n_clicks, preset_name):
    """Apply preset scenario settings"""
    if n_clicks is None or preset_name == 'Custom':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, ""
    
    preset = PRESET_SCENARIOS[preset_name]
    
    # Convert lambda to log scale
    lambda_log = np.log10(preset['lambda']) if preset['lambda'] is not None else 0
    
    rationale_text = html.P([
        html.Strong(f"Rationale: "),
        preset['rationale']
    ], className="text-muted small")
    
    return (preset['n'], preset['p'], lambda_log, preset['model'], 
            preset['rho'], preset['sigma'], preset['seed'], rationale_text)

@app.callback(
    [Output('train-rmse', 'children'),
     Output('test-rmse', 'children'),
     Output('train-r2', 'children'),
     Output('test-r2', 'children'),
     Output('error-curve', 'figure'),
     Output('coefficient-plot', 'figure'),
     Output('prediction-scatter', 'figure'),
     Output('explanations', 'children'),
     Output('status-alert', 'is_open'),
     Output('status-alert', 'children'),
     Output('status-alert', 'color'),
     Output('footer-info', 'children')],
    [Input('n-slider', 'value'),
     Input('p-slider', 'value'),
     Input('lambda-slider', 'value'),
     Input('model-dropdown', 'value'),
     Input('rho-slider', 'value'),
     Input('sigma-slider', 'value'),
     Input('seed-input', 'value'),
     Input('regenerate-btn', 'n_clicks')]
)
def update_app(n, p, lambda_val, model, rho, sigma, seed, regenerate_clicks):
    """Main callback that updates all components"""
    
    # Convert lambda from log scale
    lambda_value = 10 ** lambda_val
    
    # Generate data
    X, y, beta_true = generate_data(n, p, rho, sigma, seed)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Standardize
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(
        X_train, X_test, y_train, y_test
    )
    
    # Check for edge cases
    warning_msg = ""
    warning_color = "warning"
    show_warning = False
    
    if p >= n and model == 'OLS':
        warning_msg = f"⚠️ OLS is ill-posed when p ≥ n (XᵀX is singular); use Ridge/LASSO to regularize. Current: p={p}, n={n}"
        warning_color = "danger"
        show_warning = True
    
    # Fit model and get metrics
    metrics = fit_and_metrics(model, lambda_value, X_train_s, y_train_c, X_test_s, y_test_c)
    
    if not metrics['converged']:
        warning_msg = f"⚠️ Model failed to converge: {metrics.get('error', 'Unknown error')}"
        warning_color = "danger"
        show_warning = True
    
    # Update KPI cards
    train_rmse = f"{metrics['rmse_train']:.3f}" if not np.isnan(metrics['rmse_train']) else "N/A"
    test_rmse = f"{metrics['rmse_test']:.3f}" if not np.isnan(metrics['rmse_test']) else "N/A"
    train_r2 = f"{metrics['r2_train']:.3f}" if not np.isnan(metrics['r2_train']) else "N/A"
    test_r2 = f"{metrics['r2_test']:.3f}" if not np.isnan(metrics['r2_test']) else "N/A"
    
    # Error vs Lambda curve
    if model == 'OLS':
        error_fig = go.Figure()
        error_fig.add_annotation(
            text="λ not applicable to OLS",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        error_fig.update_layout(
            title="Error vs Regularization Strength",
            xaxis_title="log₁₀(λ)",
            yaxis_title="RMSE",
            showlegend=False
        )
    else:
        # Compute error curve
        error_df = sweep_lambda(model, lambda_grid, X_train_s, y_train_c, X_test_s, y_test_c)
        
        error_fig = go.Figure()
        error_fig.add_trace(go.Scatter(
            x=error_df['log10_lambda'],
            y=error_df['rmse_train'],
            mode='lines',
            name='Train RMSE',
            line=dict(color='blue')
        ))
        error_fig.add_trace(go.Scatter(
            x=error_df['log10_lambda'],
            y=error_df['rmse_test'],
            mode='lines',
            name='Test RMSE',
            line=dict(color='red')
        ))
        
        # Add vertical line at current lambda
        error_fig.add_vline(
            x=lambda_val,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Current λ = {lambda_value:.3f}"
        )
        
        # Add annotations for optimal lambda
        if len(error_df) > 0:
            optimal_idx = error_df['rmse_test'].idxmin()
            optimal_lambda = error_df.loc[optimal_idx, 'log10_lambda']
            optimal_rmse = error_df.loc[optimal_idx, 'rmse_test']
            
            error_fig.add_annotation(
                x=optimal_lambda,
                y=optimal_rmse,
                text=f"Optimal λ = {10**optimal_lambda:.3f}<br>RMSE = {optimal_rmse:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="green"
            )
        
        error_fig.update_layout(
            title=f"Error vs Regularization Strength ({model})",
            xaxis_title="log₁₀(λ)",
            yaxis_title="RMSE",
            hovermode='x unified'
        )
    
    # Coefficient plot
    beta_hat = metrics['beta_hat']
    beta_abs = np.abs(beta_hat)
    sorted_indices = np.argsort(beta_abs)[::-1]
    
    k_true = min(5, p)
    colors = ['red' if i < k_true else 'blue' for i in sorted_indices]
    
    coef_fig = go.Figure()
    coef_fig.add_trace(go.Bar(
        x=[f"β{i}" for i in sorted_indices],
        y=beta_hat[sorted_indices],
        marker_color=colors,
        name="Coefficients"
    ))
    
    coef_fig.update_layout(
        title=f"Coefficient Estimates ({model}, λ={lambda_value:.3f})",
        xaxis_title="Features (sorted by magnitude)",
        yaxis_title="Coefficient Value",
        showlegend=False
    )
    
    # Count non-zero coefficients for LASSO
    if model == 'LASSO':
        n_nonzero = np.sum(np.abs(beta_hat) > 1e-6)
        coef_fig.add_annotation(
            text=f"Non-zero coefficients: {n_nonzero}/{p}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15, showarrow=False,
            font=dict(size=12)
        )
    
    # Prediction scatter plot
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=y_test_c,
        y=metrics['yhat_test'],
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6),
        hovertemplate='<b>True:</b> %{x:.3f}<br><b>Predicted:</b> %{y:.3f}<br><b>Index:</b> %{pointIndex}<extra></extra>'
    ))
    
    # Add y=x reference line
    min_val = min(np.min(y_test_c), np.min(metrics['yhat_test']))
    max_val = max(np.max(y_test_c), np.max(metrics['yhat_test']))
    scatter_fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='y = x',
        line=dict(color='red', dash='dash')
    ))
    
    # Add R² annotation
    if not np.isnan(metrics['r2_test']):
        scatter_fig.add_annotation(
            text=f"Test R² = {metrics['r2_test']:.3f}",
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue"
        )
    
    scatter_fig.update_layout(
        title=f"Test Set: Predicted vs Actual ({model}, λ={lambda_value:.3f})",
        xaxis_title="Actual y",
        yaxis_title="Predicted ŷ",
        hovermode='closest'
    )
    
    # Generate explanations
    explanations = generate_explanation(metrics, model, lambda_value, n, p, rho, sigma, beta_hat)
    explanation_elements = [html.P(exp) for exp in explanations]
    
    # Footer info
    footer_text = f"n={n}, p={p}, p/n={p/n:.2f}, ρ={rho:.2f}, σ={sigma:.2f}, k_true={min(5,p)}, model={model}, λ={lambda_value:.3f}"
    
    return (train_rmse, test_rmse, train_r2, test_r2, 
            error_fig, coef_fig, scatter_fig, explanation_elements,
            show_warning, warning_msg, warning_color, footer_text)

@app.callback(
    Output('snapshot-table', 'children'),
    [Input('save-snapshot-btn', 'n_clicks')],
    [State('n-slider', 'value'),
     State('p-slider', 'value'),
     State('lambda-slider', 'value'),
     State('model-dropdown', 'value'),
     State('rho-slider', 'value'),
     State('sigma-slider', 'value'),
     State('seed-input', 'value'),
     State('preset-dropdown', 'value')]
)
def save_snapshot(n_clicks, n, p, lambda_val, model, rho, sigma, seed, preset_name):
    """Save current configuration as snapshot"""
    if n_clicks is None:
        return ""
    
    # Get current metrics (simplified - in real app you'd pass these through state)
    lambda_value = 10 ** lambda_val
    
    # Generate data and metrics for snapshot
    X, y, beta_true = generate_data(n, p, rho, sigma, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train_s, X_test_s, y_train_c, y_test_c, y_mean = standardize_train_test(X_train, X_test, y_train, y_test)
    metrics = fit_and_metrics(model, lambda_value, X_train_s, y_train_c, X_test_s, y_test_c)
    
    # Create snapshot
    config = {
        'n': n, 'p': p, 'model': model, 'lambda': lambda_value if model != 'OLS' else None,
        'rho': rho, 'sigma': sigma, 'seed': seed
    }
    
    snapshot = create_snapshot_data(preset_name, config, metrics, metrics['beta_hat'], datetime.now())
    _snapshots.append(snapshot)
    
    # Create table
    if _snapshots:
        df = pd.DataFrame(_snapshots)
        table = dbc.Table.from_dataframe(
            df[['timestamp', 'preset', 'model', 'train_rmse', 'test_rmse', 'nonzero_coefs']].tail(5),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size='sm'
        )
        return table
    return ""

@app.callback(
    Output('download-csv', 'data'),
    [Input('download-csv-btn', 'n_clicks')]
)
def download_csv(n_clicks):
    """Download snapshots as CSV"""
    if n_clicks is None or not _snapshots:
        return None
    
    df = pd.DataFrame(_snapshots)
    csv_string = df.to_csv(index=False)
    
    return dict(
        content=csv_string,
        filename=f"bias_variance_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, host='0.0.0.0', port=port)