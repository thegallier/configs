# convex_dashboard_optimize_v6_tables.py

import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import dash
# Use modern imports + dash_table
from dash import dcc, html, Input, Output, State, callback_context, no_update, dash_table
from dash.exceptions import PreventUpdate
from scipy.special import comb
from scipy.optimize import minimize
import warnings
import time
import io

# Suppressions
warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')
warnings.filterwarnings("ignore", category=RuntimeWarning)

print(f"--- Initializing ---")
# Print versions...
# ... (version prints) ...

# --- Constants ---
HEATMAP_GRID_SIZE = 11
TIER_MIN, TIER_MAX = 1, 4
DV01_MIN, DV01_MAX = 0, 2500000
DEFAULT_DEGREE = 5
TARGET_LOSING_RATIO = 0.7
OPTIMIZATION_TOLERANCE = 0.01


# =====================================================
# --- HELPER FUNCTIONS (MOVED TO TOP) ---
# =====================================================

def normalize(series, min_val, max_val):
    clipped = np.clip(series, min_val, max_val); range_val = max_val - min_val
    if range_val == 0: return np.zeros_like(clipped)
    return (clipped - min_val) / range_val

def bernstein_basis_1d_scalar(k, d, t):
    if not (0 <= k <= d): raise ValueError("k must be between 0 and d")
    t = np.clip(t, 0.0, 1.0)
    if k == 0 and t == 0: return 1.0
    elif k == d and t == 1: return 1.0
    elif (k > 0 and t == 0) or (k < d and t == 1): return 0.0
    try: binom_coeff = comb(d, k, exact=False)
    except ValueError: return 0.0
    t_pow_k = np.power(t, k); one_minus_t_pow_d_minus_k = np.power(1.0 - t, d - k)
    return binom_coeff * t_pow_k * one_minus_t_pow_d_minus_k

def evaluate_1d_bernstein(x_eval_normalized, C_coeffs, degree):
    if C_coeffs is None: return np.full(np.asarray(x_eval_normalized).shape, np.nan)
    C_coeffs = np.asarray(C_coeffs); x_eval_normalized = np.asarray(x_eval_normalized)
    num_coeffs = len(C_coeffs)
    if num_coeffs != degree + 1: raise ValueError(f"Num coeffs ({num_coeffs}) != degree+1 ({degree+1})")
    basis_matrix = np.array([[bernstein_basis_1d_scalar(k, degree, x) for k in range(degree + 1)] for x in x_eval_normalized.flat])
    y_eval = basis_matrix @ C_coeffs
    return y_eval.reshape(x_eval_normalized.shape)

def fit_1d_convex_bernstein(x_data_normalized, z_data, degree=5, monotonic_increasing=False):
    n_data = len(x_data_normalized)
    if n_data != len(z_data): raise ValueError("x/z length mismatch")
    if n_data < 2: print("Warning: Need >= 2 data points."); return None
    print(f"Attempting fit: Degree={degree}, Data Points={n_data}, Convex=True, MonotonicInc={monotonic_increasing}")
    start_fit_time = time.time(); C = cp.Variable(degree + 1, name=f"C_deg{degree}")
    constraints = []
    if degree >= 2: # Convexity
        for i in range(degree - 1): constraints.append(C[i + 2] - 2 * C[i + 1] + C[i] >= 0)
    if monotonic_increasing and degree >= 1: # Monotonicity
        for i in range(degree): constraints.append(C[i + 1] - C[i] >= 0)
    Basis_Eval = np.array([[bernstein_basis_1d_scalar(i, degree, xk) for i in range(degree + 1)] for xk in x_data_normalized])
    f_eval = Basis_Eval @ C; objective = cp.Minimize(cp.sum_squares(f_eval - z_data))
    problem = cp.Problem(objective, constraints); solver_to_use = cp.SCS
    try:
        problem.solve(solver=solver_to_use, verbose=False, eps=1e-6, max_iters=5000); end_fit_time = time.time()
        print(f"Fit Time: {end_fit_time - start_fit_time:.3f} seconds")
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]: print(f"  Fit successful. Status: {problem.status}, Obj: {problem.value:.4f}"); return C.value
        else: print(f"  Fit FAILED. Status: {problem.status}"); return None
    except Exception as e: end_fit_time = time.time(); print(f"Fit Time (Exception): {end_fit_time - start_fit_time:.3f} seconds"); print(f"  Error during fit: {e}"); return None

def calculate_metrics(df_input, C1_final, C2_final, degree1, degree2, epsilon_map):
    # This function remains the same as the corrected v5
    start_calc_time = time.time(); df_calc = df_input.copy()
    df_calc['tier_norm'] = normalize(df_calc['tier'], TIER_MIN, TIER_MAX); df_calc['value1'] = evaluate_1d_bernstein(df_calc['tier_norm'], C1_final, degree1)
    df_calc['dv01_norm'] = normalize(df_calc['dv01'], DV01_MIN, DV01_MAX); df_calc['value2'] = evaluate_1d_bernstein(df_calc['dv01_norm'], C2_final, degree2)
    df_calc['value1'] = df_calc['value1'].fillna(0); df_calc['value2'] = df_calc['value2'].fillna(0)
    df_calc['epsilon'] = df_calc['cusip'].map(epsilon_map).fillna(0.1); df_calc['side_sign'] = np.where(df_calc['side'] == 'BUY', -1, 1)
    df_calc['adjustment'] = df_calc['epsilon'] * df_calc['value1'] * df_calc['value2'] * df_calc['side_sign']; df_calc['value3_adjusted_price'] = df_calc['mid'] + df_calc['adjustment']
    winning_mask_buy = (df_calc['side'] == 'BUY') & (df_calc['value3_adjusted_price'] > df_calc['tradePrice']); winning_mask_sell = (df_calc['side'] == 'SELL') & (df_calc['value3_adjusted_price'] < df_calc['tradePrice'])
    winning_mask = winning_mask_buy | winning_mask_sell; df_calc['is_winning'] = winning_mask; df_calc['is_losing'] = ~winning_mask
    total_dv01 = df_calc['dv01'].sum(); losing_dv01 = 0; losing_dv01_ratio = 0
    if total_dv01 > 1e-9: losing_dv01 = df_calc.loc[df_calc['is_losing'], 'dv01'].sum(); losing_dv01_ratio = losing_dv01 / total_dv01
    df_calc['pnl_sign'] = np.where(df_calc['side'] == 'BUY', 1, -1); df_calc['potential_pnl'] = (df_calc['amount'] * (df_calc['mid'] - df_calc['value3_adjusted_price']) * df_calc['pnl_sign'])
    actual_winning_pnl = df_calc.loc[df_calc['is_winning'], 'potential_pnl'].sum(); potential_pnl_all_trades = df_calc['potential_pnl'].sum()
    favorable_market_mask_buy = (df_calc['side'] == 'BUY') & (df_calc['tradePrice'] < df_calc['mid']); favorable_market_mask_sell = (df_calc['side'] == 'SELL') & (df_calc['tradePrice'] > df_calc['mid'])
    favorable_market_mask = favorable_market_mask_buy | favorable_market_mask_sell; df_calc['is_favorable_market'] = favorable_market_mask
    pnl_favorable_market = df_calc.loc[favorable_market_mask, 'potential_pnl'].sum()
    efficiency_ratio = actual_winning_pnl / potential_pnl_all_trades if abs(potential_pnl_all_trades) > 1e-9 else 0.0
    return (losing_dv01_ratio, actual_winning_pnl, potential_pnl_all_trades, pnl_favorable_market, efficiency_ratio, df_calc)

# --- NEW: Aggregation Function ---
def aggregate_metrics(df_results, group_key):
    """Aggregates metrics by a specified key (cusip, tier, customerName)."""
    if df_results is None or group_key not in df_results.columns:
        return pd.DataFrame() # Return empty if data missing

    grouped = df_results.groupby(group_key)

    agg_data = {}
    # Sum DV01 total and losing DV01 per group
    agg_data['total_dv01'] = grouped['dv01'].sum()
    agg_data['losing_dv01'] = grouped.apply(lambda x: x.loc[x['is_losing'], 'dv01'].sum())
    # Sum actual winning PNL per group
    agg_data['winning_pnl'] = grouped.apply(lambda x: x.loc[x['is_winning'], 'potential_pnl'].sum())

    agg_df = pd.DataFrame(agg_data)

    # Calculate losing rate per group, handle division by zero
    agg_df['losing_rate'] = np.where(
        agg_df['total_dv01'] > 1e-9,
        agg_df['losing_dv01'] / agg_df['total_dv01'],
        0.0
    )
    # Select and rename columns for clarity
    agg_df = agg_df[['losing_rate', 'winning_pnl']].reset_index()
    agg_df.rename(columns={group_key: 'GroupValue'}, inplace=True)
    agg_df['GroupKey'] = group_key # Add key column for potential multi-table use

    return agg_df

def generate_heatmap_data(C1_base, C2_base, degree, df_base, epsilon_map, r1_grid, r2_grid):
    # This function remains the same
    print("Generating heatmap data...")
    start_heatmap_time = time.time(); n_r1 = len(r1_grid); n_r2 = len(r2_grid)
    efficiency_matrix = np.full((n_r1, n_r2), np.nan); losing_ratio_matrix = np.full((n_r1, n_r2), np.nan)
    if C1_base is None or C2_base is None: print("  Skipping heatmap: Base fits missing."); return efficiency_matrix, losing_ratio_matrix
    for i, r1 in enumerate(r1_grid):
        for j, r2 in enumerate(r2_grid):
            C1_r = C1_base * r1; C2_r = C2_base * r2
            try: (losing_dv01_ratio, _, _, _, efficiency_ratio, _) = calculate_metrics(df_base, C1_r, C2_r, degree, degree, epsilon_map); efficiency_matrix[i, j] = efficiency_ratio; losing_ratio_matrix[i, j] = losing_dv01_ratio
            except Exception as e: print(f"  Error heatmap metrics r1={r1}, r2={r2}: {e}")
    end_heatmap_time = time.time(); print(f"Heatmap data finished in {end_heatmap_time - start_heatmap_time:.3f} seconds.")
    return efficiency_matrix, losing_ratio_matrix

def create_heatmap_figure(matrix, r1_grid, r2_grid, title, z_label, colorscale='Viridis'):
    # This function remains the same
    fig = go.Figure(data=go.Heatmap(z=matrix, x=r2_grid, y=r1_grid, colorscale=colorscale, colorbar=dict(title=z_label), hoverongaps=False))
    fig.update_layout(title=title, xaxis_title="Rescaler 2 (DV01)", yaxis_title="Rescaler 1 (Tier)", yaxis_autorange='reversed', margin=dict(l=50, r=20, t=50, b=50))
    return fig

# --- Optimization Helper (Unchanged) ---
def optimization_objective_and_constraints(rescalers, C1_base, C2_base, degree, df_base, epsilon_map, target_ratio, tolerance):
    r1, r2 = rescalers; C1_r = (C1_base * r1) if C1_base is not None else None; C2_r = (C2_base * r2) if C2_base is not None else None
    if C1_r is None or C2_r is None: print("Opt Func: Base fit missing."); return 1e12, 1.0
    try: (losing_ratio, actual_winning_pnl, _, _, _, _) = calculate_metrics(df_base, C1_r, C2_r, degree, degree, epsilon_map); objective_value = -actual_winning_pnl; constraint_value = tolerance - abs(losing_ratio - target_ratio); return objective_value, constraint_value
    except Exception as e: print(f"ERROR in opt objective: {e}"); return 1e12, -1.0

# =====================================================
# --- END HELPER FUNCTIONS ---
# =====================================================


# --- Data Generation (Now uses helpers defined above) ---
print("Generating DataFrame...")
np.random.seed(42)
num_rows = 1000; num_cusips = 200
cusips = [f"CUSIP_{i:03d}" for i in range(num_cusips)]
df = pd.DataFrame({'customerName': [f"Cust_{np.random.randint(1, 50)}" for _ in range(num_rows)], 'tier': np.random.randint(1, 5, size=num_rows), 'firmAccount': [f"Acc_{np.random.randint(100, 200)}" for _ in range(num_rows)], 'cusip': np.random.choice(cusips, size=num_rows), 'amount': np.random.randint(10000, 500000, size=num_rows), 'mid': np.random.uniform(98.0, 102.0, size=num_rows), 'side': np.random.choice(['BUY', 'SELL'], size=num_rows), 'dv01': np.sort(np.random.uniform(100, 2500000, size=num_rows)), 'dv01B': lambda x: x['dv01'] * np.random.uniform(0.9, 1.1, size=num_rows), 'tradePrice': 0.0})
buy_mask = df['side'] == 'BUY'; sell_mask = df['side'] == 'SELL'
num_buys = buy_mask.sum(); num_sells = sell_mask.sum()
buy_offsets = np.random.uniform(0.01, 0.15, size=num_buys); sell_offsets = np.random.uniform(0.01, 0.15, size=num_sells)
buy_rand = np.random.rand(num_buys); sell_rand = np.random.rand(num_sells)
buy_prices = np.where(buy_rand < 0.9, df.loc[buy_mask, 'mid'] - buy_offsets, df.loc[buy_mask, 'mid'] + buy_offsets); df.loc[buy_mask, 'tradePrice'] = buy_prices
sell_prices = np.where(sell_rand < 0.9, df.loc[sell_mask, 'mid'] + sell_offsets, df.loc[sell_mask, 'mid'] - sell_offsets); df.loc[sell_mask, 'tradePrice'] = sell_prices
epsilon_val = 0.1; epsilon_map = {cusip: epsilon_val for cusip in cusips}
pwl1_x_orig = np.linspace(TIER_MIN, TIER_MAX, 4); pwl2_x_orig = np.linspace(DV01_MIN, DV01_MAX, 4)
pwl1_x_norm = normalize(pwl1_x_orig, TIER_MIN, TIER_MAX); pwl2_x_norm = normalize(pwl2_x_orig, DV01_MIN, DV01_MAX)
pwl1_z_initial = np.array([0.2, 0.3, 0.7, 0.9]) # Monotonic PWL1
pwl2_z_initial = np.array([0.2, 0.5, 0.7, 1.0]) # Monotonic PWL2
print(f"Generated DataFrame with {len(df)} rows, sorted DV01, modified trade prices, and monotonic PWL Z-init.")


# --- Dash App Initialization ---
print("Initializing Dash app..."); app = dash.Dash(__name__)

# --- App Layout (ADDED Comparison Tables) ---
app.layout = html.Div([
    html.H1("Convex Polynomial Adjustment Dashboard & Optimizer"),
    html.Div([ # Top row: Controls and Polynomial Graphs
        html.Div([ # Control Panel
            # ... (Controls and metrics display remain the same) ...
            html.H3("Controls, Metrics & Optimization"), html.Label("Polynomial Degree:"), dcc.Input(id='input-degree', type='number', value=DEFAULT_DEGREE, min=1, max=10, step=1, style={'marginLeft': '10px', 'marginRight': '20px'}),
            html.Div([ html.Label("Rescale Factor - Poly 1 (Tier):"), dcc.Slider(id='slider-rescale1', min=0.5, max=2.0, step=0.05, value=1.0, marks=None, tooltip={"placement": "bottom", "always_visible": True}) ], style={'marginTop': '15px', 'marginBottom': '10px'}),
            html.Div([ html.Label("Rescale Factor - Poly 2 (DV01):"), dcc.Slider(id='slider-rescale2', min=0.5, max=2.0, step=0.05, value=1.0, marks=None, tooltip={"placement": "bottom", "always_visible": True}) ], style={'marginBottom': '15px'}),
            html.Button('Recalculate Fits & Metrics', id='button-recalculate', n_clicks=0, style={'marginBottom': '5px'}), html.Hr(),
            html.H4("Calculated Metrics (Global)"),
            html.Div([ html.Strong("Losing DV01 Ratio:", style={'marginRight': '10px'}), html.Span(id='metric-dv01-ratio', children="N/A") ]), html.Div([ html.Strong("Actual Winning P&L:", style={'marginRight': '10px'}), html.Span(id='metric-actual-winning-pnl', children="N/A") ]), html.Div([ html.Strong("Potential P&L (All Trades):", style={'marginRight': '10px'}), html.Span(id='metric-potential-pnl-all', children="N/A") ]), html.Div([ html.Strong("P&L (Favorable Market Only):", style={'marginRight': '10px'}), html.Span(id='metric-pnl-favorable', children="N/A") ]), html.Div([ html.Strong("Efficiency Ratio:", style={'marginRight': '10px'}), html.Span(id='metric-efficiency', children="N/A") ]),
            html.Div(id='status-message', children="Adjust controls and click Recalculate.", style={'marginTop': '15px', 'padding': '8px', 'border': '1px solid lightgrey', 'borderRadius': '4px'}),
            html.Hr(style={'marginTop': '20px'}), html.H4("Optimization"), html.P(f"Target Losing DV01 Ratio: {TARGET_LOSING_RATIO:.1f} (Tol: ±{OPTIMIZATION_TOLERANCE:.2f})"), html.Button("Optimize Rescalers", id="btn-optimize", n_clicks=0), html.Div(id='optimize-status-message', style={'marginTop': '10px', 'color': 'blue'}),
            html.Hr(style={'marginTop': '20px'}), html.Button("Download Results (CSV)", id="btn-download-csv", style={'marginTop': '10px'}), dcc.Download(id="download-dataframe-csv")
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'borderRight': '1px solid #ccc'}),
        html.Div([ # Polynomial Graphs
             html.Div([ html.H4("PWL 1 (Tier basis)"), *[html.Div([ html.Label(f"Pt {i+1} Z:"), dcc.Slider(id={'type': 'slider-pwl1', 'index': i}, min=-1, max=2, step=0.05, value=pwl1_z_initial[i], marks=None, tooltip={"placement": "bottom", "always_visible": True}) ], style={'marginBottom': 15}) for i in range(len(pwl1_x_norm))], dcc.Graph(id='graph-pwl1') ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
             html.Div([ html.H4("PWL 2 (DV01 basis)"), *[html.Div([ html.Label(f"Pt {i+1} Z:"), dcc.Slider(id={'type': 'slider-pwl2', 'index': i}, min=-1, max=2, step=0.05, value=pwl2_z_initial[i], marks=None, tooltip={"placement": "bottom", "always_visible": True}) ], style={'marginBottom': 15}) for i in range(len(pwl2_x_norm))], dcc.Graph(id='graph-pwl2') ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
        ], style={'width': '69%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]), # End top row
    html.Hr(), # Separator
    html.Div([ # Heatmap Row
        html.H3("Rescaler Analysis Heatmaps"),
        html.Div([ dcc.Graph(id='graph-heatmap-efficiency') ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px'}),
        html.Div([ dcc.Graph(id='graph-heatmap-losing') ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px'})
    ]),
    html.Hr(), # Separator

    # --- NEW: Comparison Tables Row ---
    html.Div([
        html.H3("Comparison by Group (Initial vs Current)"),
        html.Div([
            html.H4("By CUSIP"),
            dash_table.DataTable(
                id='table-cusip-comparison',
                columns=[
                    {'name': 'CUSIP', 'id': 'GroupValue'},
                    {'name': 'Init. Losing Rate', 'id': 'losing_rate_initial', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Curr. Losing Rate', 'id': 'losing_rate_current', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Δ Losing Rate', 'id': 'losing_rate_delta', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Init. Winning PNL', 'id': 'winning_pnl_initial', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                    {'name': 'Curr. Winning PNL', 'id': 'winning_pnl_current', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                    {'name': 'Δ Winning PNL', 'id': 'winning_pnl_delta', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                ],
                data=[], # Initially empty
                page_size=10,
                sort_action='native',
                filter_action='native',
                style_table={'overflowX': 'auto', 'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px', 'width': '120px', 'maxWidth': '180px'},
                style_header={'fontWeight': 'bold'}
            )
        ], style={'width': '33%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign':'top'}),
        html.Div([
            html.H4("By Tier"),
            dash_table.DataTable(
                id='table-tier-comparison',
                columns=[ # Same columns as cusip, just change title
                    {'name': 'Tier', 'id': 'GroupValue'},
                    {'name': 'Init. Losing Rate', 'id': 'losing_rate_initial', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Curr. Losing Rate', 'id': 'losing_rate_current', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Δ Losing Rate', 'id': 'losing_rate_delta', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Init. Winning PNL', 'id': 'winning_pnl_initial', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                    {'name': 'Curr. Winning PNL', 'id': 'winning_pnl_current', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                    {'name': 'Δ Winning PNL', 'id': 'winning_pnl_delta', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                ],
                data=[], page_size=10, sort_action='native', filter_action='native', style_table={'height': '300px', 'overflowY': 'auto'}, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}
            )
        ], style={'width': '33%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign':'top'}),
        html.Div([
            html.H4("By Customer"),
            dash_table.DataTable(
                id='table-customer-comparison',
                columns=[ # Same columns, change title
                    {'name': 'Customer', 'id': 'GroupValue'},
                    {'name': 'Init. Losing Rate', 'id': 'losing_rate_initial', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Curr. Losing Rate', 'id': 'losing_rate_current', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Δ Losing Rate', 'id': 'losing_rate_delta', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.percentage)},
                    {'name': 'Init. Winning PNL', 'id': 'winning_pnl_initial', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                    {'name': 'Curr. Winning PNL', 'id': 'winning_pnl_current', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                    {'name': 'Δ Winning PNL', 'id': 'winning_pnl_delta', 'type': 'numeric', 'format': dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.fixed).group(True)},
                ],
                data=[], page_size=10, sort_action='native', filter_action='native', style_table={'height': '300px', 'overflowY': 'auto'}, style_cell={'textAlign': 'left'}, style_header={'fontWeight': 'bold'}
            )
        ], style={'width': '33%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign':'top'}),
    ]), # End comparison tables row

    # Stores (Unchanged)
    dcc.Store(id='store-dataframe', data=df.to_json(date_format='iso', orient='split')), dcc.Store(id='store-epsilon', data=epsilon_map), dcc.Store(id='store-calculated-df'), dcc.Store(id='store-base-c1'), dcc.Store(id='store-base-c2'),
])


# --- Main Callback (UPDATED Outputs, double metric calc, aggregation) ---
@app.callback(
    [# Graphs
     Output('graph-pwl1', 'figure'), Output('graph-pwl2', 'figure'),
     # Global Metrics
     Output('metric-dv01-ratio', 'children'), Output('metric-actual-winning-pnl', 'children'),
     Output('metric-potential-pnl-all', 'children'), Output('metric-pnl-favorable', 'children'),
     Output('metric-efficiency', 'children'),
     # Status
     Output('status-message', 'children'), Output('status-message', 'style'),
     # Stores
     Output('store-calculated-df', 'data'), Output('store-base-c1', 'data'), Output('store-base-c2', 'data'),
     # Heatmaps
     Output('graph-heatmap-efficiency', 'figure'), Output('graph-heatmap-losing', 'figure'),
     # NEW Comparison Tables Data
     Output('table-cusip-comparison', 'data'),
     Output('table-tier-comparison', 'data'),
     Output('table-customer-comparison', 'data')],
    [Input('button-recalculate', 'n_clicks')],
    [State({'type': 'slider-pwl1', 'index': dash.ALL}, 'value'), State({'type': 'slider-pwl2', 'index': dash.ALL}, 'value'), State('input-degree', 'value'), State('slider-rescale1', 'value'), State('slider-rescale2', 'value'), State('store-dataframe', 'data'), State('store-epsilon', 'data')]
)
def update_dashboard(n_clicks, slider_vals_pwl1, slider_vals_pwl2, degree, rescale1, rescale2, df_json, epsilon_map_state):

    ctx = callback_context
    initial_status_style = {'marginTop': '15px', 'padding': '8px', 'border': '1px solid lightgrey', 'borderRadius': '4px'}
    # Initial values for all outputs
    base_c1_data, base_c2_data, calculated_df_json = None, None, None
    dv01_ratio_str, actual_winning_pnl_str, potential_pnl_all_str, pnl_favorable_str, efficiency_str = "N/A", "N/A", "N/A", "N/A", "N/A"
    table_cusip_data, table_tier_data, table_customer_data = [], [], []
    fig_empty = go.Figure().update_layout(title="N/A", xaxis={'visible': False}, yaxis={'visible': False})
    heatmap_fig_empty = go.Figure().update_layout(title="Calculate base fits first", xaxis={'visible': False}, yaxis={'visible': False})

    if not ctx.triggered or ctx.triggered[0]['prop_id'] == '.' or n_clicks == 0:
         print("Initial load or no clicks yet.")
         return ([fig_empty, fig_empty] + # Graphs
                 [dv01_ratio_str, actual_winning_pnl_str, potential_pnl_all_str, pnl_favorable_str, efficiency_str] + # Metrics
                 ["Adjust sliders/degree and click Recalculate.", initial_status_style, calculated_df_json, base_c1_data, base_c2_data] + # Status, Stores
                 [heatmap_fig_empty, heatmap_fig_empty] + # Heatmaps
                 [table_cusip_data, table_tier_data, table_customer_data]) # Tables

    print(f"\n--- Recalculating (Button Click {n_clicks}) ---")
    status_message = ""; status_style = initial_status_style.copy(); success = False; calc_success = False
    current_pwl1_z = np.array(slider_vals_pwl1); current_pwl2_z = np.array(slider_vals_pwl2); df_state = pd.read_json(df_json, orient='split')
    print(f"Using Degree: {degree}, Rescale1: {rescale1}, Rescale2: {rescale2}")
    if degree is None or degree < 1: degree = DEFAULT_DEGREE; status_message += f"Invalid degree, using default {DEFAULT_DEGREE}. "; print("Warning: Invalid degree.")

    # --- Step 1: Perform BASE Fits ---
    C1_base = fit_1d_convex_bernstein(pwl1_x_norm, current_pwl1_z, degree=degree, monotonic_increasing=False)
    C2_base = fit_1d_convex_bernstein(pwl2_x_norm, current_pwl2_z, degree=degree, monotonic_increasing=True)
    base_fits_ok = C1_base is not None and C2_base is not None
    if base_fits_ok:
        base_c1_data = C1_base.tolist(); base_c2_data = C2_base.tolist()
    else:
        status_message += "ERROR: Base polynomial fit failed. Calculations aborted. "; status_style['color'] = 'red'; status_style['borderColor'] = 'salmon'

    # --- Step 2: Calculate Initial Metrics (using base fits, rescalers=1.0) ---
    df_initial_results = None
    if base_fits_ok:
        print("Calculating initial metrics (r=1.0)...")
        try:
            (_, _, _, _, _, df_initial_results) = calculate_metrics(df_state, C1_base, C2_base, degree, degree, epsilon_map_state)
        except Exception as e:
            print(f"ERROR calculating initial metrics: {e}")
            status_message += f"ERROR initial metrics: {e}. "; status_style['color'] = 'red'; status_style['borderColor'] = 'salmon'

    # --- Step 3: Calculate Current Metrics (using rescaled fits) ---
    C1_rescaled = (C1_base * rescale1) if base_fits_ok else None
    C2_rescaled = (C2_base * rescale2) if base_fits_ok else None
    df_current_results = None
    if base_fits_ok: # Only proceed if base fits worked
        print("Calculating current metrics...")
        try:
            (losing_dv01_ratio, actual_winning_pnl, potential_pnl_all_trades, pnl_favorable_market, efficiency_ratio, df_current_results) = calculate_metrics(df_state, C1_rescaled, C2_rescaled, degree, degree, epsilon_map_state)
            print(f"  Raw Current Metrics: DV01 Ratio={losing_dv01_ratio}, Winning PNL={actual_winning_pnl}, Potential PNL={potential_pnl_all_trades}, Favorable PNL={pnl_favorable_market}, Efficiency={efficiency_ratio}")
            dv01_ratio_str = f"{losing_dv01_ratio:.2%}"; actual_winning_pnl_str = f"{actual_winning_pnl:,.2f}"; potential_pnl_all_str = f"{potential_pnl_all_trades:,.2f}"; pnl_favorable_str = f"{pnl_favorable_market:,.2f}"; efficiency_str = f"{efficiency_ratio:.2%}"
            pnl_check_ok = potential_pnl_all_trades < pnl_favorable_market
            status_message += f"Calculation complete. P&L Check: {'OK' if pnl_check_ok else 'Failed!'}. "
            status_style['color'] = 'green'; status_style['borderColor'] = 'darkseagreen';
            if not pnl_check_ok: status_style['color'] = 'orange'; status_style['borderColor'] = 'darkorange'
            calc_success = True
            calculated_df_json = df_current_results.to_json(date_format='iso', orient='split') # Store current results for download
        except Exception as e:
            print(f"ERROR calculating current metrics: {e}"); status_message += f"ERROR current metrics: {e}. "
            dv01_ratio_str = "Calc Err"; actual_winning_pnl_str = "Calc Err"; potential_pnl_all_str = "Calc Err"; pnl_favorable_str = "Calc Err"; efficiency_str = "Calc Err"; status_style['color'] = 'red'; status_style['borderColor'] = 'salmon'
    elif not base_fits_ok: # Fill metrics if base fit failed
        dv01_ratio_str = "Fit Fail"; actual_winning_pnl_str = "Fit Fail"; potential_pnl_all_str = "Fit Fail"; pnl_favorable_str = "Fit Fail"; efficiency_str = "Fit Fail"

    # --- Step 4: Aggregate and Compare Metrics for Tables ---
    if calc_success and df_initial_results is not None: # Need both initial and current results
        print("Aggregating comparison tables...")
        for group_key, table_data_list in [('cusip', table_cusip_data), ('tier', table_tier_data), ('customerName', table_customer_data)]:
            try:
                agg_initial = aggregate_metrics(df_initial_results, group_key)
                agg_current = aggregate_metrics(df_current_results, group_key)

                if not agg_initial.empty and not agg_current.empty:
                    # Merge initial and current aggregated results
                    agg_merged = pd.merge(agg_initial, agg_current, on='GroupValue', suffixes=('_initial', '_current'), how='outer').fillna(0)

                    # Calculate deltas
                    agg_merged['losing_rate_delta'] = agg_merged['losing_rate_current'] - agg_merged['losing_rate_initial']
                    agg_merged['winning_pnl_delta'] = agg_merged['winning_pnl_current'] - agg_merged['winning_pnl_initial']

                    # Convert to list of dicts for Dash table
                    if group_key == 'cusip': table_cusip_data = agg_merged.to_dict('records')
                    elif group_key == 'tier': table_tier_data = agg_merged.to_dict('records')
                    elif group_key == 'customerName': table_customer_data = agg_merged.to_dict('records')
                else:
                    print(f"Warning: Aggregation produced empty results for key '{group_key}'.")

            except Exception as e:
                print(f"ERROR aggregating/merging for key '{group_key}': {e}")
                status_message += f"ERROR generating table for {group_key}. "
                status_style['color'] = 'red'; status_style['borderColor'] = 'salmon'
    else:
        status_message += " Comparison tables not generated (missing initial or current results). "


    # --- Step 5: Generate Plots (logic remains same) ---
    fig1 = go.Figure(); fig2 = go.Figure(); x_plot_normalized = np.linspace(0, 1, 100); x_plot_tier = np.linspace(TIER_MIN, TIER_MAX, 100); x_plot_dv01 = np.linspace(DV01_MIN, DV01_MAX, 100); plot1_error = False; plot2_error = False
    if C1_base is not None:
        try: y_fit1_orig = evaluate_1d_bernstein(x_plot_normalized, C1_base, degree); fig1.add_trace(go.Scatter(x=x_plot_tier, y=y_fit1_orig, mode='lines', name=f'Base Fit', line=dict(color='rgba(255,0,0,0.5)', dash='dash'))); y_fit1_rescaled = evaluate_1d_bernstein(x_plot_normalized, C1_rescaled, degree); fig1.add_trace(go.Scatter(x=x_plot_tier, y=y_fit1_rescaled, mode='lines', name=f'Rescaled (x{rescale1:.2f})', line=dict(color='red')))
        except Exception as e: print(f"Error eval/plot B1: {e}"); plot1_error = True
    fig1.add_trace(go.Scatter(x=pwl1_x_orig, y=pwl1_z_initial, mode='markers', name='Initial PWL Pts', marker=dict(color='grey', symbol='cross'))); fig1.add_trace(go.Scatter(x=pwl1_x_orig, y=current_pwl1_z, mode='markers', name='Current PWL Pts', marker=dict(color='blue', symbol='circle-open')))
    fig1.update_layout(title=f'Tier Poly (Deg {degree}){" - Err" if plot1_error else ""}', xaxis_title='Tier (Original Scale)', yaxis_title='Value 1 Basis', margin=dict(l=20, r=20, t=40, b=20), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    if C2_base is not None:
        try: y_fit2_orig = evaluate_1d_bernstein(x_plot_normalized, C2_base, degree); fig2.add_trace(go.Scatter(x=x_plot_dv01, y=y_fit2_orig, mode='lines', name=f'Base Fit (Mono)', line=dict(color='rgba(0,128,0,0.5)', dash='dash'))); y_fit2_rescaled = evaluate_1d_bernstein(x_plot_normalized, C2_rescaled, degree); fig2.add_trace(go.Scatter(x=x_plot_dv01, y=y_fit2_rescaled, mode='lines', name=f'Rescaled (x{rescale2:.2f})', line=dict(color='green')))
        except Exception as e: print(f"Error eval/plot B2: {e}"); plot2_error = True
    fig2.add_trace(go.Scatter(x=pwl2_x_orig, y=pwl2_z_initial, mode='markers', name='Initial PWL Pts', marker=dict(color='grey', symbol='cross'))); fig2.add_trace(go.Scatter(x=pwl2_x_orig, y=current_pwl2_z, mode='markers', name='Current PWL Pts', marker=dict(color='orange', symbol='circle-open')))
    fig2.update_layout(title=f'DV01 Poly (Deg {degree}, Mono){" - Err" if plot2_error else ""}', xaxis_title='DV01 (Original Scale)', yaxis_title='Value 2 Basis', margin=dict(l=20, r=20, t=40, b=20), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    if plot1_error and not calc_success: status_message += " Err plot Fit 1." # Only add if calc didn't already fail
    if plot2_error and not calc_success: status_message += " Err plot Fit 2."

    # --- Step 6: Generate Heatmaps ---
    r1_grid = np.linspace(0.5, 2.0, HEATMAP_GRID_SIZE); r2_grid = np.linspace(0.5, 2.0, HEATMAP_GRID_SIZE)
    if base_fits_ok:
        efficiency_matrix, losing_ratio_matrix = generate_heatmap_data(C1_base, C2_base, degree, df_state, epsilon_map_state, r1_grid, r2_grid)
        heatmap_fig_eff = create_heatmap_figure(efficiency_matrix, r1_grid, r2_grid, "Efficiency Ratio", "Ratio", colorscale='RdYlGn')
        heatmap_fig_losing = create_heatmap_figure(losing_ratio_matrix, r1_grid, r2_grid, "Losing DV01 Ratio", "Ratio", colorscale='Plasma')
    else: heatmap_fig_eff = heatmap_fig_empty; heatmap_fig_losing = heatmap_fig_empty; status_message += " Heatmaps not generated."

    print(f"Callback finished. Status: {status_message}")
    # Return all outputs
    return ([fig1, fig2] +
            [dv01_ratio_str, actual_winning_pnl_str, potential_pnl_all_str, pnl_favorable_str, efficiency_str] +
            [status_message, status_style, calculated_df_json, base_c1_data, base_c2_data] +
            [heatmap_fig_eff, heatmap_fig_losing] +
            [table_cusip_data, table_tier_data, table_customer_data]) # Add table data

# --- Optimization Callback (Unchanged) ---
@app.callback(
    [Output('slider-rescale1', 'value'), Output('slider-rescale2', 'value'), Output('optimize-status-message', 'children')],
    [Input('btn-optimize', 'n_clicks')],
    [State('store-base-c1', 'data'), State('store-base-c2', 'data'), State('input-degree', 'value'), State('store-dataframe', 'data'), State('store-epsilon', 'data')], prevent_initial_call=True
)
def optimize_rescalers(n_clicks, base_c1_data, base_c2_data, degree, df_json, epsilon_map_state):
    if not base_c1_data or not base_c2_data: return no_update, no_update, "Opt failed: Base fits needed first."
    print("\n--- Starting Optimization ---"); opt_start_time = time.time(); C1_base = np.array(base_c1_data); C2_base = np.array(base_c2_data); df_base = pd.read_json(df_json, orient='split')
    if degree is None or degree < 1: degree = DEFAULT_DEGREE
    def objective_func(rescalers): obj_val, _ = optimization_objective_and_constraints(rescalers, C1_base, C2_base, degree, df_base, epsilon_map_state, TARGET_LOSING_RATIO, OPTIMIZATION_TOLERANCE); return obj_val
    constraints = ({'type': 'ineq', 'fun': lambda r: optimization_objective_and_constraints(r, C1_base, C2_base, degree, df_base, epsilon_map_state, TARGET_LOSING_RATIO, OPTIMIZATION_TOLERANCE)[1]})
    bounds = [(0.5, 2.0), (0.5, 2.0)]; initial_guess = [1.0, 1.0]; opt_status_msg = "Running optimizer..."
    try:
        result = minimize(objective_func, initial_guess, method='COBYLA', bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 100})
        opt_end_time = time.time(); print(f"Opt finished in {opt_end_time - opt_start_time:.3f}s."); print(f"Opt Result:\n{result}")
        if result.success:
            optimal_r1, optimal_r2 = result.x; final_obj, final_constr_val = optimization_objective_and_constraints(result.x, C1_base, C2_base, degree, df_base, epsilon_map_state, TARGET_LOSING_RATIO, OPTIMIZATION_TOLERANCE); final_winning_pnl = -final_obj
            ratio_diff = OPTIMIZATION_TOLERANCE - final_constr_val; final_ratio = TARGET_LOSING_RATIO + ratio_diff if abs(ratio_diff) <= OPTIMIZATION_TOLERANCE else TARGET_LOSING_RATIO + np.sign(ratio_diff) * OPTIMIZATION_TOLERANCE
            if final_constr_val >= -OPTIMIZATION_TOLERANCE * 0.5 : opt_status_msg = f"Opt successful! Optimal: r1={optimal_r1:.3f}, r2={optimal_r2:.3f}. Est. Ratio: {final_ratio:.3f}, Est. Max Winning PNL: {final_winning_pnl:,.2f}. Click Recalculate."; return optimal_r1, optimal_r2, opt_status_msg
            else: opt_status_msg = f"Opt finished, constraint unmet (val: {final_constr_val:.4f}). Result: r1={optimal_r1:.3f}, r2={optimal_r2:.3f}. Est PNL: {final_winning_pnl:,.2f}"; return optimal_r1, optimal_r2, opt_status_msg
        else: opt_status_msg = f"Opt failed: {result.message}"; return no_update, no_update, opt_status_msg
    except Exception as e: opt_end_time = time.time(); print(f"Opt error after {opt_end_time - opt_start_time:.3f}s: {e}"); opt_status_msg = f"Opt error: {e}"; return no_update, no_update, opt_status_msg

# --- Download Callback (Unchanged) ---
@app.callback(Output("download-dataframe-csv", "data"), Input("btn-download-csv", "n_clicks"), State("store-calculated-df", "data"), prevent_initial_call=True)
def download_csv(n_clicks, calculated_df_json):
    if calculated_df_json is None: print("Download: No calculated data."); raise PreventUpdate
    print("Download: Preparing CSV..."); df_to_download = pd.read_json(calculated_df_json, orient='split')
    return dcc.send_data_frame(df_to_download.to_csv, "adjusted_trades.csv", index=False)

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash server...")
    app.run(debug=True, use_reloader=False, port=8051)
