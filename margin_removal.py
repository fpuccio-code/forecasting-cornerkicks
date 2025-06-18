
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize
from scipy.stats import poisson

def multiplicative_method(odds_data):
    ########Multiplicative Margin Removal
    fair_prob_mult= pd.DataFrame(index=odds_data.index) 
    Avg_norm=(1/odds_data['Avg>2.5'])+(1/odds_data['Avg<2.5'])
    Avc_norm=(1/odds_data['AvgC>2.5'])+(1/odds_data['AvgC<2.5'])
    AvHAD_norm=(1/odds_data['AvgH'])+(1/odds_data['AvgD'])+(1/odds_data['AvgA'])

    fair_prob_mult['AvgC>2.5'] = (1 / odds_data['AvgC>2.5'])*( 1/ Avc_norm)
    fair_prob_mult['Avg>2.5'] = (1 / odds_data['Avg>2.5'])*( 1/ Avg_norm)   
    fair_prob_mult["AvgH"] = (1 / odds_data['AvgH'])*( 1/ Avg_norm)
    fair_prob_mult['AvgD'] = (1 / odds_data['AvgD'])*( 1/ Avg_norm)
    fair_prob_mult['AvgA'] = (1 / odds_data['AvgA'])*( 1/ Avg_norm)

    return fair_prob_mult




def shin_method(odds_df, odds_cols, max_iter=1000, tol=1e-10):
    odds = odds_df[odds_cols].copy()
    implied_prob = 1 / odds

    # Initial overround (bookmaker margin)
    overround = implied_prob.sum(axis=1)

    # Initialize theta vector (one per row)
    theta = pd.Series(0.0, index=odds.index)

    def shin_fixed_point(theta, implied_prob):
        """ Shin fixed-point iteration """
        numerator = implied_prob
        denominator = 1 - theta[:, None] * (1 - implied_prob)
        return (numerator / denominator).sum(axis=1) - 1

    # We'll solve theta per row by fixed-point iteration
    for i in range(max_iter):
        denom = 1 - theta.values[:, None] * (1 - implied_prob)
        # Avoid division by zero or negative denominators
        denom = np.maximum(denom, 1e-12)
        p_hat = implied_prob / denom
        new_theta = 1 - (1 / p_hat.sum(axis=1))
        # Clamp theta to [0,1)
        new_theta = np.clip(new_theta, 0, 0.9999)
        
        # Check convergence
        if np.max(np.abs(new_theta - theta)) < tol:
            theta = new_theta
            break
        theta = new_theta

    # Calculate final fair probabilities
    denom = 1 - theta.values[:, None] * (1 - implied_prob)
    fair_probs = implied_prob / denom
    # Normalize to sum to 1 (just in case)
    fair_probs = fair_probs.div(fair_probs.sum(axis=1), axis=0)
    fair_probs.columns = odds_cols

    return fair_probs


def power_margin_removal(odds_df, odds_cols):
    implied_prob = 1 / odds_df[odds_cols]

    def adjust_probs(alpha, p):
        # Apply power alpha and normalize
        powered = np.power(p, alpha)
        return powered / powered.sum(axis=1, keepdims=True)

    def objective(alpha, p):
        # We want to find alpha that makes sum of adjusted probs == 1 per row,
        # but it always does because of normalization.
        # Instead, we can minimize difference between margin and 1,
        # so here, we minimize sum of squares difference or use a heuristic.
        # For simplicity, let's find alpha so that mean adjusted implied probabilities
        # matches a target like 1.0 margin.
        
        # For vectorized root-finding, we solve per row (vectorized approx)
        adjusted = adjust_probs(alpha, p)
        # margin per row (sum adjusted probs, should be 1, so difference zero)
        margin = adjusted.sum(axis=1)
        # difference from 1
        diff = margin - 1
        return np.mean(diff**2)  # minimize mean squared difference

    alpha = 1.0  # or let user set alpha manually or optimize it

    fair_probs = adjust_probs(alpha, implied_prob.values)
    fair_probs_df = pd.DataFrame(fair_probs, columns=odds_cols, index=odds_df.index)

    return fair_probs_df


####Cross information
# Convert odds to fair probabilities (margin removal)
def odds_to_probs(odds):
    probs = 1 / odds
    margin = probs.sum()
    return probs / margin

# Model implied probabilities pH', pD', pL' from lambda1, lambda2
def model_probs(lambda1, lambda2, ou_threshold=2.5):
    max_goals = 10  # cutoff for summation
    p_x = poisson.pmf(np.arange(max_goals+1), lambda1)
    p_y = poisson.pmf(np.arange(max_goals+1), lambda2)
    
    # Home win: sum over x > y
    pH_prime = 0
    for x in range(max_goals+1):
        for y in range(x):
            pH_prime += p_x[x]*p_y[y]
            
    # Draw: sum over x == y
    pD_prime = sum(p_x[i]*p_y[i] for i in range(max_goals+1))
    
    # Total goals convolution
    p_total = np.convolve(p_x, p_y)
    
    k = int(np.floor(ou_threshold))
    pL_prime = p_total[:k+1].sum()
    
    return pH_prime, pD_prime, pL_prime

# Loss function for minimization
def loss(params, pH, pD, pL):
    lambda1, lambda2 = params
    if lambda1 <= 0 or lambda2 <= 0:
        return np.inf
    
    pH_prime, pD_prime, pL_prime = model_probs(lambda1, lambda2)
    return (pH - pH_prime)**2 + (pD - pD_prime)**2 + (pL - pL_prime)**2

# Estimate lambdas from market probabilities
def estimate_lambdas(pH, pD, pL):
    initial_guess = [1.0, 1.0]
    bounds = [(1e-5, None), (1e-5, None)]
    result = minimize(loss, initial_guess, args=(pH, pD, pL), method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        return result.x
    else:
        raise RuntimeError("Optimization failed")

# Example usage for a DataFrame of odds
def process_all_matches(odds_df):
    results = []
    for idx, row in odds_df.iterrows():
        # Extract market odds
        had_odds = np.array([row['AvgH'], row['AvgD'], row['AvgA']])
        ou_under_odds = row['Avg<2.5']
        
        # Convert to fair probabilities
        pH, pD, pL = odds_to_probs(had_odds)
        pL = 1 / ou_under_odds
        # Remove margin on pL if you want more accuracy:
        # Here pL is directly taken from O/U under odds (you may normalize with over too)
        
        try:
            lambda1, lambda2 = estimate_lambdas(pH, pD, pL)
            TG = lambda1 + lambda2
            SUP = lambda1 - lambda2
            results.append({
                'MatchID': idx,
                'Lambda1': lambda1,
                'Lambda2': lambda2,
                'TG': TG,
                'SUP': SUP
            })
        except RuntimeError:
            results.append({
                'MatchID': idx,
                'Lambda1': np.nan,
                'Lambda2': np.nan,
                'TG': np.nan,
                'SUP': np.nan
            })
    return pd.DataFrame(results).set_index('MatchID')
