import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from margin_removal import multiplicative_method, shin_method, power_margin_removal,process_all_matches,process_all_matches
from data_betclic import data_set,odds_data
from regression_functions import shape_parameter_k, shape_parameter_theta, regression_covariates
import seaborn as sns

##############################
# Clean data and check biases
##############################
# Define leagues and their codes
leagues = {"Serie A": "I1", "La Liga": "SP1", "Premier League": "E0",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
     "Primeira Liga": "P1"}

# Define seasons to analyze
seasons = {
    "2021-2022": "2122",
    "2022-2023": "2223"
}

df_data=data_set(leagues,seasons)
# # === PLOTS ===
# plt.figure(figsize=(12, 6))
# # 1. Dispersion Index (Total Goals)
# ax1 = plt.subplot(1, 2, 1)
# sorted_df = df_data.sort_values(by='DispersionIndexCorners')
# ax1.bar(sorted_df.index, sorted_df['DispersionIndexCorners'] - 1, color='skyblue', edgecolor='black',alpha=0.5)
# ax1.set_xticks(range(len(sorted_df)))
# ax1.set_xticklabels(sorted_df.index, rotation=45, ha='right')
# ax1.set_ylabel("Dispersion Index Percentage")
# ax1.set_title("Dispersion Index of Total Corners (2021–2023)")
# # 2. Dispersion Index (Goals)
# ax2 = plt.subplot(1, 2, 2)
# sorted_df2 = df_data.sort_values(by='DispersionIndexGoals')
# ax2.bar(sorted_df2.index, np.abs(sorted_df2['DispersionIndexGoals'] - 1), color='lightgreen', edgecolor='black',alpha=0.5)
# ax2.set_xticks(range(len(sorted_df2)))
# ax2.set_xticklabels(sorted_df2.index, rotation=45, ha='right')
# ax2.set_ylabel("Dispersion Index Percentage")
# ax2.set_title("Dispersion Index of Total Goals (2021–2023)")
# plt.tight_layout()
# plt.show()
# exit()

##############################
# Odds data and Margin Removal
##############################
#.1 Upload odds 2024-2025 for maximum dispersion Leauge
max_key = df_data['DispersionIndexCorners'].idxmax()
print(f"\n League with maximum dispersion: {max_key}\n")
odds=odds_data(max_key,leagues,"2324")

#.2 Multiplicative Margin Removal Method
fair_prob_mult= multiplicative_method(odds)

#.3 Shin Margin Removal Methods
fair_prob_shin= pd.DataFrame(index=odds.index)
fair_prob_shin[['AvgH', 'AvgD', 'AvgA']] = shin_method(odds, ['AvgH', 'AvgD', 'AvgA'])
fair_prob_shin['Avg>2.5'] = shin_method(odds, ['Avg>2.5', 'Avg<2.5'])['Avg>2.5']
fair_prob_shin['AvgC>2.5']= shin_method(odds, ['AvgC>2.5', 'AvgC<2.5'])['AvgC>2.5']

#.4 Power Margin Removal Methods
fair_prob_power= pd.DataFrame(index=odds.index)
fair_prob_power[['AvgH', 'AvgD', 'AvgA']] = power_margin_removal(odds, ['AvgH', 'AvgD', 'AvgA'])
fair_prob_power['Avg>2.5'] = power_margin_removal(odds, ['Avg>2.5', 'Avg<2.5'])['Avg>2.5']
fair_prob_power['AvgC>2.5'] = power_margin_removal(odds, ['AvgC>2.5', 'AvgC<2.5'])['AvgC>2.5']

#.5 Cross Market information
results = process_all_matches(odds)
MM_data = pd.concat([fair_prob_mult,results], axis=1)
PM_data = pd.concat([fair_prob_power,results], axis=1)
SM_data = pd.concat([fair_prob_shin,results], axis=1)

##############################
# Model Development
##############################
import numpy as np
import pandas as pd
import statsmodels.api as sm

df=regression_covariates(SM_data)

# Prepare regressors and target
X = pd.DataFrame({
    'log_TGi': np.log(df['TG']),
    'log_SUP': np.log(np.abs(df['SUP']) + 0.01),
    'TCTarget': df['TCTarget'],
    'log_HomeAvg3': np.log(df['HomeAvg3'] + 0.01),
    'log_AwayAvg3': np.log(df['AwayAvg3'] + 0.01),
    'log_HomeShotOnGoalAvg3': np.log(df['HomeShotOnGoalAvg3'] + 0.01),
    'log_AwayShotOnGoalAvg3': np.log(df['AwayShotOnGoalAvg3'] + 0.01)
})

# Choose dependent variable, e.g. log Lambda1 or log Lambda2:
y = np.log(df['Lambda1'])
# Add constant term for intercept
X = sm.add_constant(X)
# Fit model
model_lambda = sm.OLS(y, X).fit()

# Actual and predicted values
y_true = y             # Actual target (e.g. log(Lambda1))
y_pred = model_lambda.fittedvalues  # Predicted target from regression

# Create scatter plot
plt.figure(figsize=(6,6))
scatter = sns.scatterplot(x=y_true, y=y_pred, hue=y_true, palette='viridis', s=60)
# Diagonal reference line (perfect prediction)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red', label='Regression Prediction')
plt.xlabel("Actual Values "+r"$\lambda_1$")
plt.ylabel("Predicted Values "+r"$\lambda_1$")
plt.title("Predicted vs. Actual Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#print(model.params)

###################SHAPE PARAMETERS
df['k1']=shape_parameter_k(float(df_data.loc[max_key, ["AvgCorners"]].iloc[0]),float(df_data.loc[max_key, ["SigmaCorners"]].iloc[0]))
y = np.log(np.abs(df['k1']))
# Add constant term for intercept

X = pd.DataFrame({
    'log_SUP': np.log(np.abs(df['SUP']) + 0.01),
  })
X = sm.add_constant(X)

#Fit model
model_k = sm.OLS(y, X).fit()

###################SHAPE PARAMETERS
df['theta1']=shape_parameter_theta(float(df_data.loc[max_key, ["DispersionIndexCorners"]].iloc[0]))
y = np.log(np.abs((df['theta1'])/(1-(df['theta1']))))
# Add constant term for intercept
X = pd.DataFrame({
    'log_SUP': np.log(np.abs(df['SUP']) + 0.01),
  })


X = sm.add_constant(X)

# Fit model
model_theta = sm.OLS(y, X).fit()

#print(model.params)
######################################################################
# MONTE CARLO SIMULATIONS
################################################
#####################
league_code = leagues[max_key]
season_code = "2425"  # Current or upcoming season
odds_bet=odds_data(max_key,leagues,season_code)

results=process_all_matches(odds_bet)
data_testing = pd.concat([odds_bet,results], axis=1)
data_testing = data_testing.drop(["Lambda1","Lambda2"], axis=1)
#print(data_testing)
print(regression_covariates(data_testing).columns)

X=pd.DataFrame({
    'log_SUP': np.log(np.abs(data_testing['SUP']) + 0.01),
  })
X = sm.add_constant(X)

print(model_k.predict(X))


X=pd.DataFrame({
    'logitheta': np.log(np.abs(data_testing['SUP']) + 0.01),
  })
X = sm.add_constant(X)

print(model_theta.predict(X))


# Prepare regressors and target
X = pd.DataFrame({
    'log_TGi': np.log(df['TG']),
    'log_SUP': np.log(np.abs(df['SUP']) + 0.01),
    'TCTarget': df['TCTarget'],
    'log_HomeAvg3': np.log(df['HomeAvg3'] + 0.01),
    'log_AwayAvg3': np.log(df['AwayAvg3'] + 0.01),
    'log_HomeShotOnGoalAvg3': np.log(df['HomeShotOnGoalAvg3'] + 0.01),
    'log_AwayShotOnGoalAvg3': np.log(df['AwayShotOnGoalAvg3'] + 0.01)
})
X = sm.add_constant(X)

print(model_lambda.predict(X))

