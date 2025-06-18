
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
def regression_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Step 1: Parse Date, HomeTeam, AwayTeam
    df['Date'] = pd.to_datetime(df.index.str.split('_').str[0], dayfirst=True)
    df['HomeTeam'] = df.index.str.split('_').str[1]
    df['AwayTeam'] = df.index.str.split('_').str[2]
    
    df = df.sort_values('Date')

    # Step 2: Rolling HomeTeam stats
    home_rolling = (
        df[['HomeTeam', 'Date', 'AvgH']]
        .rename(columns={'HomeTeam': 'Team'})
        .sort_values(['Team', 'Date'])
        .groupby('Team')
        .rolling(window=3, min_periods=1, on='Date')
        .mean()
        .reset_index()
        .rename(columns={'AvgH': 'HomeAvg3'})
    )

    # Step 3: Rolling AwayTeam stats
    away_rolling = (
        df[['AwayTeam', 'Date', 'AvgA']]
        .rename(columns={'AwayTeam': 'Team'})
        .sort_values(['Team', 'Date'])
        .groupby('Team')
        .rolling(window=3, min_periods=1, on='Date')
        .mean()
        .reset_index()
        .rename(columns={'AvgA': 'AwayAvg3'})
    )

    # Step 4: Merge back into original df
    df = df.reset_index()
    df = df.merge(home_rolling[['Team', 'Date', 'HomeAvg3']], 
                  left_on=['HomeTeam', 'Date'], right_on=['Team', 'Date'], how='left')
    df = df.merge(away_rolling[['Team', 'Date', 'AwayAvg3']], 
                  left_on=['AwayTeam', 'Date'], right_on=['Team', 'Date'], how='left')

    df.drop(columns=['Team_x', 'Team_y'], inplace=True)

    # Step 5: Fill missing with global mean
    df['HomeAvg3'] = df['HomeAvg3'].fillna(df['AvgH'].mean())
    df['AwayAvg3'] = df['AwayAvg3'].fillna(df['AvgA'].mean())

    # Optional: Same logic for other features like AvgC>2.5, Avg>2.5
    df['HomeShotOnGoalAvg3'] = df['AvgC>2.5'].rolling(window=3, min_periods=1).mean()
    df['AwayShotOnGoalAvg3'] = df['Avg>2.5'].rolling(window=3, min_periods=1).mean()
    df['HomeShotOnGoalAvg3'] = df['HomeShotOnGoalAvg3'].fillna(df['AvgC>2.5'].mean())
    df['AwayShotOnGoalAvg3'] = df['AwayShotOnGoalAvg3'].fillna(df['Avg>2.5'].mean())

    # Dummy target encoding — replace with real logic if available
    df['TCTarget'] = df['AvgC>2.5'].mean()

    # Restore index if needed
    df = df.set_index('MatchID') if 'MatchID' in df.columns else df.set_index(df.columns[0])

    return df

# # Rercursi framework
# def regression_covariates(df: pd.DataFrame):
#     df = df.sort_index()
#     # Parse date and teams
#     df['Date'] = pd.to_datetime(df.index.str.split('_').str[0], dayfirst=True)
#     df['HomeTeam'] = df.index.str.split('_').str[1]
#     df['AwayTeam'] = df.index.str.split('_').str[2]

#     # For missing values, impute by league average or grand average:
#     df['HomeAvg3'] = df['AvgH'].fillna(df['AvgH'].tail(3).mean())
#     df['AwayAvg3'] = df['AvgA'].fillna(df['AvgA'].tail(3).mean())
#     df['HomeShotOnGoalAvg3'] = df['AvgC>2.5'].fillna(df['AvgC>2.5'].tail(3).mean())  # Replace with correct column
#     df['AwayShotOnGoalAvg3'] = df['Avg>2.5'].fillna(df['Avg>2.5'].tail(3).mean())   # Replace with correct column
#     # Target encoding TCTarget - you need to compute or provide this column
#     # For now, let's create a dummy column:
#     df['TCTarget'] = df['AvgC>2.5'].mean()  # Replace with actual target encoding calculation

#     return df





# Shape parameter function (NegBin, GeomPois)
def shape_parameter_k(mu, std):
    Var=std**2
    shape=(mu**2)/(Var-mu)
    return shape

def shape_parameter_theta(D):
    theta=2/(D+1)
    return theta

