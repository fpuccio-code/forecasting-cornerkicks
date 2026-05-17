import pandas as pd

def data_set(leagues, seasons):
    summary_df = pd.DataFrame()

    for league_name, league_code in leagues.items():
        total_df = pd.DataFrame()

        for season_code in seasons.values():
            url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
            try:
                print(f"Loading: {url}")
                df = pd.read_csv(url)
                df['League'] = league_name
                df['Season'] = season_code
                total_df = pd.concat([total_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error loading {league_name} {season_code}: {e}")
                continue

        if not total_df.empty:
            # Ensure required columns exist or add them
            for col in ['HC', 'AC', 'FTHG', 'FTAG']:
                if col not in total_df.columns:
                    total_df[col] = pd.NA

            # Compute total corners and goals
            total_df['TotalCorners'] = pd.to_numeric(total_df['HC'], errors='coerce') + pd.to_numeric(total_df['AC'], errors='coerce')
            total_df['TotalGoals'] = pd.to_numeric(total_df['FTHG'], errors='coerce') + pd.to_numeric(total_df['FTAG'], errors='coerce')

            # Over/Under 2.5 indicators
            total_df['Over2.5'] = total_df['TotalGoals'] > 2.5
            total_df['Under2.5'] = total_df['TotalGoals'] <= 2.5

            # Summary stats
            avg_corners = total_df['TotalCorners'].mean()
            std_corners = total_df['TotalCorners'].std()
            avg_goals = total_df['TotalGoals'].mean()
            std_goals = total_df['TotalGoals'].std()
            over_pct = total_df['Over2.5'].mean() * 100
            under_pct = total_df['Under2.5'].mean() * 100

            # Betting odds columns (optional)
            avg_odds_over = pd.to_numeric(total_df.get('Avg>2.5', pd.Series(dtype=float)), errors='coerce').mean()
            avg_odds_under = pd.to_numeric(total_df.get('Avg<2.5', pd.Series(dtype=float)), errors='coerce').mean()

            # Fill summary table
            summary_df.loc[league_name, 'AvgCorners'] = avg_corners
            summary_df.loc[league_name, 'SigmaCorners'] = std_corners
            summary_df.loc[league_name, 'DispersionIndexCorners'] = (std_corners**2) / avg_corners if avg_corners != 0 else None
            summary_df.loc[league_name, 'AvgGoals'] = avg_goals
            summary_df.loc[league_name, 'SigmaGoals'] = std_goals
            summary_df.loc[league_name, 'DispersionIndexGoals'] = (std_goals**2) / avg_goals if avg_goals != 0 else None
        #    summary_df.loc[league_name, 'Over2.5_%'] = over_pct
         #    summary_df.loc[league_name, 'Under2.5_%'] = under_pct
         #   summary_df.loc[league_name, 'AvgOddsOver2.5'] = avg_odds_over
         #   summary_df.loc[league_name, 'AvgOddsUnder2.5'] = avg_odds_under

    summary_df.index.name = 'League'
    return summary_df


################################################
#Upload odds, season 2024-2025 of "key_leage" (input)
################################################
def odds_data(key_leage,leagues,season_code):
    max_key=key_leage
    league_code = leagues[max_key]
#    season_code = "2324"  # Current or upcoming season
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"

    try:
        df = pd.read_csv(url)
        # Set index as Date_HomeTeam_AwayTeam
        if all(col in df.columns for col in ['Date', 'HomeTeam', 'AwayTeam']):
            df['MatchID'] = df['Date'].astype(str) + '_' + df['HomeTeam'].astype(str) + '_' + df['AwayTeam'].astype(str)
            df.set_index('MatchID', inplace=True)
        else:
            print("Warning: Date, HomeTeam or AwayTeam column missing. Cannot set match-based index.")
    
        odds_cols = ['AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5', 'AvgC>2.5', 'AvgC<2.5']
        existing_cols = [col for col in odds_cols if col in df.columns]
        odd_data = df[existing_cols]
        
    except Exception as e:
        print(f"Error loading data for {max_key} season {season_code}: {e}")

    return odd_data


##############################################################################
############################################################################
