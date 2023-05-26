# This file contains the main pre-processing steps for the code related to tactical player roles and role performance

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helpers.student_bif_code import *
from helpers.helperFunctions import *

# Load data from the database
df = load_db_to_pd(sql_query = "SELECT * FROM sd_tableF WHERE competitionId IN (335, 412, 635, 808, 795, 465);", db_name='Development')

# Remove unnecessary columns
df = df.drop(['ball_out', 'head_pass', 'loss', 'opportunity', 'conceded_postmatch_penalty',
              'teamId', 'competitionId',
              'conceded_goal', 'deep_completition', 'pass_into_penalty_area', 'pressing_duel', 'ground_duel'],
             axis=1)

# Add extra statistics columns
df['shots_PA'] = np.where(df['typePrimary'] == 'shot', non_pen_shots(df.x, df.y), 0)
df['shots_nonPA'] = np.where(df['typePrimary'] == 'shot', pen_shots(df.x, df.y), 0)
df['ws_cross'] = df.apply(lambda row: isWhiteSpaceCross('cross', row), axis=1)
df['hs_cross'] = df.apply(lambda row: isHalfSpaceCross('cross', row), axis=1)

# Create event zone dummies
df.insert(9, 'eventZone', df.apply(lambda row: zone(row), axis=1), allow_duplicates=True)
dfx = pd.get_dummies(df['eventZone'])
dfx['eventId'] = df['eventId']
df = pd.merge(df, dfx, on='eventId')

# Binary indicator for whether positions are in-possession or out-of-possession actions
df['posAction'] = df.apply(lambda row: possession_action(row), axis=1)
df['nonPosAction'] = df.apply(lambda row: non_possession_action(row), axis=1)

# Add VAEP statistic per related statistic from the event-based VAEP
# df.insert(52, 'Zone 0 Actions', 0, allow_duplicates=True) # ONLY use if no Zone 0
temp = df.iloc[:, np.r_[10:43, 46:59]].columns # update to include added stats
for i in temp:
    name = i + '_vaep'
    df[name] = np.where(df[i] != 0, df['sumVaep'], 0)

# Drop futher irrelevant columns
df = df.drop(['eventId',
                'x', 'y', 'end_x', 'end_y', 'eventZone',
               'typePrimary',
               'matchId',
               'offensiveValue', 'defensiveValue', 'sumVaep'],
             axis=1)

# Group dataframe per season
df = df[(df.playerId != 0)]
dfc = df.groupby(['playerId', 'seasonId'], as_index=False).sum()

# Merge dataframes (in case all leagues cannot be loaded at once)
frames = [] # run ONLY in the beginning
frames.append(dfc)
print(frames)
dfc = pd.concat(frames) # run ONLY in the end
dfc.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/tempframes.csv', index=False)

# Add position labels and remove goalkeepers
df_posmin = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Positions_Minutes", db_name='Scouting')
df_pos = df_posmin.drop(['matchId', 'teamId', 'time'], axis=1)
df_pos = df_pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
dfc = pd.merge(dfc, df_pos, on=['playerId', 'seasonId'])
dfc = dfc[dfc.position != 'gk']
dfc = dfc.drop(['position'], axis=1)

# Normalize data per 90 minutes played, incl. minutes played cutoff
df_min = df_posmin.drop(['matchId', 'teamId', 'position'], axis=1)
df_min = df_min.groupby(['playerId', 'seasonId'], as_index=False).sum()
df_min['games'] = df_min['time'] / 90
df_min = df_min.drop(['time'], axis=1)
dfc = pd.merge(dfc, df_min, on=['playerId', 'seasonId'])
dfc = dfc[dfc.games > 8] # cutoff games

dfc_id = dfc.iloc[:, np.r_[0, 1]]
dfc_norm = dfc.drop(['playerId', 'seasonId'], axis=1)

dfc_norm = dfc_norm.iloc[:, np.r_[0:94]].div(dfc_norm.games, axis=0) #update iloc if changes
dfc = pd.concat([dfc_id.reset_index(drop=True),dfc_norm.reset_index(drop=True)], axis=1)
test = dfc.describe()

# Create xA/xG ratios
dfc['xA_tendency'] = dfc['assist'] / dfc['xA'] # means assist-to-xA ratio but renamed for function purpose
dfc['xG_tendency'] = dfc['goal'] / dfc['xG'] # means goal-to-xG ratio but renamed for function purpose

# Switch counting statistics to opportunity space tendencies
temp = dfc.columns
dfc = opp_space(dfc, temp)
test2 = dfc.describe()

# Create copies
dfx = dfc.copy()
dfc = dfx.copy()

# Add ELO factor for VAEP statistics
dfc = dfc.sort_values('playerId')
dfc = dfc.reset_index(drop=True)
df_elo = load_db_to_pd(sql_query = "SELECT * FROM League_Factor", db_name='Scouting')
df_elo = df_elo.drop(['date'], axis=1)
dfc = pd.merge(dfc, df_elo, on=['seasonId'])
dfc_other = dfc.drop(dfc.filter(like='_vaep').columns, axis=1)
dfc_elo = dfc.iloc[:, np.r_[7:53]].mul(dfc.leagueFactor, axis=0) #update iloc if changes
dfc = pd.concat([dfc_other.reset_index(drop=True),dfc_elo.reset_index(drop=True)], axis=1)

# Standardize data (min-max scaling)
scale = MinMaxScaler()
dfc_id = dfc[['playerId', 'seasonId']]
#dfc = dfc.sort_values('playerId')
#dfc = dfc.reset_index(drop=True)
dfc.replace([np.inf, -np.inf], 0, inplace=True)
dfc_scaled = dfc.drop(['playerId', 'seasonId'], axis=1)
dfc_scaled[dfc_scaled.columns] = scale.fit_transform(dfc_scaled[dfc_scaled.columns])
check = dfc_scaled.describe()
dfc = pd.concat([dfc_id.reset_index(drop=True),dfc_scaled.reset_index(drop=True)], axis=1)

# Nan and outlier checks
dfc.fillna(value=0, inplace=True)
nan_check = pd.DataFrame(dfc.isna().sum())
outliers = find_outliers_IQR(dfc)
check = outliers.describe()

# Remove last irrelevant features
dfc = dfc.drop(['cross_tendency', 'cross_vaep', 'xG', 'xA', 'nonPosAction', 'posAction', 'leagueFactor',
                'Zone 0 Actions', 'Zone 0 Actions_vaep', 'penalty_foul_tendency', 'penalty_foul_vaep'], axis=1)

# Export cleaned dataframe
dfc.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv', index=False)