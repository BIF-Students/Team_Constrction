from archive.helpers.student_bif_code import *
from archive.helpers.helperFunctions import *
from sklearn.preprocessing import MinMaxScaler

df = load_db_to_pd(sql_query = "SELECT * FROM sd_table", db_name='Development')

# removing irrelevant stats (freekick/penalty, etc.) // or stats wo. data
df = df[(df.subEventName != 'Hand pass') & # update once SQL is updated
        (df.subEventName != 'Launch') &
        (df.subEventName != 'Touch')]

df = df.drop(['ball_out', 'head_pass', 'loss', 'opportunity', 'conceded_postmatch_penalty',
              'teamId', 'competitionId',
              'conceded_goal', 'deep_completition', 'pass_into_penalty_area', 'pressing_duel'],
             axis=1)

# creating event zone dummies
df.insert(10, 'eventZone', df.apply(lambda row: zone(row), axis=1), allow_duplicates=True)
dfx = pd.get_dummies(df['eventZone'])
dfx['eventId'] = df['eventId']
df = pd.merge(df, dfx, on='eventId')

# indicate pos/def action
df.insert(57, 'posAction', df.apply(lambda row: possession_action(row), axis=1), allow_duplicates=True)
df.insert(57, 'nonPosAction', df.apply(lambda row: non_possession_action(row), axis=1), allow_duplicates=True)

# extra stats placeholder (missing passes into pen, FT pass)
df['shots_PA'] = np.where(df['eventName'] == 'Shot', non_pen_shots(df.x, df.y), 0)
df['shots_nonPA'] = np.where(df['eventName'] == 'Shot', pen_shots(df.x, df.y), 0)
df['ws_cross'] = df.apply(lambda row: isWhiteSpaceCross('cross', row), axis=1)
df['hs_cross'] = df.apply(lambda row: isHalfSpaceCross('cross', row), axis=1)

# adding vaep pr related stat
temp = df.iloc[:, np.r_[11:45, 48:57, 59:63]].columns # update to include added stats
for i in temp:
    name = i + '_vaep'
    df[name] = np.where(df[i] != 0, df['sumVaep'], 0)

# grouping per season
dfc = df.drop(['eventId',
                'x', 'y', 'end_x', 'end_y', 'eventZone',
               'eventName', 'subEventName',
               'matchId',
               'offensiveValue', 'defensiveValue', 'sumVaep'],
             axis=1)
dfc = dfc[(dfc.playerId != 0)]
dfc = dfc.groupby(['playerId', 'seasonId'], as_index=False).sum()

# adding positions and removing GKs
df_posmin = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Positions_Minutes", db_name='Scouting')
df_pos = df_posmin.drop(['matchId', 'teamId', 'time'], axis=1)
df_pos = df_pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
dfc = pd.merge(dfc, df_pos, on=['playerId', 'seasonId'])
dfc = dfc[dfc.position != 'gk']
dfc = dfc.drop(['position'], axis=1)

# xA/xG ratios
dfc['xA_tendency'] = dfc['assist'] / dfc['xA'] # means assist-to-xA ratio but renamed for function purpose
dfc['xG_tendency'] = dfc['goal'] / dfc['xG'] # means goal-to-xG ratio but renamed for function purpose

# normalizing (per 90 incl. cutoff)
df_min = df_posmin.drop(['matchId', 'teamId', 'position'], axis=1)
df_min = df_min.groupby(['playerId', 'seasonId'], as_index=False).sum()
df_min['games'] = df_min['time'] / 90
df_min = df_min.drop(['time'], axis=1)
dfc = pd.merge(dfc, df_min, on=['playerId', 'seasonId'])
dfc = dfc[dfc.games > 8] # cutoff games

dfc_id = dfc.iloc[:, np.r_[0, 1]]
dfc_norm = dfc.drop(['playerId', 'seasonId'], axis=1)

dfc_norm = dfc_norm.iloc[:, np.r_[0:98]].div(dfc_norm.games, axis=0) #update iloc if changes
dfc = pd.concat([dfc_id.reset_index(drop=True),dfc_norm.reset_index(drop=True)], axis=1)

# switching counting stats to opportunity spaces
temp = dfc.columns
dfc = opp_space(dfc, temp)

# ELO factor for vaep
df_elo = load_db_to_pd(sql_query = "SELECT * FROM League_Factor", db_name='Scouting')
df_elo = df_elo.drop(['date'], axis=1)
dfc = pd.merge(dfc, df_elo, on=['seasonId'])
dfc_elo = dfc.iloc[:, np.r_[7:54]].mul(dfc.leagueFactor, axis=0) #update iloc if changes
dfc_other = dfc.drop(dfc.filter(like='_vaep').columns, axis=1)
dfc = pd.concat([dfc_other.reset_index(drop=True),dfc_elo.reset_index(drop=True)], axis=1)

# normalizing (min-max scaling)
scale = MinMaxScaler()
dfc.replace([np.inf, -np.inf], 0, inplace=True)
dfc_scaled = dfc.drop(['playerId', 'seasonId'], axis=1)
dfc_scaled[dfc_scaled.columns] = scale.fit_transform(dfc_scaled[dfc_scaled.columns])

check = dfc_scaled.describe()

dfc = pd.concat([dfc_id.reset_index(drop=True),dfc_scaled.reset_index(drop=True)], axis=1)

# nan and outlier checks
dfc.fillna(value=0, inplace=True)
nan_check = pd.DataFrame(dfc.isna().sum())
outliers = find_outliers_IQR(dfc)
check = outliers.describe()

# removing irrelevant stats
dfc = dfc.drop(['cross_vaep', 'xG', 'xA', 'nonPosAction', 'posAction', 'Zone 0 Actions', 'leagueFactor'], axis=1)

# export
dfc.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv', index=False)