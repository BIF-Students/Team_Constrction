from sklearn.preprocessing import MinMaxScaler
from archive.helpers.helperFunctions import *

# load event file
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Events - Copy.csv',
                 sep=";", encoding='unicode_escape')

# drop first columns and filling nan
df.drop(columns=df.columns[0], axis=1, inplace=True)
df = df.fillna(0)

print(df.shape)  # 73 columns at this point

# drop columns that won't be used, either due to irrelevance, captured elsewhere, or poor data
df = df.drop(['left_foot',
              'right_foot',
              'take_on_left',
              'take_on_right',
              'free_space_left',
              'free_space_right',
              'high_cross',
              'low_cross',
              'lost',
              'won',
              'neutral',
              'clearance',
              'fairplay',
              'direct',
              'indirect',
              'low',
              'low_right',
              'center',
              'left',
              'low_left',
              'right',
              'high',
              'high_left',
              'high_right',
              'miss_low_right',
              'miss_left',
              'miss_low_left',
              'miss_right',
              'miss_high',
              'miss_high_left',
              'miss_high_right',
              'post_low_right',
              'post_left',
              'post_low_left',
              'post_right',
              'post_high',
              'post_high_left',
              'post_high_right',
              'opportunity',
              'anticipation',
              'through',
              'missed_ball',
              'interception',
              'sliding_tackle',
              'red_card', 'yellow_card', 'second_yellow_card',
              'counter_attack',
              'dangerous_ball_lost',
              'blocked',
              'own_goal',
              'head',
              'feint',
              'matchPeriod',
              'eventSec',
              'subEventId',
              'eventId'],
             axis=1)

print(df.shape)  # 16 columns at this point

# drop irelevant sub events, either due to pure irrelevance (e.g. GK), non-role specificity, or poor data
df = df[df.subEventName != 0]
df = df[(df.eventName != 'Interruption') & (df.eventName != 'Free Kick') & (df.eventName != 'Offside')
        & (df.eventName != 'Save attempt') & (df.eventName != 'Goalkeeper leaving line')]
df = df[(df.subEventName != 'Hand pass') & (df.subEventName != 'Launch')
        & (df.subEventName != 'Late card foul') & (df.subEventName != 'Hand foul') & (df.subEventName != 'Out of game foul') & (df.subEventName != 'Time lost foul') & (df.subEventName != 'Violent Foul')
        & (df.subEventName != 'Protest') & (df.subEventName != 'Simulation') & (df.subEventName != 'Touch')]
df = df[df.playerId != 0]

# saving passing stats
passes = df[df.eventName == 'Pass']
passes.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/passes.csv', index=False)

# merge with matches dataset to get season ID
matches = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Matches.csv',
                      sep=";", encoding='unicode_escape')
df = pd.merge(df, matches, on='matchId')
compInd = df.pop('competitionId')
df.insert(5, 'competitionId', compInd, allow_duplicates=True)
sznInd = df.pop('seasonId')
df.insert(6, 'seasonId', sznInd, allow_duplicates=True)

print(df.shape)  # 4.2m, 18 rows/columns at this point

# rearranging columns
eIdInd = df.pop('id')
df.insert(6, 'eventNameId', eIdInd, allow_duplicates=True)
eNameInd = df.pop('eventName')
df.insert(6, 'eventName', eNameInd, allow_duplicates=True)

# append df with accuracy per sub event
'''temp = pd.unique(df['subEventName'])
for i in temp:
    name = i + '_acc'
    df[name] = np.where(df['subEventName'] == i, df['accurate'], np.nan)
for i in temp:
    name = i + '_acc'
    if df[name].isnull().all():
        df.drop(name, axis=1, inplace=True)'''

# defining event zones
'''df.insert(9, 'eventZone', df.apply(lambda row: findArea(row), axis=1), allow_duplicates=True)
for i in temp:
    name = i + '_zone'
    df[name] = np.where(df['subEventName'] == i, df['eventZone'], np.nan)'''

# adding shot, passing, and crossing distances
'''df['shot_distance'] = np.where(df['subEventName'] == 'Shot', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['passing_distance'] = np.where(df['eventName'] == 'Pass', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['cross_distance'] = np.where(df['subEventName'] == 'Cross', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)'''

# adding extra passing data
df['progressive_passes'] = np.where(df['eventName'] == 'Pass', pp(df.x, df.end_x), 0.00000)
'''df['progressive_passes_distance'] = np.where(df['progressive_passes'] == 1.00000, ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)'''
df['switches'] = np.where(df['eventName'] == 'Pass', switch(df.y, df.end_y), 0.00000)
'''df['pass_direction'] = np.where(df['eventName'] == 'Pass', direction(df.x, df.end_x), np.nan)'''
df['non_forward'] = np.where(df['eventName'] == 'Pass', non_forward(df.x, df.end_x), 0.00000)

# last third def duels
df['lThird_def_duels'] = np.where(df['subEventName'] == 'Ground defending duel', last_third_def(df.x), 0.00000)

# shot zones
df['shots_penArea'] = np.where(df['eventName'] == 'Shot', pen_shots(df.x, df.y), 0.00000)
df['shots_nonPenArea'] = np.where(df['eventName'] == 'Shot', non_pen_shots(df.x, df.y), 0.00000)

print(df.shape)  # 24 columns at this point

# creating dummies for pass direction and sub events and merging
'''temp2 = pd.unique(df['pass_direction'])
dfx = pd.get_dummies(df['pass_direction'])
dfx[temp2] = dfx[temp2].replace({'0': np.nan, 0: np.nan})
dfx['eventNameId'] = df['eventNameId']
dfx.pop('nan')
df = pd.merge(df, dfx, on='eventNameId')'''

temp = pd.unique(df['subEventName'])
dfx = pd.get_dummies(df['subEventName'])
dfx['eventNameId'] = df['eventNameId']
df = pd.merge(df, dfx, on='eventNameId')

print(df.shape)  # 37 columns at this point

# nuancing crosses
df['ws_cross'] = df.apply(lambda row: isWhiteSpaceCross('Cross', row), axis=1)
df['hs_cross'] = df.apply(lambda row: isHalfSpaceCross('Cross', row), axis=1)

# reshaping to player per season stats
df_sum = df.iloc[:, np.r_[0, 4, 12:39]]
df_other = df.iloc[:, np.r_[0:12]]
dfc = df_sum.groupby(['playerId', 'seasonId'], as_index=False).sum()

# creating accuracy percentages
'''temp3 = dfc.filter(regex='_acc$', axis=1)
temp3 = temp3.columns
temp3 = [x[:-4] for x in temp3]
for i in temp3:
    acc = i + '_acc'
    name = i + '_acc_percentage'
    dfc[name] = dfc[acc] / dfc[i] * 100'''

print(dfc.shape)  # 29 columns at this point

# exporting dataframe
dfc.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_cleaned.csv', index=False)

# importing cleaned df
dfc = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_cleaned.csv',
                 sep=",", encoding='unicode_escape')

# merging with played minutes
minutes = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Positions_Minutes.csv',
                 sep=";", encoding='unicode_escape')
minutes.pop('Unnamed: 0')
minutes.pop('teamId')
minutes.pop('matchId')
minutes.pop('position')
minutes = minutes.groupby(['playerId', 'seasonId'], as_index=False).sum()
dfc = pd.merge(dfc, minutes, on=['playerId', 'seasonId'])

# normalizing with per 90 and per pass
dfc = dfc[dfc.time > 720] # cutoff minutes
dfc['games'] = dfc['time'] / 90
df_id = dfc.iloc[:, np.r_[0, 1]]
df_norm = dfc.iloc[:, np.r_[2:31]]

df_norm = df_norm.iloc[:, np.r_[0:27]].div(df_norm.games, axis=0)
dfc = pd.concat([df_id.reset_index(drop=True),df_norm.reset_index(drop=True)], axis=1)

# stat ratios
dfc['ws_cross_tendency'] = (dfc['ws_cross'] / dfc['Cross']) * dfc['ws_cross']
dfc['hs_cross_tendency'] = (dfc['hs_cross'] / dfc['Cross']) * dfc['hs_cross']

dfc['safe_pass_tendency'] = (dfc['non_forward'] / (dfc['Cross'] + dfc['Head pass'] + dfc['High pass'] + dfc['Simple pass'] + dfc['Smart pass'])) * dfc['non_forward']
dfc['smart_pass_tendency'] = (dfc['Smart pass'] / (dfc['Cross'] + dfc['Head pass'] + dfc['High pass'] + dfc['Simple pass'] + dfc['Smart pass'])) * dfc['Smart pass']
dfc['switches_tendency'] = (dfc['switches'] / (dfc['Cross'] + dfc['Head pass'] + dfc['High pass'] + dfc['Simple pass'] + dfc['Smart pass'])) * dfc['switches']
dfc['simple_pass_tendency'] = (dfc['Simple pass'] / (dfc['Cross'] + dfc['Head pass'] + dfc['High pass'] + dfc['Simple pass'] + dfc['Smart pass']))* dfc['Simple pass']
dfc['key_pass_tendency'] = (dfc['key_pass'] / (dfc['Cross'] + dfc['Head pass'] + dfc['High pass'] + dfc['Simple pass'] + dfc['Smart pass'])) * dfc['key_pass']
dfc['pp_tendency'] = (dfc['progressive_passes'] / (dfc['Cross'] + dfc['Head pass'] + dfc['High pass'] + dfc['Simple pass'] + dfc['Smart pass'])) * dfc['progressive_passes']

dfc['ptp_ratio'] = dfc['progressive_passes'] / (dfc['progressive_passes'] + dfc['Acceleration'])
dfc['ptc_ratio'] = dfc['Acceleration'] / (dfc['progressive_passes'] + dfc['Acceleration'])

dfc['anticipation_ratio'] = dfc['anticipated'] / dfc['Ground defending duel']
dfc['foul_tendency'] = (dfc['Foul'] / (dfc['Ground loose ball duel'] + dfc['Ground defending duel'])) * dfc['Foul']
dfc['air_duel_tendency'] = (dfc['Air duel'] / (dfc['Ground loose ball duel'] + dfc['Ground defending duel'] + dfc['Air duel'])) * dfc['Air duel']
dfc['gd_duel_tendency'] = (dfc['Ground defending duel'] / (dfc['Ground loose ball duel'] + dfc['Ground defending duel'] + dfc['Air duel'])) * dfc['Ground defending duel']
dfc['aThird_duel_tendency'] = (dfc['lThird_def_duels'] / (dfc['Ground loose ball duel'] + dfc['Ground defending duel'] + dfc['Air duel'])) * dfc['lThird_def_duels']

dfc['nonPA_shots_tendency'] = (dfc['shots_nonPenArea'] / dfc['Shot']) * dfc['shots_nonPenArea']
dfc['PA_shots_tendency'] = (dfc['shots_penArea'] / dfc['Shot']) * dfc['shots_penArea']

dfc = dfc.fillna(0)

# deselecting columns no longer for use
dfc = dfc.drop(['key_pass', 'progressive_passes', 'switches', 'non_forward', 'Head pass', 'High pass', 'Simple pass', 'Smart pass',
                'ws_cross', 'hs_cross', 'Cross',
                'anticipated', 'Foul', 'Ground loose ball duel', 'lThird_def_duels', 'Air duel', 'Ground defending duel',
                'accurate', 'not_accurate',
                'Acceleration',
                'shots_penArea', 'shots_nonPenArea', 'Shot'],
             axis=1)

# starting position merging and cleaning - saving IDs and merging with positions
pos = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Positions_Minutes.csv', sep=";", encoding='unicode_escape')
pos2 = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Simple_Positions.csv', sep=";", encoding='unicode_escape')
pos2.drop(columns=pos2.columns[0], axis=1, inplace=True)
pos2 = pos2.drop(['pos_group', 'radar_group'], axis=1)
pos = pd.merge(pos, pos2, on=['position'])
pos.drop(columns=pos.columns[0], axis=1, inplace=True)
pos = pos.drop(['matchId', 'teamId', 'position', 'time'], axis=1)

pos = pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
pos.insert(3, 'pos_group', pos.apply(lambda row: pos_group(row), axis=1), allow_duplicates=True)
dfc = pd.merge(dfc, pos, on=['playerId', 'seasonId'])

# removing goalkeepers
dfc = dfc[dfc.map_group != 'GK']
nan_check = pd.DataFrame(dfc.isna().sum())

# further normalization - scaling
scale = MinMaxScaler()
dfc.replace([np.inf, -np.inf], 0, inplace=True)
dfc_id = dfc.iloc[:, np.r_[0:2, 23:25]]
dfc_scale = dfc.iloc[:, np.r_[2:23]]
dfc_scaled = dfc_scale.copy()

dfc_scaled[dfc_scaled.columns] = scale.fit_transform(dfc_scaled[dfc_scaled.columns])
check = dfc_scaled.describe()
dfc = pd.concat([dfc_id.reset_index(drop=True),dfc_scaled.reset_index(drop=True)], axis=1)

# outliers
outliers = find_outliers_IQR(dfc)
check = outliers.describe()

# exporting UMAP formatted file
dfc.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv', index=False)