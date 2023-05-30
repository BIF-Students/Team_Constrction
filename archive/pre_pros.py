from sklearn.preprocessing import MinMaxScaler
from archive.helpers.helperFunctions import *

# load event file
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Events - Copy.csv',
                 sep=";",
                 encoding='unicode_escape')

# drop first columns
df.drop(columns=df.columns[0], axis=1, inplace=True)

print(df.shape)  # 73 columns at this point

# drop columns that won't be used
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
              'blocked',
              'eventId'],
             axis=1)

print(df.shape)  # 33 columns at this point

# drop irelevant sub events
df = df[df.subEventName.notnull()]
df = df[df.subEventName != 'Whistle']
df = df[df.playerId != 0]

# merge with matches dataset to get season ID
matches = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Matches.csv',
                      sep=";",
                      encoding='unicode_escape')
df = pd.merge(df, matches, on='matchId')
compInd = df.pop('competitionId')
df.insert(5, 'competitionId', compInd, allow_duplicates=True)
sznInd = df.pop('seasonId')
df.insert(6, 'seasonId', sznInd, allow_duplicates=True)

print(df.shape)  # 35 columns at this point

# saving passing stats
passes = df[df.eventName == 'Pass']
passes.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/passes.csv', index=False)

# rearranging columns
eIdInd = df.pop('id')
df.insert(6, 'eventNameId', eIdInd, allow_duplicates=True)
eNameInd = df.pop('eventName')
df.insert(6, 'eventName', eNameInd, allow_duplicates=True)
subEventInd = df.pop('subEventId')
df.insert(7, 'subEventId', subEventInd, allow_duplicates=True)
subEventNameInd = df.pop('subEventName')
df.insert(8, 'subEventName', subEventNameInd, allow_duplicates=True)

# append df with accuracy per sub event
temp = pd.unique(df['subEventName'])
for i in temp:
    name = i + '_acc'
    df[name] = np.where(df['subEventName'] == i, df['accurate'], np.nan)
for i in temp:
    name = i + '_acc'
    if df[name].isnull().all():
        df.drop(name, axis=1, inplace=True)

df_p = df[['subEventName', 'accurate', 'Cross_acc']]  # test
print(df.shape)  # 57 columns at this point

# defining event zones
df.insert(9, 'eventZone', df.apply(lambda row: findArea(row), axis=1), allow_duplicates=True)
for i in temp:
    name = i + '_zone'
    df[name] = np.where(df['subEventName'] == i, df['eventZone'], np.nan)

# adding shot, passing, and crossing distances
df['shot_distance'] = np.where(df['subEventName'] == 'Shot', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['passing_distance'] = np.where(df['eventName'] == 'Pass', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)
df['cross_distance'] = np.where(df['subEventName'] == 'Cross', ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)

# adding progressive passes
df['progressive_passes'] = np.where(df['eventName'] == 'Pass', pp(df.x, df.end_x), np.nan)
df['progressive_passes_distance'] = np.where(df['progressive_passes'] == 1.00000,
                                             ec(df['x'], df['end_x'], df['y'], df['end_y']), np.nan)

# adding extra passes types
df['pass_direction'] = np.where(df['eventName'] == 'Pass', direction(df.x, df.end_x), np.nan)

# adding swtiches
df['switches'] = np.where(df['eventName'] == 'Pass', switch(df.y, df.end_y), np.nan)

print(df.shape)  # 100 columns at this point

# creating dummies for pass direction and sub events and merging
temp2 = pd.unique(df['pass_direction'])
dfx = pd.get_dummies(df['pass_direction'])
dfx[temp2] = dfx[temp2].replace({'0': np.nan, 0: np.nan})
dfx['eventNameId'] = df['eventNameId']
dfx.pop('nan')
df = pd.merge(df, dfx, on='eventNameId')

print(df['backward'].value_counts(), '\n \n', df['horizontal'].value_counts(), '\n \n',
      df['forward'].value_counts())  # test

dfx = pd.get_dummies(df['subEventName'])
dfx[temp] = dfx[temp].replace({'0': np.nan, 0: np.nan})
dfx['eventNameId'] = df['eventNameId']
df = pd.merge(df, dfx, on='eventNameId')

print(df.shape)  # 138 columns at this point

# saving df raw
df_raw = df.copy()  # revert by df=df_raw.copy()

# reshaping to player per season stats
df_sum = df.iloc[:, np.r_[0, 4, 16:58, 96, 99:138]]
df_freq = df.iloc[:, np.r_[0, 1, 4, 58:93]]
df_avg = df.iloc[:, np.r_[0, 4, 93:96, 97]]
df_other = df.iloc[:, np.r_[0, 2:16]]

df_sum = df_sum.groupby(['playerId', 'seasonId'], as_index=False).sum()
df_freq = gmode(df_freq)
df_avg = df_avg.groupby(['playerId', 'seasonId'], as_index=False).mean()

dfc = pd.merge(df_sum, df_freq, on=['playerId', 'seasonId'])
dfc = pd.merge(dfc, df_avg, on=['playerId', 'seasonId'])

# creating accuracy percentages
temp3 = dfc.filter(regex='_acc$', axis=1)
temp3 = temp3.columns
temp3 = [x[:-4] for x in temp3]
for i in temp3:
    acc = i + '_acc'
    name = i + '_acc_percentage'
    dfc[name] = dfc[acc] / dfc[i] * 100

print(dfc.shape)  # 146 columns at this point

# exporting dataframe
dfc.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_cleaned.csv', index=False)

# importing cleaned df
dfc = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_cleaned.csv',
                 sep=",",
                 encoding='unicode_escape')

# stat ratios
dfc['goals_pr_shot'] = dfc['goal'] / dfc['Shot']
dfc['pp_pass_ratio'] = dfc['progressive_passes'] / (dfc['Cross'] + dfc['Hand pass'] + dfc['Head pass'] + dfc['High pass'] + dfc['Launch'] + dfc['Simple pass'] + dfc['Smart pass'])
dfc['key_pass_ratio'] = dfc['key_pass'] / (dfc['Cross'] + dfc['Hand pass'] + dfc['Head pass'] + dfc['High pass'] + dfc['Launch'] + dfc['Simple pass'] + dfc['Smart pass'])
dfc['anticipation_percentage'] = dfc['anticipation'] / dfc['Ground defending duel']
dfc['slide_tackle_ratio'] = dfc['sliding_tackle'] / (dfc['Ground loose ball duel'] + dfc['Ground defending duel'])

# merging with played minutes
minutes = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Positions_Minutes.csv',
                 sep=";",
                 encoding='unicode_escape')
minutes.pop('Unnamed: 0')
minutes.pop('teamId')
minutes.pop('matchId')
minutes.pop('position')
minutes = minutes.groupby(['playerId', 'seasonId'], as_index=False).sum()
dfc = pd.merge(dfc, minutes, on=['playerId', 'seasonId'])

# merging with passes
passes = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/passes.csv',
                 sep=",",
                 encoding='unicode_escape')
passes = passes[['playerId', 'seasonId']]
passes['counts'] = 1.0
passes = passes.groupby(['playerId', 'seasonId'], as_index=False).sum()

# normalizing with per 90 and per pass
dfc = dfc[dfc.time > 720] # cutoff minutes
dfc['games'] = dfc['time'] / 90
df_id = dfc.iloc[:, np.r_[0, 1]]
df_norm = dfc.iloc[:, np.r_[0, 1, 2:83, 156]]
df_none = dfc.iloc[:, np.r_[0, 1, 84:155]]

df_norm = pd.merge(df_norm, passes, on=['playerId', 'seasonId'])
df_norm = df_norm.iloc[:,2:83].div(df_norm.games, axis=0).div(df_norm.counts, axis=0)
df_norm = pd.concat([df_id.reset_index(drop=True),df_norm.reset_index(drop=True)], axis=1)

dfn = pd.merge(df_none, df_norm, on=['playerId', 'seasonId'])

# filling nan
dfn = dfn.fillna(0)

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
dfn = pd.merge(dfn, pos, on=['playerId', 'seasonId'])

# removing goalkeepers
dfn = dfn[dfn.map_group != 'GK']
nan_check = pd.DataFrame(dfn.isna().sum())
dfn_pos = dfn[['playerId', 'seasonId', "map_group", "pos_group"]]

# further normalization - scaling
scale = MinMaxScaler()
dfn.replace([np.inf, -np.inf], 0, inplace=True)
dfn_id = dfn.iloc[:, np.r_[0:3, 154:156]]
dfn_scale = dfn.iloc[:, np.r_[3:154]]
dfn_scaled = dfn_scale.copy()

dfn_scaled[dfn_scaled.columns] = scale.fit_transform(dfn_scaled[dfn_scaled.columns])
check = dfn_scaled.describe()
dfn = pd.concat([dfn_id.reset_index(drop=True),dfn_scaled.reset_index(drop=True)], axis=1)

# removing unwanted columns - accurate stats (duplicate), goalkeeper stats, irrelevant zones, irrelevant accuracy %, irrelevant stats for a player's role (not necessarily clustering)
dfn = dfn.drop(['Simple pass_acc', 'Clearance_acc', 'Air duel_acc', 'Cross_acc', 'Launch_acc', 'Ground attacking duel_acc', 'Head pass_acc', 'Ground loose ball duel_acc', 'Ground defending duel_acc', 'Free Kick_acc', 'High pass_acc', 'Throw in_acc', 'Smart pass_acc', 'Free kick cross_acc', 'Save attempt_acc', 'Acceleration_acc', 'Corner_acc', 'Shot_acc', 'Reflexes_acc', 'Hand pass_acc', 'Free kick shot_acc', 'Penalty_acc',
              'Goal kick', 'Goalkeeper leaving line', 'Hand pass', 'Reflexes', 'Save attempt',
              'Ball out of the field_zone', 'Free Kick_zone', 'Hand foul_zone', 'Throw in_zone', 'Free kick cross_zone', 'Save attempt_zone', 'Goal kick_zone', 'Corner_zone', 'Goalkeeper leaving line_zone', 'Reflexes_zone', 'Hand pass_zone', 'Free kick shot_zone', 'Protest_zone', 'Late card foul_zone', 'Penalty_zone', 'Time lost foul_zone', 'Out of game foul_zone', 'Violent Foul_zone', 'Simulation_zone',
              'Throw in_acc_percentage', 'Save attempt_acc_percentage', 'Reflexes_acc_percentage', 'Hand pass_acc_percentage',
                'own_goal', 'Ball out of the field', 'Hand foul', 'Out of game foul', 'Protest', 'Simulation', 'Throw in', 'Time lost foul', 'Launch', 'Launch_zone', 'Launch_acc_percentage', 'head', 'feint', 'Touch', 'Touch_zone', 'Clearance_acc_percentage'],
             axis=1)

# removing unwanted columns for UMAP - assigned stats (pen, free-kick, corners)
dfn = dfn.drop(['Penalty', 'Penalty_acc_percentage',
                'Free kick shot', 'Free kick shot_acc_percentage', 'Free kick cross', 'Free kick cross_acc_percentage', 'Free Kick', 'Free Kick_acc_percentage',
                'Corner', 'Corner_acc_percentage'],
             axis=1)

# outliers
outliers = find_outliers_IQR(dfn)
check = outliers.describe()
dfn = dfn.drop(['Shot_zone', 'Offside_zone', 'shot_distance',
              'Simple pass_zone', 'Clearance_zone', 'Air duel_zone', 'Cross_zone', 'Ground attacking duel_zone', 'Head pass_zone', 'Ground loose ball duel_zone', 'Ground defending duel_zone', 'High pass_zone', 'Smart pass_zone', 'Foul_zone', 'Acceleration_zone',
                'cross_distance', 'Acceleration_acc_percentage', 'counter_opportunity_ratio', 'missed_ball', 'Offside'], axis=1)

check = dfn.describe()

# exporting UMAP formatted file
dfn.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv', index=False)