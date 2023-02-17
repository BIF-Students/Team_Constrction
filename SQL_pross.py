from helpers.student_bif_code import *
from helpers.helperFunctions import *
from sklearn.preprocessing import MinMaxScaler

df = load_db_to_pd(sql_query = "SELECT * FROM table_sd", db_name='Development')

# removing irrelevant stats (freekick/penalty, etc.)
df = df[(df.subEventName != 'Hand pass') & # update once SQL is updated
        (df.subEventName != 'Launch') &
        (df.subEventName != 'Touch')]

# creating event zone dummies
df.insert(7, 'eventZone', df.apply(lambda row: zone(row), axis=1), allow_duplicates=True)
dfx = pd.get_dummies(df['eventZone'])
dfx['eventId'] = df['eventId']
df = pd.merge(df, dfx, on='eventId')

# indicate pos/def action
df.insert(64, 'posAction', df.apply(lambda row: possession_action(row), axis=1), allow_duplicates=True)
df.insert(65, 'defAction', df.apply(lambda row: defensive_action(row), axis=1), allow_duplicates=True)

# extra stats placeholder (missing passes into pen, FT pass)
df['shots_PA'] = np.where(df['eventName'] == 'Shot', non_pen_shots(df.x, df.y), 0.00000)
df['shots_nonPA'] = np.where(df['eventName'] == 'Shot', pen_shots(df.x, df.y), 0.00000)
df['ws_cross'] = df.apply(lambda row: isWhiteSpaceCross('Cross', row), axis=1)
df['hs_cross'] = df.apply(lambda row: isHalfSpaceCross('Cross', row), axis=1)

# adding vaep pr related stat
temp = df.iloc[:, np.r_[9:52]].columns # update to include added stats
for i in temp:
    name = i + '_vaep'
    df[name] = np.where(df[i] == 1, df['sumVaep'], 0)

# grouping per season
dfc = df.drop(['eventId',
                'x', 'y',
                'end_x', 'end_y',
                'eventZone',
              'subEventName',
               'offensiveValue', 'defensiveValue', 'sumVaep',
               'preds'],
             axis=1)
dfc = dfc.groupby(['playerId', 'seasonId'], as_index=False).sum()

# switching counting stats to opportunity spaces
temp = dfc.columns
dfc = opp_space(dfc, temp)

# adding positions and removing GKs
'''df_pos = load_db_to_pd(sql_query = "SELECT * FROM table_sd", db_name='Development') # update to position (or avg pos) table
dfc = pd.merge(dfc, df_pos, on=['playerId', 'seasonId'])
dfc = dfc[dfc.map_group != 'GK']'''

# normalizing (per 90 incl. cutoff + min-max scaling)
'''df_min = load_db_to_pd(sql_query = "SELECT * FROM table_sd", db_name='Development') # update to games played per season table
dfc = pd.merge(dfc, df_min, on=['playerId', 'seasonId'])
dfc = dfc[dfc.games > 8] # cutoff games

dfc_id = dfc.iloc[:, np.r_[0, 1]]
dfc_norm = df.drop(['playerId', 'seasonId', 'pos'], axis=1)

dfc_norm = dfc_norm.iloc[:, np.r_[0:27]].div(dfc_norm.games, axis=0) #update iloc
scale = MinMaxScaler()
dfc.replace([np.inf, -np.inf], 0, inplace=True)
dfc_scaled = dfc_norm.copy()
dfc_scaled[dfc_scaled.columns] = scale.fit_transform(dfc_scaled[dfc_scaled.columns])
check = dfc_scaled.describe()

dfc = pd.concat([dfc_id.reset_index(drop=True),dfc_scaled.reset_index(drop=True)], axis=1)'''

# nan and outlier checks
nan_check = pd.DataFrame(dfc.isna().sum())
outliers = find_outliers_IQR(dfc)
check = outliers.describe()

# export
dfc.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv', index=False)