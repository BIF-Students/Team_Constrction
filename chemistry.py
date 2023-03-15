# Import necessary modules
from helpers.student_bif_code import load_db_to_pd  # custom module
import pandas as pd
from collections import defaultdict
from helpers.helperFunctions import *

# Load data from a SQL database table into a pandas DataFrame
df = load_db_to_pd(sql_query="select * from sd_table", db_name='Development')

df_def = df
df_pos = df

df_def.insert(57, 'nonPosAction', df_def.apply(lambda row: non_possession_action(row), axis=1), allow_duplicates=True) # Mark if action is defensive
df_def = df_def[df_def['nonPosAction'] == 1] # Filter dataframe for defensive actions

matches_positions = {}

tester = {}
tester[(1,2)] = {'matchId': 10, 'playerId': 12, 'x': [10,20,30], 'y': [30,40,50]}

tester2 = {}

tester2[(1,2)] = {'matchId': tester[(1,2)]['matchId'], 'playerId': tester[(1,2)]['playerId'], 'avg_x': sum(tester[(1,2)]['x'])/len(tester[(1,2)]['x']), 'avg_y': sum(tester[(1,2)]['y'])/len(tester[(1,2)]['y'])}

tester[(1,2)]['matchId']


def allocate_position(row):
    x = row['x']
    y = row['y']
    mId = row['matchId']
    pId = row['playerId']
    if (mId, pId ) in matches_positions:
        matches_positions[(mId, pId)]['x'].append(x)
        matches_positions[(mId, pId)]['y'].append(y)
    else: matches_positions[(mId, pId)] = {'matchId': mId , 'playerId': pId , 'x': [x], 'y': [y]}

df_def.apply(lambda row:  allocate_position(row), axis = 1)

def get_average_positions(key_values, newDict):
    for key in key_values:
        newDict[key] = {'matchId': key_values[key]['matchId'],
                        'playerId': key_values[key]['playerId'],
                        'avg_x': sum(key_values[key]['x'])/len(key_values[key]['x']),
                        'avg_y': sum(key_values[key]['y'])/len(key_values[key]['y'])}
    return newDict

avg_position_match_df = pd.DataFrame.from_dict(get_average_positions(matches_positions), orient='index', columns=['matchId', 'playerId', 'avg_x', 'avg_y'])




# Define a class called 'EventStats' with five attributes
class EventStats:
    def __init__(self, combinedVaep, eId1, eId2, t1, t2):
        self.combinedVaep = combinedVaep
        self.eId1 = eId1
        self.eId2 = eId2
        self.t1 = t1
        self.t2 = t2

# Define a function called 'makeKey' that takes two integer parameters and returns a string
def makeKey(id1, id2):
    return str(id1) + str(id2) if id1 < id2 else str(id2) + str(id1)

# Create a default dictionary to store EventStats objects for each match and key combination
matches = defaultdict(dict)

# Iterate over the DataFrame, creating EventStats objects for each row and the following row
for i, row in df[:-1].iterrows():
    mId = row['matchId']  # get the match ID for the current row
    key = makeKey(row['playerId'], df.loc[i + 1, 'playerId'])  # create a key based on player IDs
    t1 = int(row['teamId'])  # get the team ID for the current row
    t2 = int(df.loc[i + 1, 'teamId'])  # get the team ID for the following row
    e1 = int(row['eventId'])  # get the event ID for the current row
    e2 = int(df.loc[i + 1, 'eventId'])  # get the event ID for the following row
    p1 = int(row['playerId'])  # get the player ID for the current row
    p2 = int(df.loc[i + 1, 'playerId'])  # get the player ID for the following row

    vaepSum = row['sumVaep'] + df.loc[i + 1, 'sumVaep']  # calculate the sum of the VAEP values for the two events
    # If the two events are on the same team and involve different players, add them to the matches dictionary
    if t1 == t2 and p1 != p2:
        if key in matches[mId]:
            matches[mId][key].append(EventStats(vaepSum, e1, e2, t1, t2))
        else:
            matches[mId][key] = [EventStats(vaepSum, e1, e2, t1, t2)]


def compute_net_oi_game(df):
    cols = [f'zone_{i}_net_oi' for i in range(1, 10)]
    expected_cols = [f'zone_{i}_expected_vaep' for i in range(1, 10)]
    for col, expected_col in zip(cols, expected_cols):
        df[col] = df[col.replace('_net_oi', '')] - df[expected_col]




def compute_running_avg_team_vaep (row, label):
    if row['games_played'] == 0:
        return row[label]
    else:
        return row[label] / (row['games_played'] +1)

def find_zones_and_vaep(df):
    # create dummy variables for the zone column
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    # multiply the dummy variables by the sumVaep column
    zone_vaep = zone_dummies.mul(df['sumVaep'], axis=0)

    # concatenate the original dataframe with the zone_vaep dataframe
    df = pd.concat([df, zone_vaep], axis=1)

    return df

def team_vaep_game(df):
    df = df.groupby(['matchId', 'teamId'])['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'zone_7', 'zone_8', 'zone_9'].sum().reset_index()
    df = df.sort_values(by=['teamId', 'matchId'])
    df[['zone_1_cumsum', 'zone_2_cumsum',
        'zone_3_cumsum', 'zone_4_cumsum',
        'zone_5_cumsum', 'zone_6_cumsum',
        'zone_7_cumsum', 'zone_8_cumsum', 'zone_9_cumsum'
        ]] = df.groupby('teamId')[['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'zone_7', 'zone_8', 'zone_9']].cumsum()

    df['games_played'] = df.groupby(['teamId']).cumcount()
    df['zone_1_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_1_cumsum') ,axis = 1)
    df['zone_2_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_2_cumsum') ,axis = 1)
    df['zone_3_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_3_cumsum') ,axis = 1)
    df['zone_4_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_4_cumsum') ,axis = 1)
    df['zone_5_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_5_cumsum') ,axis = 1)
    df['zone_6_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_6_cumsum') ,axis = 1)
    df['zone_7_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_7_cumsum') ,axis = 1)
    df['zone_8_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_8_cumsum') ,axis = 1)
    df['zone_9_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_9_cumsum') ,axis = 1)
    return df

















compute_net_oi_game(expected_vaep_found)