# Import necessary modules
from helpers.student_bif_code import load_db_to_pd  # custom module
import pandas as pd
from collections import defaultdict
from helpers.helperFunctions import *
import math
import numpy as np
import itertools

# Load data from a SQL database table into a pandas DataFrame
df = load_db_to_pd(sql_query="select * from sd_table", db_name='Development')
df_events_related_ids = load_db_to_pd(sql_query="select * from events_and_realted_events", db_name='Development')

df_events_related_ids_2 = df_events_related_ids[['eventId', "relatedEventId"]]
related_id_restored_df = pd.merge(df, df_events_related_ids_2, how = 'left', on='eventId')

extracted = related_id_restored_df[['eventId', 'typePrimary', 'playerId', 'teamId', 'matchId', 'sumVaep']]
extracted = extracted.rename(columns ={'eventId': 'relatedEventId', 'typePrimary': 'related_event', 'playerId': 'playerId_2', 'teamId': 'teamId_2', 'matchId': 'matchId_2'})

merged =pd.merge(extracted, related_id_restored_df, how = 'right', on='relatedEventId')
joined_df = merged[['eventId', 'relatedEventId', 'typePrimary', 'related_event', 'playerId', 'playerId_2', 'teamId', 'teamId_2', 'matchId', 'matchId_2', 'sumVaep_x', 'sumVaep_y']]

joined_df = joined_df.rename(columns ={'playerId': 'playerId_1' ,'matchId': 'matchId_1', 'teamId': 'teamId_1' ,'sumVaep_x': 'sumVaep_1', 'sumVaep_y': 'sumVaep_2'})
joined_df = joined_df[(joined_df['playerId_1'].notna()) & (joined_df['playerId_2'].notna()) & (joined_df['teamId_1'].notna()) & (joined_df['teamId_2'].notna()) & (joined_df['matchId_1'].notna()) & (joined_df['matchId_2'].notna())]

joined_df['sumVaep_1'] = joined_df['sumVaep_1'].fillna(0)
joined_df['sumVaep_2'] = joined_df['sumVaep_2'].fillna(0)


merged['relatedEventId'] = merged['relatedEventId'].fillna(0)
merged['relatedEventId'] = merged['relatedEventId'].astype('int')


df_cd = pd.merge(df_events_related_ids, df_events_related_ids, how='inner', left_on = 'eventId', right_on = 'relatedEventId')
df_tester = df_cd[['eventId_x', 'relatedEventId_x', 'typePrimary_x', 'typePrimary_y']]


df['sumVaep'] = df['sumVaep'].fillna(0)
df = df[df['playerId'] != 0]

df_def_actions_player = df
df_vaep_zone_match = df
df_pos_player = df

df_def_actions_player.insert(57, 'nonPosAction', df_def_actions_player.apply(lambda row: non_possession_action(row), axis=1), allow_duplicates=True) # Mark if action is defensive
df_def_actions_player = df_def_actions_player[df_def_actions_player['nonPosAction'] == 1] # Filter dataframe for defensive actions
df_vaep_zone_match['zone'] = df_vaep_zone_match.apply(lambda row: find_zone_chemistry(row), axis = 1)
df_def_actions_player['zone'] = df_def_actions_player.apply(lambda row: find_zone_chemistry(row), axis = 1) # Filter dataframe for defensive actions

matches_positions = {}
def allocate_position(row):
    x = row['x']
    y = row['y']
    mId = row['matchId']
    pId = row['playerId']
    tId = row['teamId']
    if (mId, pId) in matches_positions:
        matches_positions[(mId, pId)]['x'].append(x)
        matches_positions[(mId, pId)]['y'].append(y)
    else: matches_positions[(mId, pId)] = {'matchId': mId , 'teamId':tId,  'playerId': pId , 'x': [x], 'y': [y]}

df_pos_player.apply(lambda row: allocate_position(row), axis = 1)

def get_average_positions(key_values, newDict):
    for key in key_values:
        newDict[key] = {'matchId': key_values[key]['matchId'],
                        'playerId': key_values[key]['playerId'],
                        'teamId': key_values[key]['teamId'],
                        'avg_x': sum(key_values[key]['x'])/len(key_values[key]['x']),
                        'avg_y': sum(key_values[key]['y'])/len(key_values[key]['y'])}
    return newDict



# Define a class called 'EventStats' with five attributes
class EventStats:
    def __init__(self, combinedVaep, eId1, eId2, t1, t2):
        self.combinedVaep = combinedVaep
        self.eId1 = eId1
        self.eId2 = eId2
        self.t1 = t1
        self.t2 = t2

len(df)
h = []

def find_pairwise_player_vaep(df):
    # Create a default dictionary to store EventStats objects for each match and key combination
    matches = defaultdict(dict)

    df.sort_values("eventId")

    # Iterate over the DataFrame, creating EventStats objects for each row and the following row
    for i, row in df[:-1].iterrows():
        mId = row['matchId']  # get the match ID for the current row
        t1 = row['teamId']  # get the team ID for the current row
        t2 = df.loc[i + 1, 'teamId']  # get the team ID for the following row
        p1 = row['playerId']  # get the player ID for the current row
        p2 = df.loc[i + 1, 'playerId']  # get the player ID for the following row
        vaepSum = row['sumVaep'] + df.loc[i + 1, 'sumVaep']  # calculate the sum of the VAEP values for the two events
        # If the two events are on the same team and involve different players, add them to the matches dictionary
        key = (mId, min(p1, p2), max(p1, p2))
        if t1 == t2 and p1 != p2 and mId == df.loc[i+1, 'matchId']:
                if (key not in matches):
                    matches[key] = {'matchId': mId, 'teamId': t1, 'player1': min(p1, p2), 'player2': max(p1, p2),
                                    'vaepList': [vaepSum]}
                else:
                    list = (matches[key]['vaepList'])
                    list.append(vaepSum)
                    matches[key] = {'matchId': mId, 'teamId': t1, 'player1': min(p1, p2), 'player2': max(p1, p2), 'vaepList': list}

    return matches

def computeJoi(key_values, newDict):
    for key in key_values:
        newDict[key] = {'matchId': key_values[key]['matchId'],
                        'teamId': key_values[key]['teamId'],
                        'player1': key_values[key]['player1'],
                        'player2': key_values[key]['player2'],
                        'vaep': sum(key_values[key]['vaepList'])}
    return newDict

#Compute joi for each pair of players in each game
joi_match_df = pd.DataFrame.from_dict(computeJoi(find_pairwise_player_vaep(df), {}), orient='index', columns=['matchId', 'teamId', 'player1', 'player2',  'vaep']).reset_index(drop=True)



def compute_net_oi_game(df):
    cols = [f'zone_{i}_net_oi' for i in range(1, 10)]
    expected_cols = [f'zone_{i}_expected_vaep' for i in range(1, 10)]
    for col, expected_col in zip(cols, expected_cols):
        df[col] = df[col.replace('_net_oi', '')] - df[expected_col]
    return df


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

def_zones_vaep = find_zones_and_vaep(df_vaep_zone_match)
df_def_actions_vaep_player = find_zones_and_vaep(df_def_actions_player)

def_zones_vaep = def_zones_vaep.groupby(['matchId', 'teamId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                        'zone_6': 'sum',
                                                                        'zone_7': 'sum',
                                                                        'zone_8': 'sum',
                                                                        'zone_9': 'sum'})
df_running_vaep_avg = team_vaep_game(def_zones_vaep)

def_zones_vaep_player = df_def_actions_vaep_player.groupby(['playerId', 'matchId', 'teamId'], as_index=False).agg({
                                                                        'zone_1':'sum',
                                                                        'zone_2':'sum',
                                                                        'zone_3': 'sum',
                                                                        'zone_4': 'sum',
                                                                        'zone_5': 'sum',
                                                                        'zone_6': 'sum',
                                                                        'zone_7': 'sum',
                                                                        'zone_8': 'sum',
                                                                        'zone_9': 'sum'})



df_net_oi = compute_net_oi_game(df_running_vaep_avg)


avg_position_match_df = pd.DataFrame.from_dict(get_average_positions(matches_positions, {}), orient='index', columns=['matchId', 'teamId', 'playerId', 'avg_x', 'avg_y']).reset_index(drop=True)
df_matches_and_teams = (df[['matchId', 'teamId']].drop_duplicates()).reset_index(drop=True)

for i, row in df_matches_and_teams.iterrows():
    df_for_analyses = avg_position_match_df[(avg_position_match_df['matchId'] == row['matchId']) & (avg_position_match_df['teamId'] == row['teamId'])]
    print(df_for_analyses)

def compute_distances(df_mt, df_full):
    players_matches_distances = {}
    for i, row in df_mt.iterrows():
        df_for_analyses = df_full[(df_full['matchId'] == row['matchId']) & (df_full['teamId'] == row['teamId'])]
        mId = row['matchId']
        tId = row['teamId']
        for i in range(len(df_for_analyses)):
            pId = df_for_analyses.iloc[i, df_for_analyses.columns.get_loc('playerId')]
            x1 = df_for_analyses.iloc[i, df_for_analyses.columns.get_loc('avg_x')]
            y1 = df_for_analyses.iloc[i, df_for_analyses.columns.get_loc('avg_y')]
            for j in range(i, len(df_for_analyses)):
                x2 = df_for_analyses.iloc[j, df_for_analyses.columns.get_loc('avg_x')]
                y2 = df_for_analyses.iloc[j, df_for_analyses.columns.get_loc('avg_y')]
                pId2 = df_for_analyses.iloc[j, df_for_analyses.columns.get_loc('playerId')]
                dist = math.dist((x1, y1), (x2, y2))
                if ((mId, pId, pId2) not in players_matches_distances) and ((mId, pId2, pId) not in players_matches_distances) and pId != pId2:
                    if pId <= pId2:
                        players_matches_distances[(mId, pId, pId2)] = {'matchId': mId, 'teamId': tId, 'player1': pId,
                                                                       'player2': pId2, 'distance': dist}
                    else: players_matches_distances[(mId, pId2, pId)] = {'matchId': mId, 'teamId': tId, 'player1': pId2,
                                                                       'player2': pId, 'distance': dist}

    return players_matches_distances

distances = compute_distances(df_matches_and_teams, avg_position_match_df)
distances_df = pd.DataFrame.from_dict(distances, orient='index', columns=['matchId', 'teamId', 'player1', 'player2', 'distance']).reset_index(drop=True)


# For testing
players2 = df[(df['playerId'] == 21354) & (df['matchId'] == 5252305) | (df['playerId'] == 214239) & (df['matchId'] == 5252305)]
avg_x_p3 = (players2[players2['playerId'] == 21354])['x'].mean()
avg_y_p3 = (players2[players2['playerId'] == 21354])['y'].mean()
avg_x_p4 = (players2[players2['playerId'] == 214239])['x'].mean()
avg_y_p4 = (players2[players2['playerId'] == 214239])['y'].mean()
math.dist((avg_x_p3, avg_y_p3), (avg_x_p4, avg_y_p4))

players3 = df[(df['playerId'] == 26192) & (df['matchId'] == 5252305) | (df['playerId'] == 32600) & (df['matchId'] == 5252305)]
avg_x_p5 = (players3[players3['playerId'] == 21354])['x'].mean()
avg_y_p5 = (players3[players3['playerId'] == 21354])['y'].mean()
avg_x_p6 = (players3[players3['playerId'] == 214239])['x'].mean()
avg_y_p46 = (players3[players3['playerId'] == 214239])['y'].mean()
math.dist((avg_x_p3, avg_y_p3), (avg_x_p4, avg_y_p4))



def compute_distances3(df_mt, df_full):
    players_matches_distances = {}
    # create a dictionary that maps unique (matchId, teamId) pairs to the corresponding subset of df_full
    df_dict = {tuple(row[['matchId', 'teamId']]): df_full[(df_full['matchId'] == row['matchId']) & (df_full['teamId'] == row['teamId'])] for _, row in df_mt.iterrows()}
    # iterate over the dictionary items and compute distances
    for key, df_for_analyses in df_dict.items():
        matchId, teamId = key
        # iterate over pairs of rows and compute distances
        for (i1, row1), (i2, row2) in itertools.combinations(df_for_analyses.iterrows(), 2):
            pId1, pId2 = row1['playerId'], row2['playerId']
            # compute the Euclidean distance between the two rows
            dist = np.linalg.norm(row1[['x', 'y']] - row2[['x', 'y']])
            if ((matchId, pId1, pId2) not in players_matches_distances) and ((matchId, pId2, pId1) not in players_matches_distances):
                if pId1 <= pId2:
                    players_matches_distances[(matchId, pId1, pId2)] = {'matchId': matchId, 'teamId': teamId, 'player1': pId1, 'player2': pId2, 'distance': dist}
                else:
                    players_matches_distances[(matchId, pId2, pId1)] = {'matchId': matchId, 'teamId': teamId, 'player1': pId2, 'player2': pId1, 'distance': dist}
    return players_matches_distances

distances = compute_distances3(df_matches_and_teams, df)
distances_df = pd.DataFrame.from_dict(distances, orient='index', columns=['matchId', 'teamId', 'player1', 'player2', 'distance']).reset_index(drop=True)


# Euclidian distances between players per game
distances_df

#Joi