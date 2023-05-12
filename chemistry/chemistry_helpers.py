import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

from helpers.student_bif_code import load_db_to_pd
from datetime import date, datetime


def find_zones_and_vaep(df):
    # create dummy variables for the zone column
    df['zone_1'] = np.where(df.zone == 1, df.sumVaep, 0)
    df['zone_2'] = np.where(df.zone == 2, df.sumVaep, 0)
    df['zone_3'] = np.where(df.zone == 3, df.sumVaep, 0)
    df['zone_4'] = np.where(df.zone == 4, df.sumVaep, 0)
    df['zone_5'] = np.where(df.zone == 5, df.sumVaep, 0)
    df['zone_6'] = np.where(df.zone == 6, df.sumVaep, 0)

    return df


def find_zone_chemistry(row):
    s = ""  # Initialize variable s to an empty string
    x = row['x']  # Extract the value of x from the input row
    y = row['y']  # Extract the value of y from the input row

    # Check which zone the (x, y) pair belongs to
    if x >= 0 and x < 50 and y >= 0 and y < 33:
        s = 1  # Assign s to 1 if the pair belongs to zone 1
    elif x >= 50 and x <= 100 and y >= 0 and y < 33:
        s = 4  # Assign s to 4 if the pair belongs to zone 4
    elif x >= 0 and x < 50 and y >= 33 and y < 66:
        s = 2  # Assign s to 2 if the pair belongs to zone 2
    elif x >= 50 and x <= 100 and y >= 33 and y < 66:
        s = 5  # Assign s to 5 if the pair belongs to zone 5
    elif x >= 0 and x < 50 and y >= 66 and y <= 100:
        s = 3  # Assign s to 3 if the pair belongs to zone 3
    elif x >= 50 and x <= 100 and y >= 66 and y <= 100:
        s = 6  # Assign s to 6 if the pair belongs to zone 6
    else:
        s = 0  # Assign s to 0 if the (x, y) pair doesn't belong to any zone

    return s  # Return the zone number that the (x, y) pair belongs to
def find_zone_chemistry_pred(row):
    s = ""  # Initialize variable s to an empty string
    x = (row['spatial_pos_y'])[0]  # Extract the value of x from the input row
    y = (row['spatial_pos_y'])[1]  # Extract the value of y from the input row

    # Check which zone the (x, y) pair belongs to
    if x >= 0 and x < 50 and y >= 0 and y < 33:
        s = 1  # Assign s to 1 if the pair belongs to zone 1
    elif x >= 50 and x <= 100 and y >= 0 and y < 33:
        s = 4  # Assign s to 4 if the pair belongs to zone 4
    elif x >= 0 and x < 50 and y >= 33 and y < 66:
        s = 2  # Assign s to 2 if the pair belongs to zone 2
    elif x >= 50 and x <= 100 and y >= 33 and y < 66:
        s = 5  # Assign s to 5 if the pair belongs to zone 5
    elif x >= 0 and x < 50 and y >= 66 and y <= 100:
        s = 3  # Assign s to 3 if the pair belongs to zone 3
    elif x >= 50 and x <= 100 and y >= 66 and y <= 100:
        s = 6  # Assign s to 6 if the pair belongs to zone 6
    else:
        s = 0  # Assign s to 0 if the (x, y) pair doesn't belong to any zone

    return s  # Return the zone number that the (x, y) pair belongs to


def find_zone_chemistry_v2(row):
    s = ""
    #  id = row['id']
    # print(row)
    x = row['x']
    y = row['y']
    if (x >= 0 and x <= 33 and y >= 0 and y <= 33):
        s = 1
    elif (x >= 0 and x <= 33 and y > 33 and y <= 67):
        s = 2
    elif (x >= 0 and x <= 33 and y > 67 and y <= 100):
        s = 3
    elif (x > 33 and x <= 67 and y >= 0 and y <= 33):
        s = 4
    elif (x > 33 and x <= 67 and y > 33 and y <= 67):
        s = 5
    elif (x > 33 and x <= 67 and y > 67 and y <= 100):
        s = 6
    elif (x > 67 and x <= 100 and y >= 0 and y <= 33):
        s = 7
    elif (x > 67 and x <= 100 and y > 33 and y <= 67):
        s = 8
    elif (x >= 67 and x <= 100 and y >= 67 and y <= 100):
        s = 9
    else:
        s = 0
    return s



def get_average_positions(key_values, newDict):
    # Loop through each key in key_values
    for key in key_values:
        # Create a new dictionary entry for the key with the following keys and values
        newDict[key] = {'matchId': key_values[key]['matchId'],
                        'playerId': key_values[key]['playerId'],
                        'teamId': key_values[key]['teamId'],
                        'avg_x': sum(key_values[key]['x'])/len(key_values[key]['x']),
                        'avg_y': sum(key_values[key]['y'])/len(key_values[key]['y'])}
    # Return the new dictionary
    return newDict


def allocate_position(row, matches_positions):
    # extract relevant values from row
    x = row['x']
    y = row['y']
    mId = row['matchId']
    pId = row['playerId']
    tId = row['teamId']

    # check if the (matchId, playerId) pair is already in the dictionary
    if (mId, pId) in matches_positions:
        # if it is, add the new x and y values to the existing list
        matches_positions[(mId, pId)]['x'].append(x)
        matches_positions[(mId, pId)]['y'].append(y)
    else:
        # if it isn't, create a new entry in the dictionary with the matchId, teamId, playerId, and the x and y values
        matches_positions[(mId, pId)] = {'matchId': mId, 'teamId': tId, 'playerId': pId, 'x': [x], 'y': [y]}



def compute_relative_player_impact(df_player, df_team):
    # Merge two dataframes based on matchId and teamId using inner join
    df_full = pd.merge(df_player, df_team, on=['matchId', 'teamId'], how='inner')

    # Compute relative player impact for each of the six zones
    # If the number of passes in a zone is greater than 0, compute the impact as the ratio of passes completed to passes attempted in that zone
    # If the number of passes in a zone is 0, set the impact for that zone to 0
    df_full['zone_1_imp'] = np.where(df_full.zone_1_pl > 0, df_full.zone_1_pl / df_full.zone_1_t, 0)
    df_full['zone_2_imp'] = np.where(df_full.zone_2_pl > 0, df_full.zone_2_pl / df_full.zone_2_t, 0)
    df_full['zone_3_imp'] = np.where(df_full.zone_3_pl > 0, df_full.zone_3_pl / df_full.zone_3_t, 0)
    df_full['zone_4_imp'] = np.where(df_full.zone_4_pl > 0, df_full.zone_4_pl / df_full.zone_4_t, 0)
    df_full['zone_5_imp'] = np.where(df_full.zone_5_pl > 0, df_full.zone_5_pl / df_full.zone_5_t, 0)
    df_full['zone_6_imp'] = np.where(df_full.zone_6_pl > 0, df_full.zone_6_pl / df_full.zone_6_t, 0)

    # Return the dataframe with the relative player impact for each zone
    return df_full


def compute_relative_player_impact_v4(df_player, df_team):
    # Merge two dataframes based on matchId and teamId using inner join
    df_full = pd.merge(df_player, df_team, on=['matchId', 'teamId'], how='inner')

    # Compute relative player impact for each of the six zones
    # If the number of passes in a zone is greater than 0, compute the impact as the ratio of passes completed to passes attempted in that zone
    # If the number of passes in a zone is 0, set the impact for that zone to 0
    df_full['zone_1_imp'] = np.where(df_full.zone_1_pl > 0, df_full.zone_1_pl / df_full.zone_1_t, 0)
    df_full['zone_2_imp'] = np.where(df_full.zone_2_pl > 0, df_full.zone_2_pl / df_full.zone_2_t, 0)
    df_full['zone_3_imp'] = np.where(df_full.zone_3_pl > 0, df_full.zone_3_pl / df_full.zone_3_t, 0)
    df_full['zone_4_imp'] = np.where(df_full.zone_4_pl > 0, df_full.zone_4_pl / df_full.zone_4_t, 0)
    df_full['zone_5_imp'] = np.where(df_full.zone_5_pl > 0, df_full.zone_5_pl / df_full.zone_5_t, 0)
    df_full['zone_6_imp'] = np.where(df_full.zone_6_pl > 0, df_full.zone_6_pl / df_full.zone_6_t, 0)
    df_full['zone_7_imp'] = np.where(df_full.zone_7_pl > 0, df_full.zone_7_pl / df_full.zone_7_t, 0)
    df_full['zone_8_imp'] = np.where(df_full.zone_8_pl > 0, df_full.zone_8_pl / df_full.zone_8_t, 0)
    df_full['zone_9_imp'] = np.where(df_full.zone_9_pl > 0, df_full.zone_9_pl / df_full.zone_9_t, 0)

    # Return the dataframe with the relative player impact for each zone
    return df_full


def compute_relative_player_impact_v3(df_player, df_team):
    # Merge two dataframes based on matchId and teamId using inner join
    df_full = pd.merge(df_player, df_team, on=['matchId', 'teamId'], how='inner')

    # Compute relative player impact for each of the six zones
    # If the number of passes in a zone is greater than 0, compute the impact as the ratio of passes completed to passes attempted in that zone
    # If the number of passes in a zone is 0, set the impact for that zone to 0
    df_full['zone_1_imp'] = np.where(df_full.zone_1_pl > 0, df_full.zone_1_pl / df_full.zone_1_t, 0)
    df_full['zone_2_imp'] = np.where(df_full.zone_2_pl > 0, df_full.zone_2_pl / df_full.zone_2_t, 0)
    df_full['zone_3_imp'] = np.where(df_full.zone_3_pl > 0, df_full.zone_3_pl / df_full.zone_3_t, 0)
    df_full['zone_4_imp'] = np.where(df_full.zone_4_pl > 0, df_full.zone_4_pl / df_full.zone_4_t, 0)
    df_full['zone_5_imp'] = np.where(df_full.zone_5_pl > 0, df_full.zone_5_pl / df_full.zone_5_t, 0)
    df_full['zone_6_imp'] = np.where(df_full.zone_6_pl > 0, df_full.zone_6_pl / df_full.zone_6_t, 0)

    df_full['zone_total'] = df_full.zone_1_t + df_full.zone_2_t + df_full.zone_3_t + df_full.zone_4_t + df_full.zone_5_t + df_full.zone_6_t

    df_full['zone_1_weight'] = df_full.zone_1_t / df_full.zone_total
    df_full['zone_2_weight'] = df_full.zone_2_t / df_full.zone_total
    df_full['zone_3_weight'] = df_full.zone_3_t / df_full.zone_total
    df_full['zone_4_weight'] = df_full.zone_4_t / df_full.zone_total
    df_full['zone_5_weight'] = df_full.zone_5_t / df_full.zone_total
    df_full['zone_6_weight'] = df_full.zone_6_t / df_full.zone_total
    # Return the dataframe with the relative player impact for each zone
    return df_full


def compute_relative_player_impact_v2(df_player, df_team):
    # Merge two dataframes based on matchId and teamId using inner join
    df_full = pd.merge(df_player, df_team, on=['matchId', 'teamId'], how='inner')

    # Compute relative player impact for each of the six zones
    # If the number of passes in a zone is greater than 0, compute the impact as the ratio of passes completed to passes attempted in that zone
    # If the number of passes in a zone is 0, set the impact for that zone to 0
    df_full['impact'] = df_full.def_actions_count_pl / df_full.def_actions_count_t

    # Return the dataframe with the relative player impact for each zone
    return df_full




def find_zones_and_counts_pl(df):
    # Use the pandas "get_dummies" method to create dummy variables for each "zone" category,
    # prefixing them with "zone_" in the column names.
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    # Concatenate the "zone_dummies" DataFrame with the input DataFrame "df" column-wise (axis=1).
    df = pd.concat([df, zone_dummies], axis=1)

    # Group the DataFrame by playerId, matchId, and teamId columns, and calculate the sum of each zone column.
    # Reset the index to create a flat DataFrame with the original columns and aggregated zone columns.
    df = df.groupby([df.playerId, df.matchId, df.teamId], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum'
    })

    # Rename the aggregated zone columns, appending "_pl" to each column name to denote that the counts
    # represent the number of times each player appeared in each zone.
    df = df.rename(columns={
        'zone_1': 'zone_1_pl',
        'zone_2': 'zone_2_pl',
        'zone_3': 'zone_3_pl',
        'zone_4': 'zone_4_pl',
        'zone_5': 'zone_5_pl',
        'zone_6': 'zone_6_pl'
    })

    # Return the modified DataFrame with aggregated zone counts.
    return df

def find_zones_and_counts_pl_v2(df):
    # Use the pandas "get_dummies" method to create dummy variables for each "zone" category,
    # prefixing them with "zone_" in the column names.
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    # Concatenate the "zone_dummies" DataFrame with the input DataFrame "df" column-wise (axis=1).
    df = pd.concat([df, zone_dummies], axis=1)

    # Group the DataFrame by playerId, matchId, and teamId columns, and calculate the sum of each zone column.
    # Reset the index to create a flat DataFrame with the original columns and aggregated zone columns.
    df = df.groupby([df.playerId, df.matchId, df.teamId], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum',
        'zone_7': 'sum',
        'zone_8': 'sum',
        'zone_9': 'sum'
    })

    # Rename the aggregated zone columns, appending "_pl" to each column name to denote that the counts
    # represent the number of times each player appeared in each zone.
    df = df.rename(columns={
        'zone_1': 'zone_1_pl',
        'zone_2': 'zone_2_pl',
        'zone_3': 'zone_3_pl',
        'zone_4': 'zone_4_pl',
        'zone_5': 'zone_5_pl',
        'zone_6': 'zone_6_pl',
        'zone_7': 'zone_7_pl',
        'zone_8': 'zone_8_pl',
        'zone_9': 'zone_9_pl'
    })

    # Return the modified DataFrame with aggregated zone counts.
    return df


def find_zones_and_counts_t(df):
    #extact dummies from zone columns
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    #Append zone dummies to original dataframe - columnswise
    df = pd.concat([df, zone_dummies], axis=1)

    #Groupby statement to compute counts for the amount of times a teams has
    #engaged in actoins within a zone
    df = df.groupby([df.matchId, df.teamId], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum'})
    df = df.rename(columns={'zone_1': 'zone_1_t',
                            'zone_2': 'zone_2_t',
                            'zone_3': 'zone_3_t',
                            'zone_4': 'zone_4_t',
                            'zone_5': 'zone_5_t',
                            'zone_6': 'zone_6_t'
                            })

    return df

def find_zones_and_counts_t_v2(df):
    #extact dummies from zone columns
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    #Append zone dummies to original dataframe - columnswise
    df = pd.concat([df, zone_dummies], axis=1)

    #Groupby statement to compute counts for the amount of times a teams has
    #engaged in actoins within a zone
    df = df.groupby([df.matchId, df.teamId], as_index=False).agg({
        'zone_1': 'sum',
        'zone_2': 'sum',
        'zone_3': 'sum',
        'zone_4': 'sum',
        'zone_5': 'sum',
        'zone_6': 'sum',
        'zone_7': 'sum',
        'zone_8': 'sum',
        'zone_9': 'sum'
    })
    df = df.rename(columns={'zone_1': 'zone_1_t',
                            'zone_2': 'zone_2_t',
                            'zone_3': 'zone_3_t',
                            'zone_4': 'zone_4_t',
                            'zone_5': 'zone_5_t',
                            'zone_6': 'zone_6_t',
                            'zone_7': 'zone_7_t',
                            'zone_8': 'zone_8_t',
                            'zone_9': 'zone_9_t'
                            })

    return df

def compute_jdi (df):
    #Compute jdi per zone per pair of players
    df['jdi_zone_1'] = np.where(df.zone_1_t > 3, df.zone_1_net_oi * df.impact  * (1/df.distance), 0)
    df['jdi_zone_2'] = np.where(df.zone_2_t > 3, df.zone_2_net_oi * df.impact * (1/df.distance), 0)
    df['jdi_zone_3'] = np.where(df.zone_3_t > 3, df.zone_3_net_oi * df.impact * (1/df.distance), 0)
    df['jdi_zone_4'] = np.where(df.zone_4_t > 3, df.zone_4_net_oi * df.impact * (1/df.distance), 0)
    df['jdi_zone_5'] = np.where(df.zone_5_t > 3, df.zone_5_net_oi * df.impact * (1/df.distance), 0)
    df['jdi_zone_6'] = np.where(df.zone_6_t > 3, df.zone_6_net_oi * df.impact * (1/df.distance), 0)

    #Compute total jdi across zones for pairs of players
    df['jdi'] = df.jdi_zone_1 + df.jdi_zone_2 + df.jdi_zone_3 + df.jdi_zone_4 + df.jdi_zone_5 + df.jdi_zone_6
    return df

def compute_jdi (df):
    df['total_actions'] = df['zones_total'] = df.zone_1_team + df.zone_2_team + df.zone_3_team + df.zone_4_team + df.zone_5_team + df.zone_6_team
    df['jdi_zone_1'] = np.where(df.zone_1_team > 3, (((df.zone_1_net_oi * df.zone_1_imp1) / df.distance ) + ((df.zone_1_team* df.zone_1_imp2) / df.distance)), 0)
    df['jdi_zone_2'] = np.where(df.zone_2_team > 3, ((df.zone_2_net_oi * df.zone_2_imp1) / df.distance ) + ((df.zone_2_team* df.zone_2_imp2) / df.distance) , 0)
    df['jdi_zone_3'] = np.where(df.zone_3_team > 3, ((df.zone_3_net_oi * df.zone_3_imp1) / df.distance ) + ((df.zone_3_team* df.zone_3_imp2) / df.distance), 0)
    df['jdi_zone_4'] = np.where(df.zone_4_team > 3, ((df.zone_4_net_oi * df.zone_4_imp1) / df.distance ) + ((df.zone_4_team* df.zone_4_imp2) / df.distance), 0)
    df['jdi_zone_5'] = np.where(df.zone_5_team > 3, ((df.zone_5_net_oi * df.zone_5_imp1) / df.distance ) + ((df.zone_5_team* df.zone_5_imp2) / df.distance), 0)
    df['jdi_zone_6'] = np.where(df.zone_6_team > 3, ((df.zone_6_net_oi * df.zone_6_imp1) / df.distance ) + ((df.zone_6_team* df.zone_6_imp2) / df.distance), 0)

    #Compute total jdi across zones for pairs of players
    df['jdi'] = df.jdi_zone_1 + df.jdi_zone_2 + df.jdi_zone_3 + df.jdi_zone_4 + df.jdi_zone_5 + df.jdi_zone_6
    return df

#Zone game importance included
'''
 df['jdi_zone_1'] = np.where(df.zone_1_team > 3, (((df.zone_1_net_oi * df.zone_1_imp1) / df.distance ) + ((df.zone_1_team* df.zone_1_imp2) / df.distance))*(df.zone_1_team/df.zones_total), 0)
    df['jdi_zone_2'] = np.where(df.zone_2_team > 3, ((df.zone_2_net_oi * df.zone_2_imp1) / df.distance ) + ((df.zone_2_team* df.zone_2_imp2) / df.distance)*(df.zone_2_team/df.zones_total) , 0)
    df['jdi_zone_3'] = np.where(df.zone_3_team > 3, ((df.zone_3_net_oi * df.zone_3_imp1) / df.distance ) + ((df.zone_3_team* df.zone_3_imp2) / df.distance)*(df.zone_3_team/df.zones_total), 0)
    df['jdi_zone_4'] = np.where(df.zone_4_team > 3, ((df.zone_4_net_oi * df.zone_4_imp1) / df.distance ) + ((df.zone_4_team* df.zone_4_imp2) / df.distance)*(df.zone_4_team/df.zones_total), 0)
    df['jdi_zone_5'] = np.where(df.zone_5_team > 3, ((df.zone_5_net_oi * df.zone_5_imp1) / df.distance ) + ((df.zone_5_team* df.zone_5_imp2) / df.distance)*(df.zone_5_team/df.zones_total), 0)
    df['jdi_zone_6'] = np.where(df.zone_6_team > 3, ((df.zone_6_net_oi * df.zone_6_imp1) / df.distance ) + ((df.zone_6_team* df.zone_6_imp2) / df.distance)*(df.zone_6_team/df.zones_total), 0)
'''

def create_def_succes_frame(ground_duels, air_duels):
    ground_duels_suc_rec = ground_duels[['eventId', 'matchId', 'groundDuel_recoveredPossession']]
    ground_duels_suc_stop = ground_duels[['eventId', 'matchId', 'groundDuel_stoppedProgress']]
    ground_duels_suc_rec_1 = (ground_duels_suc_rec[ground_duels_suc_rec['groundDuel_recoveredPossession'] == True]).rename(columns = {'groundDuel_recoveredPossession': 'def_action'})
    ground_duels_suc_stop_1 = (ground_duels_suc_stop[ground_duels_suc_stop['groundDuel_stoppedProgress'] == True]).rename(columns = {'groundDuel_stoppedProgress': 'def_action'})
    air_duels_suc = (air_duels[(air_duels['aerialDuel_firstTouch'] == True)]).rename(columns={'aerialDuel_firstTouch': 'def_action'})
    df_succes_tot = pd.concat([ground_duels_suc_rec_1, ground_duels_suc_stop_1, air_duels_suc[['eventId', 'matchId', 'def_action']] ])
    df_succes_tot['def_action'] = df_succes_tot['def_action'].replace(True, 1)
    return df_succes_tot



def jdi_compute(df, def_suc):
    df['zones_total'] = np.sum(df[['zone_1_team', 'zone_2_team', 'zone_3_team', 'zone_4_team', 'zone_5_team', 'zone_6_team']], axis=1)
    df['p1_sum'] = np.sum(df[['zone_1_pl1', 'zone_2_pl1', 'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1', 'zone_6_pl1']], axis=1)
    df['p2_sum'] = np.sum(df[['zone_1_pl2', 'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2', 'zone_5_pl2', 'zone_6_pl2']], axis=1)
    df['pairwise_involvement'] = ((df.p1_sum +df.p2_sum)/df.zones_total)
    df['jdi_zone_1'] = np.where(df.zone_1_team > 15, (((((((df.zone_6_net_oi * df.zone_1_imp1)) + ((df.zone_6_net_oi* df.zone_1_imp2))))))), 0)
    df['jdi_zone_2'] = np.where(df.zone_2_team > 15, ((((((df.zone_5_net_oi * df.zone_2_imp1)) + ((df.zone_5_net_oi* df.zone_2_imp2)))))), 0)
    df['jdi_zone_3'] = np.where(df.zone_3_team > 15, ((((((df.zone_4_net_oi * df.zone_3_imp1)) + ((df.zone_4_net_oi* df.zone_3_imp2)))))), 0)
    df['jdi_zone_4'] = np.where(df.zone_4_team > 15, ((((((df.zone_3_net_oi * df.zone_4_imp1)) + ((df.zone_3_net_oi* df.zone_4_imp2)))))), 0)
    df['jdi_zone_5'] = np.where(df.zone_5_team > 15, ((((((df.zone_2_net_oi * df.zone_5_imp1)) + ((df.zone_2_net_oi* df.zone_5_imp2)))))), 0)
    df['jdi_zone_6'] = np.where(df.zone_6_team > 15, ((((((df.zone_1_net_oi * df.zone_6_imp1)) + ((df.zone_1_net_oi* df.zone_6_imp2)))))), 0)
    df = df.merge(def_suc, left_on=['playerId1', 'matchId'], right_on= ['playerId', 'matchId'], how='left')
    df = df.merge(def_suc, left_on=['playerId2', 'matchId'], right_on= ['playerId', 'matchId'], how='left')
    df['def_action_x'] = df['def_action_x'].fillna(0)
    df['def_action_y'] = df['def_action_y'].fillna(0)
    df['jdi'] = (((df.jdi_zone_1 + df.jdi_zone_2 + df.jdi_zone_3 + df.jdi_zone_4 + df.jdi_zone_5 + df.jdi_zone_6) * (df.def_action_x + df.def_action_y)) /df.distance)
    return df

def jdi_computed_v2(df):
    df['zones_total'] = np.sum(df[['zone_1_team', 'zone_2_team', 'zone_3_team', 'zone_4_team', 'zone_5_team', 'zone_6_team', 'zone_7_team', 'zone_8_team', 'zone_9_team']], axis=1)
    df['p1_sum'] = np.sum(df[['zone_1_pl1', 'zone_2_pl1', 'zone_3_pl1', 'zone_4_pl1', 'zone_5_pl1', 'zone_6_pl1', 'zone_7_pl1', 'zone_8_pl1', 'zone_9_pl1']], axis=1)
    df['p2_sum'] = np.sum(df[['zone_1_pl2', 'zone_2_pl2', 'zone_3_pl2', 'zone_4_pl2', 'zone_5_pl2', 'zone_6_pl2', 'zone_7_pl2', 'zone_8_pl2', 'zone_9_pl2']], axis=1)
    df['pairwise_involvement'] = ((df.p1_sum +df.p2_sum)/df.zones_total)
    df[['zone_1_weight', 'zone_2_weight', 'zone_3_weight', 'zone_4_weight', 'zone_5_weight', 'zone_6_weight', 'zone_7_weight', 'zone_8_weight', 'zone_9_weight']] = df[['zone_1_team', 'zone_2_team', 'zone_3_team', 'zone_4_team', 'zone_5_team', 'zone_6_team', 'zone_7_team', 'zone_8_team', 'zone_9_team']] / df[ 'zones_total'].values[:,None]
    df['jdi_zone_1'] = np.where(df.zone_1_team > 10, (((((((df.zone_1_net_oi * df.zone_1_imp1)) + ((df.zone_1_net_oi* df.zone_1_imp2))))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_2'] = np.where(df.zone_2_team > 10, ((((((df.zone_2_net_oi * df.zone_2_imp1) ) + ((df.zone_2_net_oi* df.zone_2_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_3'] = np.where(df.zone_3_team > 10, ((((((df.zone_3_net_oi * df.zone_3_imp1) ) + ((df.zone_3_net_oi* df.zone_3_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_4'] = np.where(df.zone_4_team > 10, ((((((df.zone_4_net_oi * df.zone_4_imp1)) + ((df.zone_4_net_oi* df.zone_4_imp2)))*(1/df.distance)))* df.pairwise_involvement), 0)
    df['jdi_zone_5'] = np.where(df.zone_5_team > 10, ((((((df.zone_5_net_oi * df.zone_5_imp1)) + ((df.zone_5_net_oi* df.zone_5_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_6'] = np.where(df.zone_6_team > 10, ((((((df.zone_6_net_oi * df.zone_6_imp1)) + ((df.zone_6_net_oi* df.zone_6_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_7'] = np.where(df.zone_7_team > 10, ((((((df.zone_7_net_oi * df.zone_7_imp1)) + ((df.zone_7_net_oi* df.zone_7_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_8'] = np.where(df.zone_8_team > 10, ((((((df.zone_8_net_oi * df.zone_8_imp1)) + ((df.zone_8_net_oi* df.zone_8_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi_zone_9'] = np.where(df.zone_9_team > 10, ((((((df.zone_9_net_oi * df.zone_9_imp1)) + ((df.zone_9_net_oi* df.zone_9_imp2)))*(1/df.distance))) * df.pairwise_involvement), 0)
    df['jdi'] = df.jdi_zone_1 + df.jdi_zone_2 + df.jdi_zone_3 + df.jdi_zone_4 + df.jdi_zone_5 + df.jdi_zone_6+ df.jdi_zone_7 + df.jdi_zone_8 + df.jdi_zone_9
    return df
#Compute total jdi across zones for pairs of players




def generate_chemistry_ability(df):
    df1 = df[['p1','shortName_x', 'role_name_x','areaName_x', 'df_jdi90', 'df_joi90', 'chemistry']].rename(columns={'p1':'playerId', 'shortName_x': 'shortName', 'role_name_x': 'role_name' ,'areaName_x': 'areaName'})
    df2 = df[['p2', 'shortName_y', 'role_name_y','areaName_y', 'df_jdi90', 'df_joi90', 'chemistry']].rename(columns={'p2':'playerId','shortName_y': 'shortName', 'role_name_y': 'role_name' ,'areaName_y': 'areaName'})
    players_and_chemistry = pd.concat([df1, df2])
    players_and_chemistry_season = players_and_chemistry.groupby(['playerId', 'shortName',  'role_name' , 'areaName'], as_index=False)['chemistry'].sum().reset_index(drop=True)

    return players_and_chemistry_season
def generate_chemistry_ability_v2(df):
    df1 = df[['p1','shortName_x', 'role_name_x','areaName_x', 'df_jdi90', 'df_joi90', 'chemistry']].rename(columns={'p1':'playerId', 'shortName_x': 'shortName', 'role_name_x': 'role_name' ,'areaName_x': 'areaName'})
    df1  = df1.drop_duplicates()
    df2 = df[['p2', 'shortName_y', 'role_name_y','areaName_y', 'df_jdi90', 'df_joi90', 'chemistry']].rename(columns={'p2':'playerId','shortName_y': 'shortName', 'role_name_y': 'role_name' ,'areaName_y': 'areaName'})
    df2  = df2.drop_duplicates()
    players_and_chemistry = pd.concat([df1, df2])
    players_and_chemistry_season = players_and_chemistry.groupby(['playerId', 'shortName',  'role_name' , 'areaName'], as_index=False)['chemistry'].mean().reset_index(drop=True)
    players_and_chemistry_season = players_and_chemistry_season.rename(columns = {'chemistry': 'chem_ability'})
    return players_and_chemistry_season

def generate_chemistry_ability_v3(df):
    df1 = (df[[ 'seasonId', 'p1', 'df_jdi90', 'df_joi90', 'chemistry']]).rename(columns ={'p1': 'playerId'})
    df1  = df1.drop_duplicates()
    df2 = (df[['seasonId', 'p2', 'df_jdi90', 'df_joi90', 'chemistry']]).rename(columns ={'p2': 'playerId'})
    df2  = df2.drop_duplicates()
    players_and_chemistry = pd.concat([df1, df2])
    players_and_chemistry_season = (players_and_chemistry.groupby(['playerId', 'seasonId'], as_index=False)['chemistry'].mean().reset_index(drop=True)).rename(columns = {'chemistry': 'chem_ability'})
    return players_and_chemistry_season





def compute_distances(df_full):
    #Merge dataframw with itself to obtain pairing of players
    merged = df_full.merge(df_full, on=['matchId', 'teamId'], suffixes=('1', '2'))

    #Extract observatoins where players are not paired with itself and
    #Ensure that the player with the lowest playerId is always found on the 'playerId1' column
    merged = merged[(merged.playerId1 != merged.playerId2) & (merged.playerId1 < merged.playerId2)]
    x1 = merged['avg_x1'].values
    y1 = merged['avg_y1'].values
    x2 = merged['avg_x2'].values
    y2 = merged['avg_y2'].values

    # Calculate the Euclidean distance between two points and store the result in the 'distance' column
    merged['distance'] = np.linalg.norm(np.column_stack((x1 - x2, y1 - y2)), axis=1)
    return merged


def compute_net_oi_game(df):
    # make a copy of the input dataframe to avoid modifying it directly
    copy = df

    # compute net offensive impact for each zone based on number of games played
    copy['zone_1_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_1_expected_vaep - copy.zone_1_prior_avg),
                                     (copy.zone_1_expected_vaep - copy.zone_1))
    copy['zone_2_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_2_expected_vaep - copy.zone_2_prior_avg),
                                     (copy.zone_2_expected_vaep - copy.zone_2))
    copy['zone_3_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_3_expected_vaep - copy.zone_3_prior_avg),
                                     (copy.zone_3_expected_vaep - copy.zone_3))
    copy['zone_4_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_4_expected_vaep - copy.zone_4_prior_avg),
                                     (copy.zone_4_expected_vaep - copy.zone_4))
    copy['zone_5_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_5_expected_vaep - copy.zone_5_prior_avg),
                                     (copy.zone_5_expected_vaep - copy.zone_5))
    copy['zone_6_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_6_expected_vaep - copy.zone_6_prior_avg),
                                     (copy.zone_6_expected_vaep - copy.zone_6))

    # return the updated dataframe with net offensive impact for each zone
    return copy


def compute_net_oi_game_v2(df):
    # make a copy of the input dataframe to avoid modifying it directly
    copy = df

    # compute net offensive impact for each zone based on number of games played
    copy['zone_1_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_1_expected_vaep - copy.zone_1_prior_avg),
                                     (copy.zone_1_expected_vaep - copy.zone_1))
    copy['zone_2_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_2_expected_vaep - copy.zone_2_prior_avg),
                                     (copy.zone_2_expected_vaep - copy.zone_2))
    copy['zone_3_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_3_expected_vaep - copy.zone_3_prior_avg),
                                     (copy.zone_3_expected_vaep - copy.zone_3))
    copy['zone_4_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_4_expected_vaep - copy.zone_4_prior_avg),
                                     (copy.zone_4_expected_vaep - copy.zone_4))
    copy['zone_5_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_5_expected_vaep - copy.zone_5_prior_avg),
                                     (copy.zone_5_expected_vaep - copy.zone_5))
    copy['zone_6_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_6_expected_vaep - copy.zone_6_prior_avg),
                                     (copy.zone_6_expected_vaep - copy.zone_6))
    copy['zone_7_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_7_expected_vaep - copy.zone_7_prior_avg),
                                     (copy.zone_7_expected_vaep - copy.zone_7))

    copy['zone_8_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_8_expected_vaep - copy.zone_8_prior_avg),
                                     (copy.zone_8_expected_vaep - copy.zone_8))
    copy['zone_9_net_oi'] = np.where(copy.games_played <= 3,
                                     (copy.zone_9_expected_vaep - copy.zone_9_prior_avg),
                                     (copy.zone_9_expected_vaep - copy.zone_9))

    # return the updated dataframe with net offensive impact for each zone
    return copy


#Method for computing running aver vaep using observatoins as input and with a label input
def compute_running_avg_team_vaep (row, label):
    #If it is the first game of the season, we use the current cumulative sum
    if row['games_played'] == 0:
        return row[label]
    else: # Else we divide the current sum with the amout of games played
        return row[label] / (row['games_played'] +1)

def team_vaep_game(df):
    # Sort the data by teamId and matchId
    df = df.sort_values(by=['teamId', 'matchId'])

    # Compute the cumulative sum of each zone for each team
    df[['zone_1_cumsum', 'zone_2_cumsum',
        'zone_3_cumsum', 'zone_4_cumsum',
        'zone_5_cumsum', 'zone_6_cumsum'
        ]] = df.groupby('teamId')[['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6']].cumsum()

    # Compute the number of games played by each team
    df['games_played'] = df.groupby(['teamId']).cumcount()

    # Compute the running average of expected VAEP for each zone
    df['zone_1_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_1_cumsum'), axis=1)
    df['zone_2_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_2_cumsum'), axis=1)
    df['zone_3_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_3_cumsum'), axis=1)
    df['zone_4_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_4_cumsum'), axis=1)
    df['zone_5_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_5_cumsum'), axis=1)
    df['zone_6_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_6_cumsum'), axis=1)

    # Return the updated dataframe
    return df

def team_vaep_game_v2(df):
    # Group data by matchId and teamId, summing the values of the zones
    df = df.groupby(['matchId', 'teamId'])[
        'zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'zone_7','zone_8','zone_9', ].sum().reset_index()

    # Sort the data by teamId and matchId
    df = df.sort_values(by=['teamId', 'matchId'])

    # Compute the cumulative sum of each zone for each team
    df[['zone_1_cumsum', 'zone_2_cumsum',
        'zone_3_cumsum', 'zone_4_cumsum',
        'zone_5_cumsum', 'zone_6_cumsum',
        'zone_7_cumsum', 'zone_8_cumsum', 'zone_9_cumsum'
        ]] = df.groupby('teamId')[['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6' , 'zone_7', 'zone_8', 'zone_9']].cumsum()

    # Compute the number of games played by each team
    df['games_played'] = df.groupby(['teamId']).cumcount()

    # Compute the running average of expected VAEP for each zone
    df['zone_1_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_1_cumsum'), axis=1)
    df['zone_2_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_2_cumsum'), axis=1)
    df['zone_3_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_3_cumsum'), axis=1)
    df['zone_4_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_4_cumsum'), axis=1)
    df['zone_5_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_5_cumsum'), axis=1)
    df['zone_6_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_6_cumsum'), axis=1)
    df['zone_7_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_7_cumsum'), axis=1)
    df['zone_8_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_8_cumsum'), axis=1)
    df['zone_9_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_9_cumsum'), axis=1)

    # Return the updated dataframe
    return df



def process_for_jdi (df_net_oi, matches_all, distances_df, df_player_share):
    #MAke a copy
    df_net_oi_v2 = df_net_oi[['matchId', 'teamId',
                              'zone_1_net_oi', 'zone_2_net_oi',
                              'zone_3_net_oi', 'zone_4_net_oi',
                              'zone_5_net_oi', 'zone_6_net_oi'
                              ]]
    #Merge with mataches to discover home and away teams
    df_net_oi_v2 = pd.merge(df_net_oi_v2, matches_all, on='matchId')

    #Determine the opposing team in order for us to check whether opposing team over or underperforms in an area
    df_net_oi_v2['opposing_team'] = np.where(df_net_oi_v2.teamId == df_net_oi_v2.home_teamId, df_net_oi_v2.away_teamId,df_net_oi_v2.home_teamId)

    #Remove irrelevant columns
    df_net_oi_v2 = df_net_oi_v2.drop(['home_teamId', 'away_teamId'], axis=1)

    #Copy of df
    duplicate_for_merge = df_net_oi_v2.copy()

    #merge the two dataframes to get who is the opposing team for each game and what are their net oi
    merged_opposing_vaep_values = pd.merge(df_net_oi_v2, duplicate_for_merge, left_on=(['matchId', 'teamId']),right_on=(['matchId', 'opposing_team']))


    #Remove unusable columns
    merged_opposing_vaep_values = merged_opposing_vaep_values.drop(['teamId_y', 'zone_1_net_oi_x', 'zone_2_net_oi_x',
                                                                    'zone_3_net_oi_x', 'zone_4_net_oi_x',
                                                                    'zone_5_net_oi_x', 'zone_6_net_oi_x',
                                                                    'opposing_team_y'
                                                                    ], axis=1)

    #Assign appropriate feature names
    merged_opposing_vaep_values = merged_opposing_vaep_values.rename(
        columns={'teamId_x': 'teamId', 'opposing_team_x': 'opposing_team',
                 'zone_1_net_oi_y': 'zone_1_net_oi', 'zone_2_net_oi_y': 'zone_2_net_oi',
                 'zone_3_net_oi_y': 'zone_3_net_oi', 'zone_4_net_oi_y': 'zone_4_net_oi',
                 'zone_5_net_oi_y': 'zone_5_net_oi', 'zone_6_net_oi_y': 'zone_6_net_oi'})

    #Merge to obtain net oi in each dataframe and share for each player
    df_snoi_netoi = pd.merge(merged_opposing_vaep_values, df_player_share, on=(['matchId', 'teamId']), how='inner')

    #merge with distances
    df_share_dist = pd.merge(df_snoi_netoi, distances_df, left_on=(['matchId', 'teamId', 'playerId']), right_on=(['matchId', 'teamId', 'playerId1']), how='inner')
    df_share_dist_2 = pd.merge(df_snoi_netoi, distances_df, left_on=(['matchId', 'teamId', 'playerId']), right_on=(['matchId', 'teamId', 'playerId2']), how='inner')
    df_share_dist_final = pd.concat([df_share_dist, df_share_dist_2])
    return df_share_dist_final


def pairwise_playing_time(df):
    # Merge the DataFrame with itself based on the matchId and teamId columns
    paired = df.merge(df, on=(['matchId', 'teamId']), how='inner', suffixes=('1', '2'))
    # Remove any rows where playerId1 is equal to playerId2
    paired = paired[paired.playerId1 != paired.playerId2]
    # Create new columns 'p1' and 'p2' to keep the IDs of the two paired players
    paired['id'] = paired.apply(lambda row: tuple(sorted([row['matchId'], row['playerId1'], row['playerId2']])), axis=1)
    paired = paired.drop_duplicates(subset=['id'], keep='first')
    paired = paired.drop(['id'], axis=1)
    paired['p1'] = np.where(paired.playerId1 < paired.playerId2, paired.playerId1, paired.playerId2)
    paired['p2'] = np.where(paired.playerId2 > paired.playerId1, paired.playerId2, paired.playerId1)

    # Set the 'minutes' column to the smaller of minutes1 and minutes2
    paired['minutes'] = np.where(paired.minutes1 < paired.minutes2, paired.minutes1, paired.minutes2)
    # Keep only the relevant columns in the resulting DataFrame
    paired = paired[['teamId', 'matchId','p1', 'p2', 'minutes']]
    # Remove any duplicate rows in the DataFrame
    paired = paired.drop_duplicates()
    # Group the DataFrame by teamId, p1, and p2, and sum the minutes column
    paired = paired.groupby(['teamId', 'p1','p2'], as_index=False )['minutes'].sum()
    return paired


def compute_normalized_values(df_joi_game, df_jdi_game, df_pairwise_time):
    # Aggregate sum of 'joi' for each player pair in each team for the season
    df_joi_season = df_joi_game.groupby(['p1', 'p2', 'teamId'], as_index = False).agg({
                                                                                     'joi': 'sum',
                                                                                     'goals': 'sum',
                                                                                     'assists': 'sum',
                                                                                     'second_assists': 'sum'
                                                                                     })
    # Aggregate sum of 'jdi' for each player pair in each team for the season
    df_jdi_season = df_jdi_game.groupby(['p1', 'p2', 'teamId'], as_index = False)['jdi'].sum()
    # Merge the two dataframes based on the player pairs
    df_merged = pd.merge(df_joi_season, df_jdi_season, on=['p1', 'p2'])
    df_merged = df_merged.merge(df_pairwise_time, on=['p1', 'p2'])

    # Compute the normalized value of 'joi' and 'jdi' by dividing them by the norm90 value
    df_merged['winners90'] =  (df_merged.assists + df_merged.goals) * 90/df_merged.minutes
    df_merged['df_joi90'] = (df_merged.joi * 90/df_merged.minutes) #* df_merged.winners90
    df_merged['df_jdi90'] = df_merged.jdi * 90/df_merged.minutes
    df_merged = df_merged[['p1', 'p2', 'teamId', 'joi', 'jdi', 'minutes', 'df_jdi90', 'df_joi90', 'winners90']]

    # Filter out player pairs who have played less than 300 minutes together
    df_merged = df_merged.query('minutes >= 500')
    return df_merged

def get_TVP(df_vaep, squad, stamps):
    # Select only necessary columns from 'stamps' dataframe
    stamps = stamps[['eventId', 'matchPeriod', 'minute', 'second', 'matchTimestamp']]
    # Merge 'df_vaep' and 'stamps' dataframes based on 'eventId'
    df_vaep = df_vaep.merge(stamps, on=('eventId'))
    # Convert 'minutes' column to integer type
    squad['minutes_passed'] = squad['minutes'].astype(int)
    # Calculate 'seconds_passed_within_minute' column based on 'minutes_passed'
    squad['seconds_passed_within_minute'] = ((squad['minutes'] - squad['minutes_passed']) * 60).astype(float)

    player_TVG = []
    # Loop through each row of 'squad' dataframe
    for i, row in squad.iterrows():
        # Select rows from 'df_vaep' dataframe based on match and team ID
        g_vaep = df_vaep.loc[(df_vaep['matchId'] == row['matchId']) & (df_vaep['teamId'] == row['teamId'])]
        # Select only rows where the event happened before or within the current minute
        g_vaep = g_vaep.loc[(g_vaep['minute'] < (row['minutes_passed'] +1)) & ((g_vaep['minute'] < row['minutes_passed']) | (g_vaep['minute'] == row['minutes_passed']) & (g_vaep['second'] <= row['seconds_passed_within_minute']))]
        # Append a list of values to 'player_TVG' list
        player_TVG.append([row.playerId, row.matchId, row.teamId, row.minutes_passed, row.seconds_passed_within_minute, sum(g_vaep['sumVaep'])])
    # Create a new dataframe 'frame' with columns and data from 'player_TVG'
    frame = pd.DataFrame(data=player_TVG, columns=['playerId', 'matchId', 'teamId', 'minutes_played', 'seconds_in_minute', 'in_game_team_vaep'])
    # Group 'frame' dataframe by 'teamId' and 'playerId' and calculate mean of 'in_game_team_vaep' for each group
    team_vaep_p_in_game = frame.groupby(['teamId', 'playerId'], as_index=False).agg({'minutes_played': 'sum', 'seconds_in_minute': 'sum', 'in_game_team_vaep': 'sum'})
    team_vaep_p_in_game['seconds_to_minutes'] = team_vaep_p_in_game['seconds_in_minute'] / 60
    team_vaep_p_in_game['total_minutes'] = team_vaep_p_in_game['minutes_played'] + team_vaep_p_in_game['seconds_to_minutes']
    team_vaep_p_in_game['in_game_team_vape_per_90'] = team_vaep_p_in_game['in_game_team_vaep'] / (team_vaep_p_in_game['total_minutes'] / 90)
    return team_vaep_p_in_game






def prepare_for_scaling(df, squad, stamps, match_duration):
    tvp = get_TVP(df, squad, stamps)
    league_avg_vaep_per_90 = get_league_vaep(df, match_duration)
    tvp['league_vaep_per_90'] = league_avg_vaep_per_90  # Place league vaep as column
    tvp['factor'] = tvp['in_game_team_vape_per_90'] / tvp['league_vaep_per_90']  # Compute global factor values

    merged = tvp.merge(tvp, on='teamId', suffixes=('1', '2'))  # Pair players
    merged = merged[['teamId', 'playerId1', 'playerId2', 'factor1', 'factor2']]
    # Filter columns for pairings of players with themselves and establish an order
    # by having the player with the smallest id in the p1 column
    merged = merged[(merged.playerId1 != merged.playerId2) & (merged.playerId1 < merged.playerId2)]
    merged = merged.rename(columns={'playerId1': 'p1', 'playerId2': 'p2'})  # Rename player columns
    merged = merged.sort_values(by=['p1', 'p2', 'teamId'])  # Sort dataframe in ascending order(default)
    merged['combined_factor'] = merged['factor1'] + merged['factor2']
    return merged



#def scale()

def add_pos(df, df_players):
    df_player_1_added = pd.merge(df, df_players, left_on='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on='p2', right_on="playerId")
    cols = df.columns
    columns = np.concatenate((['role_name_x', 'role_name_y'], cols))
    df_filtered = df_player_2_added[columns]
    return df_filtered


def get_weighted_chemistry(df_chem):
    df_chem['chemistry'] = np.where((df_chem.role_name_x == 'Forward') & (df_chem.role_name_y == 'Forward'),
                                    ((df_chem.df_joi90 * 0.75) + (df_chem.df_jdi90 * 0.25)) * (df_chem.combined_factor),
                                    np.where((df_chem.role_name_x == 'Defender') & (df_chem.role_name_y == 'Defender'),
                                             ((df_chem.df_joi90 * 0.25) + (df_chem.df_jdi90 * 0.75)) * (df_chem.combined_factor),
                                             np.where(((df_chem.role_name_x == 'Midfielder') & (df_chem.role_name_y == 'Forward'))
                                             | ((df_chem.role_name_x == 'Forward') & ( df_chem.role_name_y == 'Midfielder')),
                                                      ((df_chem.df_joi90 * 0.60) + (df_chem.df_jdi90 * 0.40)) * (df_chem.combined_factor),
                                                      np.where(((df_chem.role_name_x == 'Midfielder') & (df_chem.role_name_y == 'Defender')) |
                                                               ((df_chem.role_name_x == 'Defender') & (df_chem.role_name_y == 'Midfielder')),
                                                               ((df_chem.df_joi90 * 0.40) + (df_chem.df_jdi90 * 0.60)) * (df_chem.combined_factor),
                                                               ((df_chem.df_joi90 * 0.50) + ( df_chem.df_jdi90 * 0.50)) * (df_chem.combined_factor)
                                                               )
                                                      )
                                             )
                                    )
    return df_chem

def get_weighted_chemistry_t(df_chem):
    df_chem['chemistry'] = np.where((df_chem.role_name_x == 'Forward') & (df_chem.role_name_y == 'Forward'),
                                    ((df_chem.df_joi90 * 0.75) + (df_chem.df_jdi90 * 0.25)) * (df_chem.factor1 + df_chem.factor2),
                                    np.where((df_chem.role_name_x == 'Defender') & (df_chem.role_name_y == 'Defender'),
                                             ((df_chem.df_joi90 * 0.25) + (df_chem.df_jdi90 * 0.75)) * (df_chem.factor1 + df_chem.factor2),
                                             np.where(((df_chem.role_name_x == 'Midfielder') & (
                                                         df_chem.role_name_y == 'Forward')) |
                                                      ((df_chem.role_name_x == 'Forward') & (df_chem.role_name_y == 'Midfielder')),
                                                      ((df_chem.df_joi90 * 0.60) + (df_chem.df_jdi90 * 0.40)) * (df_chem.factor1 + df_chem.factor2),
                                                      np.where(((df_chem.role_name_x == 'Midfielder') & (
                                                                  df_chem.role_name_y == 'Defender')) |
                                                               ((df_chem.role_name_x == 'Defender') & ( df_chem.role_name_y == 'Midfielder')),
                                                               ((df_chem.df_joi90 * 0.40) + (df_chem.df_jdi90 * 0.60)) * (df_chem.factor1 + df_chem.factor2),
                                                               ((df_chem.df_joi90 * 0.50) + (df_chem.df_jdi90 * 0.50)) * (df_chem.factor1 + df_chem.factor2)
                                                               )
                                                      )
                                             )
                                    )
    return df_chem


def get_chemistry(df_factor_values, df_joi90_jdi90, df_players_teams):
    dfm_id = df_factor_values[['p1', 'p2', 'teamId', 'seasonId']]  # Extract columns that should not be scaled
    mask_m = ~df_factor_values.columns.isin(['p1', 'p2', 'teamId', 'seasonId'])  # Extract column names for scaling
    df_scale_m = df_factor_values.loc[:, mask_m]
    scale1 = MinMaxScaler(feature_range=(0.8, 1.2 ))
    df_scale_m[df_scale_m.columns] = scale1.fit_transform(df_scale_m[df_scale_m.columns])  # Perform min/max scaling
    df_factors_scaled = pd.concat([dfm_id.reset_index(drop=True), df_scale_m.reset_index(drop=True)], axis=1)

    # Merge with joi and jdi dataframes
    dfc = pd.merge(df_joi90_jdi90, df_factors_scaled, on=['teamId', 'p1', 'p2', 'seasonId'], how='inner')
    dfc = dfc.sort_values(by=['p1', 'p2', 'teamId'])  # Sort dataframe in ascending order(default)
    df_id = dfc[['p1', 'p2', 'teamId','seasonId', 'minutes', 'factor1', 'factor2', 'combined_factor']]  # Extract columns that should not be scaled
    mask = ~dfc.columns.isin( ['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2', 'combined_factor'])  # Extract column names for scaling
    df_scale = dfc.loc[:, mask]  # Extract columns for scaling

    scale2 = MinMaxScaler()  # Initiate sclaing instance
    df_scale[df_scale.columns] = scale2.fit_transform(df_scale[df_scale.columns])  # Perform min/max scaling
    # Re-establosh dataframe with id's
    df_chemistry = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)

    df_chemistry_pos = add_pos(df_chemistry, df_players_teams)
    # Compute chemistry columns based on formula
    df_chem_final = get_weighted_chemistry(df_chemistry_pos)

    return df_chem_final

def get_chemistry_t(df_factor_values, df_joi90_jdi90, df_players_teams):
    dfm_id = df_factor_values[['p1', 'p2', 'teamId', 'seasonId']]  # Extract columns that should not be scaled
    mask_m = ~df_factor_values.columns.isin(['p1', 'p2', 'teamId', 'seasonId'])  # Extract column names for scaling
    df_scale_m = df_factor_values.loc[:, mask_m]
    scale1 = MinMaxScaler(feature_range=(0.9, 1.1 ))
    df_scale_m[df_scale_m.columns] = scale1.fit_transform(df_scale_m[df_scale_m.columns])  # Perform min/max scaling
    df_factors_scaled = pd.concat([dfm_id.reset_index(drop=True), df_scale_m.reset_index(drop=True)], axis=1)

    # Merge with joi and jdi dataframes
    dfc = pd.merge(df_joi90_jdi90, df_factors_scaled, on=['teamId', 'p1', 'p2', 'seasonId'], how='inner')
    dfc = dfc.sort_values(by=['p1', 'p2', 'teamId'])  # Sort dataframe in ascending order(default)
    df_id = dfc[['p1', 'p2', 'teamId','seasonId', 'minutes', 'factor1', 'factor2']]  # Extract columns that should not be scaled
    mask = ~dfc.columns.isin( ['p1', 'p2', 'teamId', 'seasonId', 'minutes', 'factor1', 'factor2'])  # Extract column names for scaling
    df_scale = dfc.loc[:, mask]  # Extract columns for scaling

    scale2 = MinMaxScaler()  # Initiate sclaing instance
    df_scale[df_scale.columns] = scale2.fit_transform(df_scale[df_scale.columns])  # Perform min/max scaling
    # Re-establosh dataframe with id's
    df_chemistry = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)

    df_chemistry_pos = add_pos(df_chemistry, df_players_teams)
    # Compute chemistry columns based on formula
    df_chem_final = get_weighted_chemistry_t(df_chemistry_pos)

    return df_chem_final


def get_overview_frame(df_chem, df_players):
    df_player_1_added = pd.merge(df_chem, df_players, left_on ='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on ='p2', right_on="playerId")
    df_filtered = df_player_2_added[['p1','p2', 'seasonId', 'shortName_x', 'shortName_y', 'minutes', 'teamId_x', 'role_name_x', 'role_name_y', 'areaName_x', 'areaName_y', 'df_jdi90', 'df_joi90','chemistry']]
    df_teams = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Teams]", db_name = 'Scouting_Raw')
    df_filtered = df_filtered.merge(df_teams[['teamId', 'name']], left_on="teamId_x", right_on="teamId")
    return df_filtered

def get_overview_frame_fac(df_chem, df_players):
    df_player_1_added = pd.merge(df_chem, df_players, left_on ='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on ='p2', right_on="playerId")
    df_filtered = df_player_2_added[['p1','p2', 'seasonId', 'shortName_x', 'shortName_y', 'minutes', 'teamId_x', 'role_name_x', 'role_name_y', 'areaName_x', 'areaName_y', 'factor1', 'factor2', 'df_jdi90', 'df_joi90','chemistry']]
    df_teams = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Teams]", db_name = 'Scouting_Raw')
    df_filtered = df_filtered.merge(df_teams[['teamId', 'name']], left_on="teamId_x", right_on="teamId")
    df1 = df_filtered[['p1', 'shortName_x', 'role_name_x', 'areaName_x', 'factor1', 'df_jdi90', 'df_joi90', 'chemistry']].rename(
        columns={'p1': 'playerId', 'shortName_x': 'shortName', 'role_name_x': 'role_name','factor1':'factor', 'areaName_x': 'areaName'})
    df1 = df1.drop_duplicates()
    df2 = df_filtered[['p2', 'shortName_y', 'role_name_y', 'areaName_y', 'factor2', 'df_jdi90', 'df_joi90', 'chemistry']].rename(
        columns={'p2': 'playerId', 'shortName_y': 'shortName', 'role_name_y': 'role_name', 'factor2':'factor', 'areaName_y': 'areaName'})
    df2 = df2.drop_duplicates()
    players_and_chemistry = pd.concat([df1, df2])

    return players_and_chemistry

def get_overview_frame_jdi(df_chem, df_players):
    df_player_1_added = pd.merge(df_chem, df_players, left_on ='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on ='p2', right_on="playerId")
    df_filtered = df_player_2_added[['p1','p2', 'shortName_x', 'shortName_y', 'teamId_x', 'role_name_x', 'role_name_y', 'areaName_x', 'areaName_y', 'seasonId', 'jdi']]
    df_teams = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Teams]", db_name = 'Scouting_Raw')
    df_filtered = df_filtered.merge(df_teams[['teamId', 'name']], left_on="teamId_x", right_on="teamId")
    return df_filtered

def get_overview_frame_jdi_oi(jdi_frame_t, df_players):
    df_player_1_added = pd.merge(jdi_frame_t, df_players, left_on ='p1', right_on="playerId")
    df_player_2_added = pd.merge(df_player_1_added, df_players, left_on ='p2', right_on="playerId")
    cols = jdi_frame_t.columns
    columns = np.concatenate(([ 'shortName_x', 'shortName_y', 'role_name_x', 'role_name_y'], cols))

    df_filtered = df_player_2_added[columns]
    #df_filtered = df_filtered.merge(share, left_on=['matchId','p2'], right_on=['matchId', 'playerId'])
    #df_filtered = df_player_2_added[['p1','p2', 'shortName_x', 'shortName_y', 'teamId_x', 'role_name_x', 'role_name_y', 'areaName_x', 'areaName_y', 'jdi']]
    #df_teams = load_db_to_pd(sql_query="SELECT * FROM [Scouting_Raw].[dbo].[Wyscout_Teams]", db_name = 'Scouting_Raw')
    #df_filtered = df_filtered.merge(df_teams[['teamId', 'name']], left_on="teamId_x", right_on="teamId")
    return df_filtered

def get_age(birthdate):
    today = date.today()
    birthdate_formatted = datetime.strptime(birthdate, '%Y-%m-%d').date()
    age = today.year - birthdate_formatted.year - ((today.month, today.day) < (birthdate_formatted.month, birthdate_formatted.day))
    return age


def restructure_matches(df):
    df1 = df[['seasonId', 'competitionId', 'home_teamId']]
    df2 = df[['seasonId', 'competitionId', 'away_teamId']]
    df1 = df1.rename(columns={'home_teamId': 'teamId'})
    df2 = df1.rename(columns={'away_teamId': 'teamId'})
    df_all = pd.concat([df1, df2])
    df_all = df_all.drop_duplicates()
    return df_all

def find_countries(df):
    fromCountry_x = df['areaName_x'].unique()
    fromCountry_y = df['areaName_y'].unique()
    all = fromCountry_x + fromCountry_y
    all_distinct = all.uniwue()




def get_league_vaep(df, match_duration):
    vaep = df['sumVaep'].sum()
    total_minutes = match_duration['minutes'].sum()
    league_vaep_per_90 = vaep/(total_minutes/90)
    return league_vaep_per_90

def get_adaptability(df):
    print('tester')

def get_new_arrivals(df_transfers, df_chem):
    df_transfers = df_transfers[df_transfers['startDate'] != '0000-00-00']
    df_transfers['date'] = df_transfers.apply(lambda row: (datetime.strptime(row['startDate'], '%Y-%m-%d').date()).year, axis = 1)
    df_transfers = df_transfers[(df_transfers['date'] > 2020) & (df_transfers['date'] < 2023)]
    df_arrivals_1 = (df_transfers.merge(df_chem, left_on='playerId', right_on='p1')).rename(columns = {'playerId': 'arrival'})
    df_arrivals_2 = df_transfers.merge(df_chem, left_on='playerId', right_on='p2' ).rename(columns = {'playerId': 'arrival'})
    df_arrivals = pd.concat([df_arrivals_1, df_arrivals_2])

    return df_arrivals

def get_chem_profficiency(arrivals, chemistries):
    test_lists = []
    arrival_ids = arrivals['arrival'].unique()
    training_set = chemistries[~((chemistries['p1'].isin(arrival_ids)) | (chemistries['p2'].isin(arrival_ids)))]
    unique_training_ids = pd.concat([training_set['p1'], training_set['p2']]).unique()
    training_lists = []
    for id in arrival_ids:
        df_a = chemistries[(chemistries['p1'] == id) | (chemistries['p2'] == id)]
        for i, row in df_a.iterrows():
            adap_rows = df_a[df_a.index != i]
            adpt_p = adap_rows['chemistry'].mean()
            new_row = [id, adpt_p, row['chemistry'], row['seasonId'],row['teamId'],row['df_joi90'], row['df_jdi90'], row['p1'] if id != row['p1'] else row['p2']]
            test_lists.append(new_row)
    for e_id in unique_training_ids:
        df_a_2 = training_set[(training_set['p1'] == e_id) | (training_set['p2'] == e_id)]
        e_id_chem_ability = df_a_2['chemistry'].mean()
        training_lists.append([e_id, e_id_chem_ability])

    test_set = pd.DataFrame(test_lists, columns=['new_player', 'chem_coef', 'chemistry','seasonId','teamId','df_joi90', 'df_jdi90', 'existing_player'])
    ability_set = pd.DataFrame(training_lists, columns=['player', 'chem_coef'])
    test_set_abilities = test_set.merge(ability_set, left_on='existing_player', right_on='player')
    training_set_final_1 = training_set.merge(ability_set, left_on='p1', right_on='player')
    training_set_final_2 = training_set_final_1.merge(ability_set, left_on='p2', right_on='player')
    return training_set_final_2, test_set_abilities








