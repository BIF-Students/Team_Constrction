import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def find_zones_and_vaep(df):
    # create dummy variables for the zone column
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    # multiply the dummy variables by the sumVaep column
    zone_vaep = zone_dummies.mul(df['sumVaep'], axis=0)

    # concatenate the original dataframe with the zone_vaep dataframe
    df = pd.concat([df, zone_vaep], axis=1)
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

def generate_joi (df):
    #Copy dataframe
    df_events_related_ids = df

    #filter for relevant columns
    df_events_related_ids  =df_events_related_ids[['eventId', 'matchId', 'playerId', 'typePrimary', 'teamId', 'sumVaep']]

    #change naming convention
    df_events_related_ids = df_events_related_ids.rename(columns = {'eventId': 'relatedEventId', 'matchId': 'matchId_2', 'playerId': 'playerId_2', 'teamId' : 'teamId_2', 'typePrimary': 'related_event', 'sumVaep': 'sumVaep_2'})
    df = df.rename(columns = {'matchId': 'matchId_1', 'playerId': 'playerId_1', 'teamId' : 'teamId_1', 'sumVaep': 'sumVaep_1'})

    #Merge on relate ids to obtain a dataframe with atributes of main eventId and relatedEventId in same observation
    joined_df = pd.merge(df, df_events_related_ids, how = 'left', on='relatedEventId')

    #Remove missing values
    joined_df = joined_df[ (joined_df['playerId_1'].notna()) & (joined_df['playerId_2'].notna()) & (joined_df['teamId_1'].notna())& (joined_df['teamId_2'].notna()) & (joined_df['matchId_1'].notna()) & (joined_df['matchId_2'].notna())]

    #Remplace missing vaep values with zero-values
    joined_df['sumVaep_1'] = joined_df['sumVaep_1'].fillna(0)
    joined_df['sumVaep_2'] = joined_df['sumVaep_2'].fillna(0)

    '''
    Make a filter that extracts only observatoins where;
    1: The same team is represented in both main eventId and relatedEventID
    2: The same match is represented in both main eventId and relatedEventID
    3: It is not a sequence of events produced by the same player
    4 & 5: Not relevant events ar removed from both typePrimary and related events
    '''
    joined_df_filtered = joined_df[(joined_df.teamId_1 == joined_df.teamId_2)
                                   & (joined_df.matchId_1 == joined_df.matchId_2)
                                   & (joined_df.playerId_1 != joined_df.playerId_2)
                                   & (~joined_df.typePrimary.isin(['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))
                                   & (~joined_df.related_event.isin(['interception', 'clearence', 'goal_kick', 'infraction', 'offside', 'shot_against']))]

    #Order player ids such that the player with the lowest Id will be represented in the p1 attribute
    joined_df_filtered['p1'] = np.where(joined_df_filtered['playerId_1'] > joined_df_filtered['playerId_2'], joined_df_filtered['playerId_2'], joined_df_filtered['playerId_1'])
    joined_df_filtered['p2'] = np.where(joined_df_filtered['playerId_1'] == joined_df_filtered['p1'],joined_df_filtered['playerId_2'], joined_df_filtered['playerId_1'])

    # Drop unnecessary columns
    joined_df_filtered = joined_df_filtered.drop(['playerId_1', 'playerId_2'], axis=1)
    joined_df_filtered['joi'] = joined_df_filtered['sumVaep_1'] + joined_df_filtered['sumVaep_2']

    #Compute jois as a sum of the sumVaep values related to the main event and related event
    joi_df = joined_df_filtered.groupby(['matchId_1', 'matchId_2', 'p1', 'p2', 'teamId_1', 'teamId_2'], as_index=False)[
        'joi'].sum()
    return  joi_df


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



def compute_jdi (df):
    #Compute jdi per zone per pair of players
    df['jdi_zone_1'] = np.where(df.zone_1_t > 3, df.zone_1_net_oi * df.zone_1_imp * (1/df.distance), 0)
    df['jdi_zone_2'] = np.where(df.zone_2_t > 3, df.zone_2_net_oi * df.zone_2_imp * (1/df.distance), 0)
    df['jdi_zone_3'] = np.where(df.zone_3_t > 3, df.zone_3_net_oi * df.zone_3_imp * (1/df.distance), 0)
    df['jdi_zone_4'] = np.where(df.zone_4_t > 3, df.zone_4_net_oi * df.zone_4_imp * (1/df.distance), 0)
    df['jdi_zone_5'] = np.where(df.zone_5_t > 3, df.zone_5_net_oi * df.zone_5_imp * (1/df.distance), 0)
    df['jdi_zone_6'] = np.where(df.zone_6_t > 3, df.zone_6_net_oi * df.zone_6_imp * (1/df.distance), 0)

    #Compute total jdi across zones for pairs of players
    df['jdi'] = df.jdi_zone_1 + df.jdi_zone_2 + df.jdi_zone_3 + df.jdi_zone_4 + df.jdi_zone_5 + df.jdi_zone_6
    return df


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

#Method for computing running aver vaep using observatoins as input and with a label input
def compute_running_avg_team_vaep (row, label):
    #If it is the first game of the season, we use the current cumulative sum
    if row['games_played'] == 0:
        return row[label]
    else: # Else we divide the current sum with the amout of games played
        return row[label] / (row['games_played'] +1)

def team_vaep_game(df):
    # Group data by matchId and teamId, summing the values of the zones
    df = df.groupby(['matchId', 'teamId'])[
        'zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6'].sum().reset_index()

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


def compute_pairwise_playing_time (df):
    # Merge the DataFrame with itself based on the matchId and teamId columns
    paired = df.merge(df, on=(['matchId', 'teamId']), how='inner', suffixes=('1', '2'))

    # Remove any rows where playerId1 is equal to playerId2
    paired = paired[paired.playerId1 != paired.playerId2]
    # Create new columns 'p1' and 'p2' to keep the IDs of the two paired players

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

    # Compute the normalized playing time for each pair by dividing the total minutes played by 90
    paired['norm90'] = paired.minutes/90
    return paired


def compute_normalized_values(df_joi_game, df_jdi_game, df_pairwise_time):
    # Aggregate sum of 'joi' for each player pair in each team for the season
    df_joi_season = df_joi_game.groupby(['p1', 'p2', 'teamId_1'], as_index = False)['joi'].sum()

    # Aggregate sum of 'jdi' for each player pair in each team for the season
    df_jdi_season = df_jdi_game.groupby(['p1', 'p2', 'teamId'], as_index = False)['jdi'].sum()

    # Merge the two dataframes based on the player pairs
    df_merged = (pd.merge(df_joi_season, df_jdi_season, on=(['p1', 'p2']))).merge(df_pairwise_time, on= (['p1', 'p2']))

    # Compute the normalized value of 'joi' and 'jdi' by dividing them by the norm90 value
    df_merged['df_joi90'] = df_merged.joi / df_merged.norm90
    df_merged['df_jdi90'] = df_merged.jdi / df_merged.norm90

    # Rename the column 'teamId_1' to 'teamId' and select relevant columns
    df_merged = df_merged.rename(columns={'teamId_1': 'teamId'})
    df_merged = df_merged[['p1', 'p2', 'teamId', 'joi', 'jdi', 'minutes', 'norm90', 'df_jdi90', 'df_joi90']]

    # Filter out player pairs who have played less than 300 minutes together
    df_merged = df_merged.query('minutes > 300')
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
    team_vaep_p_in_game = frame.groupby(['teamId', 'playerId'], as_index=False)['in_game_team_vaep'].mean()

    # Return the new dataframe
    return team_vaep_p_in_game


def get_league_vaep(df):
    df = df.groupby(['matchId'], as_index =False)['sumVaep'].sum()#Compute vaep values for eacg game in a season
    league_vaep =  df['sumVaep'].mean() # compute mean of match vaep values
    return league_vaep #Return league vaep


def compute_chemistry(df, squad, stamps, df_joi90_jdi90):

    '''
    tvp is a dataframe containing;
    The average match vaep for a team in a season.
    The match vaep for a team related to a player, only inspects
    the team vaep for the minutes a player was on the pitch
    '''
    tvp = get_TVP(df, squad, stamps)
    league_avg_vaep = get_league_vaep(df) # Compute league average vaep per game
    tvp['league_vaep'] = league_avg_vaep # Place league vaep as column
    tvp['factor'] = tvp['in_game_team_vaep'] / tvp['league_vaep'] # Compute global factor values
    merged = tvp.merge(tvp, on='teamId', suffixes=('1', '2')) # Pair players

    #Extract columns of interest
    merged = merged[['teamId', 'playerId1', 'playerId2', 'factor1', 'factor2']]
    '''
    Filter columns for pairings of players with themselves and establish an order
    by having the player with the smallest id in the p1 column
    '''
    merged = merged[(merged.playerId1 != merged.playerId2) & (merged.playerId1 < merged.playerId2)]
    merged = merged.rename(columns={'playerId1': 'p1', 'playerId2': 'p2'}) #Rename player columns

    #Merge with joi and jdi dataframes
    dfc = pd.merge(df_joi90_jdi90, merged, on=['teamId', 'p1', 'p2'], how='inner')
    dfc = dfc.sort_values(by=['p1', 'p2', 'teamId']) #Sort dataframe in ascending order(default)

    df_id = dfc[['p1', 'p2', 'teamId']] #Extract columns that should not be scaled
    mask = ~dfc.columns.isin(['p1', 'p2', 'teamId'])# Extract column names for scaling
    df_scale = dfc.loc[:, mask] # Extract columns for scaling

    scale = MinMaxScaler() #Initiate sclaing instance
    df_scale[df_scale.columns] = scale.fit_transform(df_scale[df_scale.columns]) #Perform min/max scaling

    #Re-establosh dataframe with id's
    df_scaled = pd.concat([df_id.reset_index(drop=True), df_scale.reset_index(drop=True)], axis=1)

    #Compute chemistry columns based on formula
    df_chemistry = df_scaled.assign(chem1=(df_scaled.df_joi90 * df_scaled.df_jdi90 * df_scaled.factor1),
                                    chem2=(df_scaled.df_joi90 * df_scaled.df_jdi90 * df_scaled.factor2))
