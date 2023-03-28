import pandas as pd
import numpy as np
import math


def find_zones_and_vaep(df):
    # create dummy variables for the zone column
    zone_dummies = pd.get_dummies(df['zone'], prefix='zone')

    # multiply the dummy variables by the sumVaep column
    zone_vaep = zone_dummies.mul(df['sumVaep'], axis=0)

    # concatenate the original dataframe with the zone_vaep dataframe
    df = pd.concat([df, zone_vaep], axis=1)
    return df


def find_zone_chemistry(row):
    s = ""
    x = row['x']
    y = row['y']
    if x >= 0 and x < 50 and y >= 0 and y < 33:
        s = 1
    elif x >= 50 and x <= 100 and y >= 0 and y < 33:
        s = 4
    elif x >= 0 and x < 50 and y >= 33 and y < 66:
        s = 2
    elif x >= 50 and x <= 100 and y >= 33 and y < 66:
        s = 5
    elif x >= 0 and x < 50 and y >=66 and y <= 100:
        s = 3
    elif x >= 50 and x <= 100 and y >= 66 and y <= 100:
        s = 6
    else:
        s = 0
    return s

def get_average_positions(key_values, newDict):
    for key in key_values:
        newDict[key] = {'matchId': key_values[key]['matchId'],
                        'playerId': key_values[key]['playerId'],
                        'teamId': key_values[key]['teamId'],
                        'avg_x': sum(key_values[key]['x'])/len(key_values[key]['x']),
                        'avg_y': sum(key_values[key]['y'])/len(key_values[key]['y'])}
    return newDict

def allocate_position(row, matches_positions):
    x = row['x']
    y = row['y']
    mId = row['matchId']
    pId = row['playerId']
    tId = row['teamId']
    if (mId, pId) in matches_positions:
        matches_positions[(mId, pId)]['x'].append(x)
        matches_positions[(mId, pId)]['y'].append(y)
    else: matches_positions[(mId, pId)] = {'matchId': mId , 'teamId':tId,  'playerId': pId , 'x': [x], 'y': [y]}


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
    df_full = pd.merge(df_player, df_team, on= ['matchId', 'teamId'], how='inner')
    df_full['zone_1_imp'] = np.where(df_full.zone_1_pl > 0, df_full.zone_1_pl/df_full.zone_1_t,0)
    df_full['zone_2_imp'] = np.where(df_full.zone_2_pl > 0, df_full.zone_2_pl/df_full.zone_2_t,0)
    df_full['zone_3_imp'] = np.where(df_full.zone_3_pl > 0, df_full.zone_3_pl/df_full.zone_3_t,0)
    df_full['zone_4_imp'] = np.where(df_full.zone_4_pl > 0, df_full.zone_4_pl/df_full.zone_4_t,0)
    df_full['zone_5_imp'] = np.where(df_full.zone_5_pl > 0, df_full.zone_5_pl/df_full.zone_5_t,0)
    df_full['zone_6_imp'] = np.where(df_full.zone_6_pl > 0, df_full.zone_6_pl/df_full.zone_6_t,0)
    return df_full

def find_zones_and_counts_pl(df):
        zone_dummies = pd.get_dummies(df['zone'], prefix='zone')
        df = pd.concat([df, zone_dummies], axis=1)
        df = df.groupby([df.playerId, df.matchId, df.teamId], as_index=False).agg({
            'zone_1': 'sum',
            'zone_2': 'sum',
            'zone_3': 'sum',
            'zone_4': 'sum',
            'zone_5': 'sum',
            'zone_6': 'sum'})
        df = df.rename(columns={'zone_1': 'zone_1_pl',
                                'zone_2': 'zone_2_pl',
                                'zone_3': 'zone_3_pl',
                                'zone_4': 'zone_4_pl',
                                'zone_5': 'zone_5_pl',
                                'zone_6': 'zone_6_pl'
                                })
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

def compute_net_oi_game(df):
    copy = df
    copy['zone_1_net_oi'] = np.where(copy.games_played <= 3, (copy.zone_1_expected_vaep - copy.zone_1_prior_avg), (copy.zone_1_expected_vaep - copy.zone_1))
    copy['zone_2_net_oi'] = np.where(copy.games_played <= 3, (copy.zone_2_expected_vaep - copy.zone_2_prior_avg), (copy.zone_2_expected_vaep - copy.zone_2))
    copy['zone_3_net_oi'] = np.where(copy.games_played <= 3, (copy.zone_3_expected_vaep - copy.zone_3_prior_avg), (copy.zone_3_expected_vaep - copy.zone_3))
    copy['zone_4_net_oi'] = np.where(copy.games_played <= 3, (copy.zone_4_expected_vaep - copy.zone_4_prior_avg), (copy.zone_4_expected_vaep - copy.zone_4))
    copy['zone_5_net_oi'] = np.where(copy.games_played <= 3, (copy.zone_5_expected_vaep - copy.zone_5_prior_avg), (copy.zone_5_expected_vaep - copy.zone_5))
    copy['zone_6_net_oi'] = np.where(copy.games_played <= 3, (copy.zone_6_expected_vaep - copy.zone_6_prior_avg), (copy.zone_6_expected_vaep - copy.zone_6))

    return copy

def compute_running_avg_team_vaep (row, label):
    if row['games_played'] == 0:
        return row[label]
    else:
        return row[label] / (row['games_played'] +1)

def team_vaep_game(df):
    df = df.groupby(['matchId', 'teamId'])['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6'].sum().reset_index()
    df = df.sort_values(by=['teamId', 'matchId'])
    df[['zone_1_cumsum', 'zone_2_cumsum',
        'zone_3_cumsum', 'zone_4_cumsum',
        'zone_5_cumsum', 'zone_6_cumsum'
        ]] = df.groupby('teamId')[['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6']].cumsum()

    df['games_played'] = df.groupby(['teamId']).cumcount()
    df['zone_1_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_1_cumsum') ,axis = 1)
    df['zone_2_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_2_cumsum') ,axis = 1)
    df['zone_3_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_3_cumsum') ,axis = 1)
    df['zone_4_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_4_cumsum') ,axis = 1)
    df['zone_5_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_5_cumsum') ,axis = 1)
    df['zone_6_expected_vaep'] = df.apply(lambda row: compute_running_avg_team_vaep(row, 'zone_6_cumsum') ,axis = 1)
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
    df_share_dist = pd.merge(df_snoi_netoi, distances_df, left_on=(['matchId', 'teamId', 'playerId']), right_on=(['matchId', 'teamId', 'player1']), how='inner')
    df_share_dist_2 = pd.merge(df_snoi_netoi, distances_df, left_on=(['matchId', 'teamId', 'playerId']), right_on=(['matchId', 'teamId', 'player2']), how='inner')
    df_share_dist_final = pd.concat([df_share_dist, df_share_dist_2])
    return df_share_dist_final


def compute_pairwise_playing_time (df):
    paired = df.merge(df, on=(['matchId', 'teamId']), how='inner', suffixes=('1', '2'))
    paired = paired[paired.playerId1 != paired.playerId2]
    paired['p1'] = np.where(paired.playerId1 < paired.playerId2, paired.playerId1, paired.playerId2)
    paired['p2'] = np.where(paired.playerId2 > paired.playerId1, paired.playerId2, paired.playerId1)
    paired['minutes'] = np.where(paired.minutes1 < paired.minutes2, paired.minutes1, paired.minutes2)
    paired = paired[['teamId', 'matchId','p1', 'p2', 'minutes']]
    paired = paired.drop_duplicates()
    paired = paired.groupby(['teamId', 'p1','p2'], as_index=False )['minutes'].sum()
    paired['norm90'] = paired.minutes/90
    return paired


def compute_normalized_values(df_joi_game, df_jdi_game, df_pairwise_time):
    df_joi_season = df_joi_game.groupby(['p1', 'p2', 'teamId_1'], as_index = False)['joi'].sum()
    df_jdi_season = df_jdi_game.groupby(['p1', 'p2', 'teamId'], as_index = False)['jdi'].sum()
    df_merged = (pd.merge(df_joi_season, df_jdi_season, on=(['p1', 'p2']))).merge(df_pairwise_time, on= (['p1', 'p2']))
    df_merged['df_joi90'] = df_merged.joi / df_merged.norm90
    df_merged['df_jdi90'] = df_merged.jdi / df_merged.norm90
    df_merged = df_merged.rename(columns={'teamId_1': 'teamId'})
    df_merged = df_merged[['p1', 'p2', 'teamId', 'joi', 'jdi', 'minutes', 'norm90', 'df_jdi90', 'df_joi90']]
    df_merged = df_merged.query('minutes > 300')
    return df_merged

def get_TVP(df_vaep, squad, stamps):
    stamps = stamps[['eventId', 'matchPeriod', 'minute', 'second', 'matchTimestamp']]
    df_vaep = df_vaep.merge(stamps, on =('eventId'))
    squad['minutes_passed'] = squad['minutes'].astype(int)
    squad['seconds_passed_within_minute'] = ((squad['minutes'] - squad['minutes_passed']) * 60).astype(float)

    player_TVG = []
    for i, row in squad.iterrows():
        g_vaep = df_vaep.loc[(df_vaep['matchId'] == row['matchId']) & (df_vaep['teamId'] == row['teamId'])]
        g_vaep = g_vaep.loc[(g_vaep['minute'] < (row['minutes_passed'] +1)) & ((g_vaep['minute'] < row['minutes_passed']) | (g_vaep['minute'] == row['minutes_passed']) & (g_vaep['second'] <= row['seconds_passed_within_minute']))]
        player_TVG.append([row.playerId, row.matchId, row.teamId, row.minutes_passed, row.seconds_passed_within_minute, sum(g_vaep['sumVaep'])])
    frame = pd.DataFrame(data = player_TVG, columns=['playerId', 'matchId', 'teamId', 'minutes_played', 'seconds_in_minute', 'in_game_team_vaep'])

    team_vaep_p_in_game = frame.groupby(['teamId', 'playerId'], as_index = False)['in_game_team_vaep'].mean()

    return team_vaep_p_in_game, frame, df_vaep

def get_league_vaep(df):
    df = df.groupby(['matchId'], as_index =False)['sumVaep'].sum()
    league_vaep =  df['sumVaep'].mean()
    return league_vaep